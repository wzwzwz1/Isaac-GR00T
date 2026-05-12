# npu_prompt_flash_attention 在 DiT 上的实验记录

## 目标

将 GR00T DiT 的手工 SDPA（matmul+softmax+matmul）替换为 310P3 上唯一可用的融合 attention 算子 `npu_prompt_flash_attention`，降低 attention 延迟。

## 基线数据

| 指标 | 数值 |
|------|------|
| 模型 | GR00T-N1.6-3B FP16 |
| 芯片 | 1× Ascend 310P3 |
| action_horizon | 50 |
| action_dim | 128 |
| DiT 层数 | 32 (16 cross-attn + 16 self-attn) |
| 去噪步数 | 4 |
| Action Head 总延迟 | **439.5ms** |
| DiT 单步延迟 | **~103ms** |

### DiT 单步耗时分布

```
DiT 单步 103ms:
  ├─ 残差加法 + reshape/contiguous + 未分类:   38ms (37%)
  ├─ MLP (gate+up+down):                      18ms (17%)
  ├─ Attention compute (matmul+softmax+matmul): 17ms (16%)
  ├─ AdaNorm (SiLU+Linear投影+LayerNorm):      12ms (12%)
  ├─ QKV Projection + Out Projection:          11ms (11%)
  └─ LayerNorm + Dropout:                       7ms ( 7%)
```

Attention 计算本身仅占 DiT 单步的 **16%**（~17ms）。

### Attention 单次调用耗时分解

```
单次 cross-attention (Q=[B,32,17,48], KV=[B,32,256,48]):
  QKV 投影 (to_q, to_k, to_v):     ~0.25ms
  Reshape to BNSD:                 ~0.02ms
  Q·K^T (matmul):                  ~0.15ms  ← 硬件加速良好
  Softmax:                         ~0.02ms
  Attn·V (matmul):                 ~0.15ms
  Reshape back:                    ~0.02ms
  Output projection (to_out):      ~0.08ms
  Residual add:                    ~0.01ms
  ─────────────────────────────────────
  合计:                            ~0.70ms
```

310P3 的 cube 单元对 matmul 已有良好硬件加速，每次仅 0.15ms。

## 探索过程

### Phase 1: torch_npu 现有算子调研

| 算子 | 310P3 支持 | 原因 |
|------|:---:|------|
| `npu_fusion_attention` | ✗ | AscendC kernel 被 `#ifdef __DAV_C220_CUBE__` 守卫，仅限 910B |
| `npu_fused_infer_attention_score` | ✗ | 代码中显式空实现 `#if __CCE_AICORE__ == 310` |
| `npu_prompt_flash_attention` | ✓ | `tbe/kernel/ascend310p/ops_transformer/prompt_flash_attention/` 有预编译二进制 |

唯一可用的是 `npu_prompt_flash_attention`。

### Phase 2: 独立算子验证 (bench_pfa.py)

```
测试: 无 mask, Q=[1,32,17,48], KV=[1,32,256,48], fp16

  手工 SDPA:  0.432ms
  PFA:        0.375ms
  加速比:     1.15x
  精度:       max_diff = 0.000977 ✓
```

无 mask 时 PFA 可用且精度正确，但加速比仅 1.15x。

```
测试: 带 bool mask [B, S_kv], Qs=17, Kvs=256

  结果: FAILED
  错误: "attention mask must be NULL, when Qs,Kvs is unAlign 
         or Qs is not equal to Kvs, Qs = 17, Kvs = 256"
```

交叉注意力带 mask 时直接崩溃。两个条件同时触发：

| 条件 | 含义 | DiT 触发 |
|------|------|:---:|
| `Qs,Kvs is unAlign` | Q/K 序列长度不是对齐边界（64/128）的倍数 | ✓ S=17 |
| `Qs is not equal to Kvs` | Q 和 K 序列长度不同 | ✓ 17≠256 |

### Phase 3: Self-Attention mask 测试

```
测试: Q=K=V=[1,32,17,48], causal mask [1,1,17,17]

  结果: FAILED
  错误: "attention mask must be NULL, when Qs,Kvs is unAlign 
         or Qs is not equal to Kvs, Qs = 17, Kvs = 17"
```

即使 Qs==Kvs，只要 S=17 不是对齐边界，mask 也不可用。

```
测试: Q=K=V=[1,32,256,48], causal mask [1,1,256,256]

  结果: SUCCESS (S=256 对齐)
```

S=256 是 64 的整数倍，对齐条件满足，mask 正常工作。

### Phase 4: 预切片 Workaround

DiT 的 cross-attention mask 本质上是在 image token（前 128）和 text token（后 128）之间二选一。因此可以**预切片 K/V 到对应的 128 个 token**，然后调用无 mask 的 PFA。

```
测试: 预切片 K/V[:,:,0:128,:], 无 mask PFA

  精度:   max_diff = 0.000665 ✓
  手工 masked: 0.619ms
  PFA slice:   0.417ms
  加速比:      1.48x  (主要来自 KV 减半 256→128)
```

精度验证通过，相对加速比提升（因为 KV 减半）。

### Phase 5: 端到端 DiT 集成

在 `DiT.__init__()` 中自动应用 `AttnProcessorNPU310P3`，将所有 32 层 Attention 的处理器替换为 PFA：

```
PFA 处理器逻辑:
  ┌─ 无 mask ──→ PFA (直接调用)
  │
  ├─ 2D bool mask ──→ 预切片 K/V ──→ PFA (无 mask)
  │
  └─ 复杂 mask ──→ fallback 到手工 SDPA
```

**实测结果**:

| 方案 | Action Head 总延迟 | 变化 |
|------|-------------------|---|
| 手工 SDPA (基线) | **439.5ms** | — |
| PFA 处理器 | **794.4ms** | +81% |

**PFA 不仅没有加速，反而慢了 81%**。

## 失败根因分析

### 原因 1: kernel launch 开销大于计算节省

```
单次 attention 的计算时间:  0.15ms (Q·K^T) + 0.02ms (softmax) + 0.15ms (attn·V) = 0.32ms
PFA kernel launch 开销:    ~0.30ms (固定开销)
                         ─────
PFA 净收益:                ~0.02ms (可忽略)
```

对于 DiT 的小矩阵（D=48, Sq=17），matmul 在 310P 的 cube 单元上已被硬件高度加速。PFA 每次调用节省的计算量只有 0.02ms，而 PFA kernel 自身的启动开销约 0.3ms。

### 原因 2: 预切片的 CPU 同步开销

`_can_slice_mask()` 需要扫描 mask 找 True 区间：

```python
for i, val in enumerate(row.cpu().tolist()):  # CPU-GPU 同步!
```

每次 attention 调用都触发一次 NPU→CPU 数据传输和 Python 循环，这在 128 次 attention 调用中累积了大量延迟。

### 原因 3: kernel launch 数量不降反增

```
手工 SDPA (每层):
  to_q + to_k + to_v  →  3 kernel
  matmul + softmax + matmul  →  3 kernel
  to_out  →  1 kernel
  ─────
  7 kernel launches per attention call

PFA (每层):
  to_q + to_k + to_v  →  3 kernel
  切片操作  →  1 kernel
  PFA  →  1 kernel
  to_out  →  1 kernel
  ─────
  6 kernel launches per attention call

仅减少 1 个 kernel launch，但 PFA 本身的启动开销比单个 matmul 更大
```

### 原因 4: PFA 是为大模型设计的

310P3 的 PFA 实现针对的是大语言模型推理（seq≥128, D≥64, causal mask）。DiT 的 attention 规模太小，不在 PFA 的目标优化范围内。

### 为什么 910B 有完整实现而 310P3 没有

```
CANN 源码分析:
  fused_infer_attention_score_v3.cpp:83-84
    #if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    // EMPTY - no implementation
    #endif

  flash_attention_interface.cpp:47
    using ArchTag = Arch::AtlasA2;  // 硬编码为 910B 架构
```

这是 CANN 团队未投入资源为 310P3 实现完整 PFA 功能的结果，并非硬件限制。

## 核心洞察

### Attention compute 不是瓶颈

重新审视 DiT 单步的耗时分布：

```
真正的瓶颈:
  kernel launch 调度开销:    ~18-45ms  (18-44%)  ← 900+ 次 kernel launch
  残差加法 + tensor 整形:    ~38ms     (37%)
  MLP + Norm + Proj:        ~48ms     (47%)
  Attention compute:         ~17ms     (16%)   ← 不是瓶颈
```

即使写出完美的 Ascend C kernel，把 attention compute 从 17ms 压到 0ms，也只能省 17ms/步 × 4步 = **68ms**。总延迟从 439ms 降到 371ms，仅改善 15%。

### 正确的优化方向

| 优先级 | 方案 | 原理 | 预期效果 |
|--------|------|------|----------|
| **P0** | ATC 编译整个 DiT | 融合数百个小算子为少数大 kernel，消除 kernel launch 瓶颈 | 439→~280ms |
| **P1** | 减少去噪步数 (蒸馏) | 4步→2步直接减半 DiT 计算 | 439→~230ms |
| **P2** | 多芯片并行 | 8 芯片同时跑独立推理 | 8x 吞吐 |

## 保留的代码

`gr00t/model/modules/attention_processors.py` 保留了 PFA 处理器实现，通过环境变量控制：

```bash
# 默认不启用
python3 test_inference.py

# 强制启用 (仅 seq_len > 128 的大模型场景有用)
GR00T_ENABLE_NPU_PFA=1 python3 test_inference.py
```

`scripts/profiling/bench_pfa.py` 保留为独立的 attention 算子验证工具，可用于测试未来 CANN 版本是否修复了 310P3 的 PFA mask 支持。

## 关键文件

| 文件 | 说明 |
|------|------|
| `gr00t/model/modules/attention_processors.py` | PFA 处理器 (opt-in) |
| `gr00t/model/modules/dit.py` | DiT 定义，PFA 集成点 |
| `scripts/profiling/bench_pfa.py` | PFA vs 手工 SDPA 独立 benchmark |
| `profiling_results/pfa/` | PFA 端到端实测数据 |
| `profiling_results/baseline_final/` | 手工 SDPA 基线数据 |
