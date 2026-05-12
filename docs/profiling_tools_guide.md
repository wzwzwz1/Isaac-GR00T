# GR00T-N1.6 昇腾NPU 推理Profiling工具使用指南

## 实测数据 (2026-05-12, Ascend 310P3)

| 指标 | 数值 |
|------|------|
| 模型 | GR00T-N1.6-3B FP16 |
| action_horizon | 50 |
| action_dim | 128 |
| max_state_dim | 128 |
| NPU芯片 | 1× Ascend310P3 |

### 管线阶段耗时

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Encode features (VLLN + state encoder) | 0.8 ms | 0.2% |
| DiT Denoise Step 0 | 95.1 ms | 23.3% |
| DiT Denoise Step 1 | 98.5 ms | 24.1% |
| DiT Denoise Step 2 | 96.0 ms | 23.5% |
| DiT Denoise Step 3 | 95.4 ms | 23.3% |
| **Action Head 总计** | **408.9 ms** | 100% |
| 其中 Denoise (4步合计) | **385.0 ms** | **94%** |

### DiT 逐Block算子分布 (跨层平均, 按相对占比)

| 算子类别 | Cross-Attn (16层) | Self-Attn (16层) |
|----------|:---:|:---:|
| Attention计算 (QK+Softmax+AV) + Residual | **34.6%** | **37.5%** |
| AdaNorm (SiLU+Linear投影+Norm+Scale) | 33.5% | 29.0% |
| QKV投影 (to_q/to_k/to_v) | 8.7% | 6.6% |
| MLP up/gate/down | 13.1% | 13.7% |
| Out Projection | 3.8% | 6.6% |
| LayerNorm | 5.6% | 5.3% |
| Dropout | 0.7% | 1.3% |

**关键发现**: Attention计算 + AdaNorm 合计占每层耗时约 **68%**，是优化的核心目标。

---

## 工具清单

| 工具 | 文件 | 功能 | 自动持久化 |
|------|------|------|:---:|
| **一键Profiling** | `scripts/profiling/run_benchmark.py` | 运行全部测试并保存 | ✓ |
| 管线阶段Profiler | `scripts/profiling/latency_profiler.py` | 按推理阶段计时 | ✓ |
| 细粒度逐算子Profiler | `scripts/profiling/fine_grained_profiler.py` | 按Block×算子类别分解 | ✓ |
| Attention Benchmark | `scripts/profiling/attention_bench.py` | 对比attention实现延迟 | ✓ |
| 优化报告生成 | `scripts/profiling/optimization_report.py` | 生成HTML/文本报告 | ✓ |

> **所有工具默认自动持久化**: 结果保存到 `profiling_results/<时间戳>_<工具名>/` 目录，同时包含 JSON 和 TXT 格式。可通过 `--output_dir` 指定输出路径。

---

## 1. 管线阶段Profiler (`latency_profiler.py`)

### 用途

将推理管线分解为阶段性计时，快速定位哪个阶段最耗时。

### 使用方法

```bash
# 完整模式 (steps + hooks + dit_layers)
python3 scripts/profiling/latency_profiler.py \
    --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
    --mode all --num_warmup 5 --num_runs 20

# 仅阶段分解 (最快)
python3 scripts/profiling/latency_profiler.py \
    --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
    --mode steps --num_warmup 5 --num_runs 30 \
    --output_json profile_results.json
```

### 模式说明

- `--mode steps`: 按推理阶段计时（backbone / encode / denoise_step_N / total）
- `--mode hooks`: 为所有子模块安装钩子，输出模块级热力统计
- `--mode dit_layers`: 仅对DiT的32层block逐层计时
- `--mode all`: 全部运行

### 输出示例

```
================================================================================
  GR00T-N1.6 推理管线阶段耗时分解 (Ascend Ascend310P3)
================================================================================
  Stage                        Avg(ms)   Min(ms)   Max(ms)  P95(ms)      %
  ───────────────────────── ──────── ──────── ──────── ──────── ──────
  01_encode                       0.8      0.7      2.5      0.9    0.2%
  02_denoise_step_0              95.1     90.9    106.3    102.6   23.3%
  02_denoise_step_1              98.5     94.1    110.2    102.8   24.1%
  02_denoise_step_2              96.0     90.9    111.5    102.7   23.5%
  02_denoise_step_3              95.4     90.6    111.7    101.1   23.3%
  05_total                      408.9    391.2    464.2    433.6  100.0%
  ───────────────────────── ──────── ──────── ──────── ──────── ──────
  denoise ALL STEPS (total)     385.0ms  (94.0%)
```

### 作为Python库使用

```python
from scripts.profiling.latency_profiler import ManualStepProfiler

profiler = ManualStepProfiler(warmup=5)
for _ in range(5 + 30):
    inputs = build_inputs()
    profiler.run_one(model, inputs)
profiler.print_stats()
```

---

## 2. 细粒度逐算子Profiler (`fine_grained_profiler.py`)

### 用途

按每个Transformer Block × 每个算子类别（QKV Linear / Attention / MLP / LayerNorm / AdaNorm等）精确分解耗时。

### 使用方法

```bash
python3 scripts/profiling/fine_grained_profiler.py \
    --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
    --num_warmup 5 --num_runs 10
```

### 输出结构

```
╔══ DiT Step 0  [95.1 ms] ═══════════════════════════════════════
┌── Block 0   (cross-attn (text tokens))  [2.95 ms]
│   ├── AdaNorm                       0.89ms (30.2%) ██████
│   ├── Linear (QKV)                  0.41ms (13.9%) ██
│   ├── Attention·QK                  0.52ms (17.6%) ███
│   ├── Attention·Softmax             0.15ms ( 5.1%) █
│   ├── Attention·AV                  0.31ms (10.5%) ██
│   ├── Linear (Output)               0.10ms ( 3.4%) 
│   ├── LayerNorm                     0.12ms ( 4.1%) 
│   ├── Linear (MLP up+gate)          0.18ms ( 6.1%) █
│   ├── Linear (MLP down)             0.08ms ( 2.7%) 
│   └── Transpose/Reshape             0.19ms ( 6.4%) █
└───────────────────────────────────────────────────────
...
═══ 跨Step全局算子类别汇总 ═══
  Attention·QK                 66.56ms     ← 最大热点
  Linear (QKV)                 61.44ms
  AdaNorm                      55.12ms
  ...
```

### 注意事项

- **钩子开销**: 每个子模块的 pre/post hook 包含 `torch.npu.synchronize()` 调用，会显著增加总耗时。相对占比是准确的，但绝对值偏高。
- **"Other"类别**: 包含未分类的模块（如attention内部的matmul/softmax函数调用）。这部分可以通过函数monkey-patch进一步细分（见脚本中的 `patch_torch_ops()` 方法）。
- **CANN msprof**: 如需获取不含hook开销的硬件级精确计时，请使用第4节的msprof方案。

### 作为Python库使用

```python
from scripts.profiling.fine_grained_profiler import ActionHeadProfiler

profiler = ActionHeadProfiler(model, warmup=5, num_runs=10)
profiler.run(backbone_output, action_input)
profiler.cleanup()
```

---

## 3. Attention Benchmark (`attention_bench.py`)

### 用途

对比不同attention实现在NPU上的延迟和显存占用，评估优化空间。

### 使用方法

```bash
# 完整benchmark (需要NPU)
python3 scripts/profiling/attention_bench.py \
    --batch 1 --seq_len 51 --ctx_len 256 --d_k 48 \
    --num_heads 32 --device npu

# 仅显存分析 (可在任意环境运行)
python3 scripts/profiling/attention_bench.py --memory
```

### 输出示例

```
============================================================
  Attention Benchmark — NPU
  形状: Q=[1,32,51,48] K=[1,32,256,48] V=[1,32,256,48]
============================================================

  manual (matmul+softmax+matmul)          0.65ms  (p95=0.72ms)
  chunked manual (chunk=32)               0.58ms  (p95=0.63ms)
  torch SDPA                               ERROR: can not cast format...

  --- 相对加速比 (vs manual) ---
  chunked manual (chunk=32)                 1.12x

  --- 估算 DiT 总 Attention 开销 (32层 x 4步) ---
  单次 Cross-Attn: 0.65ms
  16层 Cross-Attn x 4步: 41.6ms
  DiT Attention 总计: 54.1ms

============================================================
  Attention 显存分析
============================================================
  Q 显存:          195.84 KB
  K 显存:          983.04 KB
  Attention Scores: 2.09 MB  ← 主要瓶颈! (seq=51 x ctx=256 x FP32)
  分块方案峰值节省:   1.09 MB (32%)
```

---

## 4. CANN msprof (硬件级精确Profiling)

### 用途

获取NPU硬件级别的精确计时，包括Python层不可见的算子（TransData, Cast, 各算子的L1/UB利用率等）。

### 使用方法

```bash
# 1. 准备 profiling 脚本
cat > /tmp/profile_run.py << 'EOF'
import torch, torch_npu, sys
sys.path.insert(0, '/home/wangzhe/Isaac-GR00T')
import gr00t.model
from transformers import AutoModel, BatchFeature
import torch.nn.functional as F

# ... (模型加载 + 推理代码, 同前) ...

# 运行30次推理
for _ in range(30):
    result = model.action_head.get_action(backbone_output, action_input)
EOF

# 2. msprof 采集
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msprof \
    --application="python3 /tmp/profile_run.py" \
    --output=./profiler_output \
    --aic-metrics=Memory,MemoryL0,MemoryUB,PipeUtilization \
    --aicpu-profiling=off \
    --sys-hardware-mem=on \
    --dvpp-profiling=off

# 3. 查看结果
# 打开 profiler_output/ 目录中的 .json 文件
# 或使用 CANN 自带的查看工具
```

### msprof可以回答的问题

- 哪个算子花费的L1/UB周期最多？
- TransData（格式转换）占总时间的比例？
- 是否有kernel launch间隔（bubble）？
- FP32 attention scores的HBM读写带宽利用率？

---

## 5. 优化报告生成 (`optimization_report.py`)

### 使用方法

```bash
# 文本报告
python3 scripts/profiling/optimization_report.py --format text

# HTML报告
python3 scripts/profiling/optimization_report.py \
    --format html --output optimization_report.html

# 结合实测数据
python3 scripts/profiling/latency_profiler.py \
    --mode steps --output_json profile.json
python3 scripts/profiling/optimization_report.py \
    --profile_json profile.json --output report.html
```

---

## 实测数据总结与优化优先级

基于 `2026-05-12` 在 Ascend310P3 上的实测数据:

```
Action Head总延迟: 408.9ms
  └─ DiT Denoising: 385.0ms (94%)
       ├─ Attention计算 (QK+Softmax+AV): ~35% → 约135ms
       ├─ AdaNorm (含内部Linear):       ~31% → 约119ms
       ├─ QKV+OutProj+MLP Linear:      ~24% → 约92ms
       └─ LayerNorm+Dropout+其他:       ~10% → 约38ms
```

### 优化优先级 (基于实测)

| 优先级 | 方案 | 目标算子 | 预期节省 | 难度 |
|--------|------|----------|----------|------|
| **P0** | Ascend C fused attention | Attention·QK+Softmax+AV (135ms) | ↓100ms | 高 |
| **P1** | ATC编译DiT (算子融合) | AdaNorm + Linear链 (~211ms) | ↓50-80ms | 中 |
| **P2** | F.silu+GELU融合 + contiguous消除 | 张量整形(~10ms) | ↓5-10ms | 低 |
| P3 | 减少去噪步数 4→2 (需蒸馏) | 全DiT (385ms) | ↓190ms | 中(需训练) |

---

## 目录结构

```
scripts/profiling/
├── latency_profiler.py        # 管线阶段Profiler
├── fine_grained_profiler.py   # 细粒度逐算子Profiler
├── attention_bench.py          # Attention实现Benchmark
└── optimization_report.py     # 优化报告生成

docs/
├── 部署文档.md                 # 昇腾NPU部署总结
├── ATC_开发方案.md             # ATC编译开发方案
├── profiling_tools_guide.md   # 本文件
└── optimization_report.html   # HTML优化报告
```
