# DrivingSDK GR00T-N1.6 NPU 推理实验记录

## 背景

昇腾官方项目 [DrivingSDK](https://gitcode.com/Ascend/DrivingSDK) 对 GR00T-N1.6 进行了 NPU 适配，通过算子替换补丁（`patch.py`）将模型中的关键算子映射到 `torch_npu` 融合算子。本次实验旨在评估这些补丁在当前环境（310P3 + CANN 9.0.0）下的实际效果。

## 环境

| 项目 | 版本/型号 |
|------|----------|
| 芯片 | 6× Ascend 310P3 (单卡测试) |
| CANN | 9.0.0 |
| torch | 2.8.0+cpu |
| torch_npu | 2.8.0.post4 |
| transformers | 4.57.6 |
| 模型 | GR00T-N1.6-3B (HuggingFace: nvidia/GR00T-N1.6-3B) |
| 驱动 SDK 源码 | `/home/wangzhe/DrivingSDK/model_examples/GR00T-N1.6/` |

## DrivingSDK 补丁清单

DrivingSDK 的 `patch.py` 对 GR00T-N1.6 应用 4 个补丁：

| # | 补丁 | 目标模块 | 原始算子 | 替换算子 |
|---|------|---------|---------|---------|
| 1 | `rmsnorm_patch` | Eagle Backbone (Qwen3) | `Qwen3RMSNorm.forward` (手写) | `torch_npu.npu_rms_norm` |
| 2 | `rope_patch` | Eagle Backbone (Qwen3) | `apply_rotary_pos_emb` (手写) | `torch_npu.npu_rotary_mul` |
| 3 | `flash_attn_func_patch` | Eagle Backbone + transformers | `flash_attn_func` / `flash_attn_varlen_func` | `torch_npu.npu_fusion_attention` |
| 4 | `attn_processor_patch` | DiT (diffusers) | `AttnProcessor2_0.__call__` (sdpa) | 修复 attention_mask shape + `npu_fusion_attention` |

### GR00T-N1.6 架构与补丁覆盖

```
输入(视频/状态/语言)
    │
    ▼
┌────────────────────────────┐
│  Backbone: Eagle VLM        │  补丁1 (RMSNorm)
│  nvidia/Eagle-Block2A-2B-v2 │  补丁2 (RoPE)
│  (SigLIP2 + Qwen3 LLM)     │  补丁3 (FlashAttn)
│                             │
│  输出: backbone_features   │
│  [B, seq_len, 2048]        │
└───────────┬────────────────┘
            │
            ▼
┌────────────────────────────┐
│  Action Head: DiT           │  补丁4 (AttnProcessor)
│  AlternateVLDiT, 32层       │
│  4步去噪                    │
│                             │
│  输出: action_pred [B,50,128]│
└────────────────────────────┘
```

## 实验过程

### Phase 1: 环境与兼容性修复

**问题 1**: transformers 4.57.6 移除/重命名了多个导出，导致 Eagle 模型的缓存在线处理器无法加载。

缺失的导出：
- `transformers.image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING`
- `transformers.image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS`
- `transformers.image_utils.VideoInput`
- `transformers.image_utils.make_batched_videos`

修复了 `~/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2/image_processing_eagle3_vl_fast.py`，添加兼容性填充。

**问题 2**: 即使修复上述导入，processor 的 `from_args_and_dict` 仍因 API 变更报错 `dictionary update sequence element #0 has length 6; 2 is required`。这是 transformers 4.57 中 `validate_init_kwargs` 返回值格式变更导致的深层不兼容。

**决策**: 绕过 processor，使用合成 tensor 直接构造 backbone 输出和 action head 输入，避免完整的 Eagle processor 加载链。

### Phase 2: FlashAttention 内核可用性测试

测试了 NPU 上所有可用的注意力 API：

| API | 支持 | 错误 |
|-----|:---:|------|
| `F.scaled_dot_product_attention` | ✗ | `can not cast format when output is input` |
| `npu_fusion_attention` (BSND/SBH/BSH) | ✗ | 161001: `Parse dynamic kernel config fail` |
| `npu_fusion_attention_v2` | ✗ | 161001 |
| `npu_fusion_attention_v3` | ✗ | 161001 |
| `npu_fused_infer_attention_score` | ✗ | 参数签名不匹配 |
| `npu_prompt_flash_attention` | ✓ | 仅无 mask / 对齐序列可用 |

`npu_prompt_flash_attention` 是唯一可用的融合注意力算子，但此前实验已证明其对 DiT 小矩阵场景效果不佳（详见 `npu_prompt_flash_attention_实验记录.md`）。

**根因**: 310P3 的 FlashAttentionScore 内核在 CANN 9.0.0 中未编译/未安装。CANN 源码中：
```cpp
// fused_infer_attention_score_v3.cpp
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
// EMPTY - no implementation
#endif
```

### Phase 3: DiT 推理基准测试

#### 方法

- 绕过 Eagle backbone，直接用合成 tensor 构造 backbone 输出
- 补丁1/2/3 虽然挂载但由于 backbone 未运行，实际未生效
- 补丁4 降级为**手工 matmul+softmax 注意力**（因 `npu_fusion_attention` 不可用）
- 分别测量单步 DiT forward 和 4 步完整去噪过程

#### 合成输入规格

| 输入 | Shape | Dtype |
|------|-------|-------|
| backbone_features | [1, 512, 2048] | fp16 |
| backbone_attention_mask | [1, 512] | bool |
| image_mask | [1, 512] | bool |
| state | [1, 1, 128] | fp16 |
| embodiment_id | [1] | int64 |

#### 实测结果

| 组件 | 时延 (中位数) |
|------|-------------|
| 单步 DiT Forward | 79.88 ms |
| Action Head (4步) | 321.86 ms |
| 每步 Mean | 80.47 ms |
| 推理频率 | 3.11 Hz |

```
统计 (20 iterations):
  DiT单步: mean=80.06, min=79.26, max=82.10, p90=80.90, σ=0.65ms
  ActionHead: mean=323.14, min=315.98, max=336.42, p90=330.13, σ=5.28ms
  Per-step 开销 (AH/4 - DiT): 0.59ms (action_encoder + action_decoder)
```

#### 模型加载

| 指标 | 数值 |
|------|------|
| 加载时间 | ~20s |
| 总参数量 | 2,975,968,192 |
| 可训参数 | 1,619,968,000 |
| DiT 参数量 | 1,091,722,240 |
| 模型精度 | bfloat16 (backbone) / float16 (DiT) |
| NPU:0 显存占用 | ~7.0 GB |

## 补丁生效情况总结

| 补丁 | 是否应用 | 是否生效 | 原因 |
|------|:---:|:---:|------|
| RMSNorm → npu_rms_norm | ✓ | **✗** | Backbone 未运行 |
| RoPE → npu_rotary_mul | ✓ | **✗** | Backbone 未运行 |
| FlashAttn → npu_fusion_attention | ✓ | **✗** | Backbone 未运行 + 内核不可用 |
| AttnProcessor → fusion attention | ✓ (降级) | **部分** | 降级为手工 matmul，原版不可用 |

## 核心发现

1. **DrivingSDK 补丁对 Backbone 的优化未能在本次实验中验证** — 由于 Eagle processor 与 transformers 4.57.6 不兼容，无法加载完整推理管线，benchmark 只覆盖了 DiT 部分。

2. **310P3 缺少融合注意力内核是主要瓶颈** — `npu_fusion_attention` 全家桶在 310P3 上全部不可用。即使补丁逻辑正确，运行时也会因内核缺失而失败。Attention compute 占 DiT 单步约 16%，即使有融合内核，加速空间也有限（详见 PFA 实验记录）。

3. **手工 matmul 注意力是当前唯一可行的方案** — 在 310P3 上，DSL 中的 cube 单元已对 matmul 有良好的硬件加速（单次 0.15ms），融合内核替换的收益有限。

4. **与 ATC 编译方案互补** — 之前实验指出 DiT 的真正瓶颈是 kernel launch 开销（900+ 次小 kernel），ATC 编译可以融合这些算子。DrivingSDK 的补丁替换单个算子，两者互补：ATC 解决调度开销，补丁解决算子实现效率。

## 后续建议

| 优先级 | 方案 | 说明 |
|--------|------|------|
| P0 | 降级 transformers 或修复 Eagle processor 兼容性 | 让 backbone 跑起来，验证补丁 1/2 在 Qwen3 上的效果 |
| P1 | ATC 编译 DiT | 消除 kernel launch 开销，与手工 matmul 注意力并行推进 |
| P2 | 检查 CANN 升级是否修复 310P3 融合注意力 | 关注 CANN 新版本的 FlashAttentionScore 310P3 支持 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `/home/wangzhe/DrivingSDK/model_examples/GR00T-N1.6/patch.py` | DrivingSDK 原始补丁 |
| `/home/wangzhe/DrivingSDK/model_examples/GR00T-N1.6/README.md` | DrivingSDK GR00T-N1.6 使用说明 |
| `/home/wangzhe/Isaac-GR00T/scripts/deployment/benchmark_npu_inference.py` | 本次实验的 NPU 推理基准脚本 |
| `npu_prompt_flash_attention_实验记录.md` | 310P3 融合注意力算子独立实验 |
| `ATC_开发方案.md` | ATC 编译 DiT 方案 |
