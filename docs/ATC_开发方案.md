# GR00T-N1.6 DiT ATC 编译方案

## 1. 目标与范围

将 GR00T-N1.6 推理管线中的 **AlternateVLDiT (32层扩散Transformer)** 通过昇腾 ATC 编译为 OM 离线模型，替代当前 PyTorch eager 模式的 DiT forward。

### 1.1 定量目标

| 指标 | 当前 (NPU eager) | ATC 后目标 |
|------|-----------------|-----------|
| DiT 单步耗时 | ~63 ms | ~44 ms |
| 4步去噪总耗时 | ~252 ms | ~176 ms |
| 端到端延迟 | ~304 ms | ~225 ms |
| 吞吐量 | 3.3 Hz | ~4.4 Hz |

### 1.2 范围边界

```
┌─────────────────────────────────────────────────────────┐
│                   GR00T 推理管线                         │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐  │
│  │ Eagle     │   │ State    │   │ AlternateVLDiT     │  │
│  │ Backbone  │   │ Encoder  │   │  (32层 × 4步)      │  │
│  │ (SigLIP2  │   │ (MLP)    │   │                    │  │
│  │  + LLM)   │   │          │   │  ← ATC 编译目标    │  │
│  │           │   │          │   │                    │  │
│  └──────────┘   └──────────┘   └────────────────────┘  │
│       ✗               ✗               ✓                │
│   不编译           不编译           编译                │
└─────────────────────────────────────────────────────────┘
```

- **编译范围**: `AlternateVLDiT.forward()` 及其 32 个 `BasicTransformerBlock`
- **不编译**: Eagle Backbone (结构复杂，含 HuggingFace 模型，不适合 ATC)、State Encoder / Action Decoder (计算量小，不值得)
- **宿主框架**: Python + PyTorch，通过 `torch_npu` 的 `npu` 后端加载 OM 模型

---

## 2. DiT 结构分析

### 2.1 模型规格

| 参数 | 值 | 来源 |
|------|-----|------|
| 层数 | 32 | `diffusion_model_cfg.num_layers` |
| 注意力头数 | 32 | `diffusion_model_cfg.num_attention_heads` |
| 头维度 | 48 | `diffusion_model_cfg.attention_head_dim` |
| 内部维度 | 1536 | 32×48 |
| Cross-attn 维度 | 2048 | `backbone_embedding_dim` |
| 输出维度 | 1024 | `diffusion_model_cfg.output_dim` |
| 层间模式 | interleaved | 偶数层 cross-attn，奇数层 self-attn |
| Norm 类型 | ada_norm | timestep-conditioned LayerNorm |

### 2.2 计算图结构

```
AlternateVLDiT.forward:
  input:
    hidden_states:      [B, 66, 1536]    # sa_embs (state + action features)
    encoder_hidden_states: [B, 256, 2048] # vl_embeds (backbone features)
    timestep:           [B]               # int64, 离散时间步
    image_mask:         [B, 256]          # bool, backbone 中 image token 的位置
    backbone_attention_mask: [B, 256]     # bool, backbone 有效 token

  flow:
    1. timestep → TimestepEncoder → temb [B, 1536]
    2. for each block in 32 layers:
       ├─ cross-attn block (idx even):
       │   ├─ AdaLayerNorm(hidden, temb) → normed_hidden
       │   ├─ QKV_project(normed_hidden) + KV_project(encoder)
       │   ├─ Attention: Q·K^T → softmax → ·V
       │   ├─ residual: hidden = hidden + attn_out
       │   ├─ LayerNorm(hidden) → normed
       │   └─ FeedForward: Linear(1536→6144) → GELU → Linear(6144→1536)
       │       residual: hidden = hidden + ff_out
       └─ self-attn block (idx odd):
           ├─ AdaLayerNorm(hidden, temb) → normed_hidden
           ├─ QKV_project(normed_hidden)
           ├─ Self-Attention: Q·K^T → softmax → ·V
           ├─ residual: hidden = hidden + attn_out
           ├─ LayerNorm(hidden) → normed
           └─ FeedForward: 同上
    3. output: proj_out_2(norm_out(hidden) * (1+scale) + shift)
       → [B, 66, 1024]
```

### 2.3 关键张量形状汇总

| 张量 | 形状 | dtype | 大小 (FP16) |
|------|------|-------|------------|
| hidden_states | [1, 66, 1536] | float16 | 198 KB |
| encoder_hidden_states | [1, 256, 2048] | float16 | 1 MB |
| temb | [1, 1536] | float16 | 3 KB |
| Q (per-block) | [1, 32, 66, 48] | float16 | 198 KB |
| K (cross-attn) | [1, 32, 256, 48] | float16 | 768 KB |
| V (cross-attn) | [1, 32, 256, 48] | float16 | 768 KB |
| Attention Scores | [1, 32, 66, 256] | float32 | **2.16 MB** |
| FFN 中间 | [1, 66, 6144] | float16 | 792 KB |
| image_mask | [1, 256] | bool | 256 B |
| backbone_attention_mask | [1, 256] | bool | 256 B |

### 2.4 mask 的使用方式

mask 决定 cross-attention 时 query 能 attend 到 encoder 的哪些 token：

```python
# dit.py:357-358
image_attention_mask = image_mask & backbone_attention_mask      # attend to image tokens
non_image_attention_mask = (~image_mask) & backbone_attention_mask  # attend to text tokens
```

两种 mask 在 32 层中交替使用（按 `attend_text_every_n_blocks=2` 配置）：
- 层 0, 4, 8, 12, 16, 20, 24, 28: attend to **non-image** (text) tokens
- 层 2, 6, 10, 14, 18, 22, 26, 30: attend to **image** tokens
- 层 1, 3, 5, ..., 31: **self-attention** (不使用 encoder)

mask 格式对应 Diffusers Attention 的 `attention_mask` 参数，以 `baddbmm` beta=1 模式传入（即 `attention_scores = mask + Q·K^T·scale`），其中有效位置填 0，无效位置填 `-inf` 或极小的负数。

---

## 3. ATC 编译方案

### 3.1 总体策略：分层导出

由于 AlternateVLDiT 每一层的结构相同但权重不同，且包含 32 次迭代循环，有两种导出策略：

#### 方案 A: 全模型导出 (推荐)

将整个 `AlternateVLDiT.forward()` 导出为单个 ONNX，内部包含 32 层循环展开。

```
优点: 单文件，ATC 可跨层做算子融合
缺点: ONNX 图巨大 (~3000 节点)，导出/编译时间长
```

#### 方案 B: 单层 Block 导出

只导出一个 `BasicTransformerBlock` (cross-attn 版本 + self-attn 版本)，在 Python 层循环调用 32 次。

```
优点: ONNX 图小，编译快，灵活
缺点: 32次 Python→ACL 调用开销，无法跨层融合
```

**推荐方案 A**，理由：310P3 的 kernel launch 开销已经是瓶颈之一（估算 10%），方案 B 会加剧此问题。全模型导出让 ATC 在更大范围内做算子融合和内存规划。

### 3.2 ONNX 导出

#### 3.2.1 导出准备：提取 DiT 子模型

```python
# scripts/atc/export_dit_onnx.py

import torch
import torch.nn as nn


class ExportableAlternateVLDiT(nn.Module):
    """
    包装 AlternateVLDiT 为可导出的独立模块.

    关键适配:
    1. 移除 BatchFeature 包装，所有输入转为纯 Tensor
    2. mask 预计算 (image_mask 和 backbone_attention_mask 在 Python 侧算好传入)
    3. 32 层循环在模块内部展开 (ONNX 导出时 unroll)
    """

    def __init__(self, dit_model):
        super().__init__()
        # 直接引用原模型的子模块
        self.timestep_encoder = dit_model.timestep_encoder
        self.transformer_blocks = dit_model.transformer_blocks  # ModuleList[32]
        self.norm_out = dit_model.norm_out
        self.proj_out_1 = dit_model.proj_out_1
        self.proj_out_2 = dit_model.proj_out_2
        self.config = dit_model.config
        self.attend_text_every_n_blocks = dit_model.attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, 66, 1536]
        encoder_hidden_states: torch.Tensor,   # [B, 256, 2048]
        timestep: torch.Tensor,                # [B]  int64
        image_attention_mask: torch.Tensor,    # [B, 256] bool
        non_image_attention_mask: torch.Tensor,# [B, 256] bool
    ) -> torch.Tensor:                         # [B, 66, 1024]
        # 1. timestep encoding
        temb = self.timestep_encoder(timestep)  # [B, 1536]

        # 2. 32 transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # self-attention block
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # cross-attention block
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_mask = non_image_attention_mask
                else:
                    curr_mask = image_attention_mask

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    temb=temb,
                )

        # 3. output projection
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)
```

#### 3.2.2 关键导出问题处理

##### 问题 1: Attention 内部算子

`BasicTransformerBlock.attn1` 是 `diffusers.models.attention.Attention`，其 `forward` 通过 `processor` 调用 `F.scaled_dot_product_attention`。在 NPU 上这被 monkey-patch 为 manual attention。

**ONNX 视角**: 手动 attention 会展开为 matmul + softmax + matmul —— 这正是 ATC 可以融合的部分。对 ONNX 导出本身没有问题，因为 ONNX 是 op-level 的。

但有潜在问题：`F.scaled_dot_product_attention` 如果被 monkey-patch 为一个 Python 函数（含多个 torch op），`torch.onnx.export` 以 `torch.jit.trace` 方式运行时能正确 trace 到内部的 matmul/softmax/matmul。

**验证方法**: 先用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 在 CPU 上测试 ONNX 导出，确认图结构正确。

##### 问题 2: bool mask 不兼容 ONNX

```python
image_attention_mask: [B, 256] bool → 转为 float [B, 256]
non_image_attention_mask: [B, 256] bool → 转为 float [B, 256]
```

处理方法：在 Python 包装层预先将 bool mask 转为 float，无效位置设为 `-10000.0`（等价于 `-inf` 在 softmax 中的效果），有效位置设为 `0.0`。

##### 问题 3: timestep int64 → int32

ATC 对 int64 类型支持有限，在导出前将 timestep 转为 int32。

##### 问题 4: diffusers Attention 的 baddbmm

Diffusers 的 `Attention.get_attention_scores` 使用 `torch.baddbmm` 做 Q·K^T + mask。这个算子在 ONNX 中会展开为 batch matrix multiply + add，是标准操作，ONNX 支持。

##### 问题 5: 32 层循环 → 图膨胀

32 层 × 每层 ~100 节点 ≈ 3200 节点。ATC 支持的 ONNX 模型通常可达数万节点，3200 节点在可接受范围内。

但 `BasicTransformerBlock` 内部有 `if/else`（ada_norm vs layer_norm 分支）。由于在 trace 时分支固定（推理时 `norm_type` 不变），trace 只会记录实际执行路径，不会产生动态控制流。

#### 3.2.3 动态轴设置

```python
dynamic_axes = {
    "hidden_states":         {0: "batch", 1: "seq_len"},
    "encoder_hidden_states": {0: "batch", 1: "ctx_len"},
    "timestep":              {0: "batch"},
    "image_attention_mask":  {0: "batch", 1: "ctx_len"},
    "non_image_attention_mask": {0: "batch", 1: "ctx_len"},
    "output":                {0: "batch", 1: "seq_len"},
}
```

关于 `seq_len` 是否需要动态：
- 当前 `seq_len = 1(state) + action_horizon(16)` = 17（加上可能的 padding 到 66）
- 如果 action_horizon 固定为 16，seq_len 可以固定为 66
- 建议初期固定尺寸以获取最佳 ATC 优化，后续再支持动态

**推荐**: 第一阶段固定 B=1, seq_len=66, ctx_len=256，编译为静态形状 OM，获得最优性能。后期按需编译多档形状。

#### 3.2.4 完整导出脚本

```python
# scripts/atc/export_dit_onnx.py

import torch
import torch.nn.functional as F
from pathlib import Path


def export_dit_to_onnx(model, output_path: str, opset_version: int = 14):
    """
    将 GR00T AlternateVLDiT 导出为 ONNX.

    Args:
        model: 完整的 Gr00tN1d6 模型 (已加载权重)
        output_path: ONNX 输出路径
        opset_version: ONNX opset 版本 (14 对 NPU 兼容性最好)

    注意:
        - 需要在 NPU 机器上运行 (或使用 TORCH_DEVICE_BACKEND_AUTOLOAD=0 在 CPU 上)
        - 导出的 ONNX 包含 32 层完整 DiT 计算图
    """
    dit = model.action_head.model  # AlternateVLDiT

    class ExportWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.timestep_encoder = dit.timestep_encoder
            self.transformer_blocks = dit.transformer_blocks
            self.norm_out = dit.norm_out
            self.proj_out_1 = dit.proj_out_1
            self.proj_out_2 = dit.proj_out_2
            self.attend_text_every_n_blocks = dit.attend_text_every_n_blocks

        def forward(self, hidden_states, encoder_hidden_states, timestep,
                    image_mask_float, non_image_mask_float):
            temb = self.timestep_encoder(timestep)

            for idx, block in enumerate(self.transformer_blocks):
                if idx % 2 == 1:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        temb=temb,
                    )
                else:
                    curr_mask = (non_image_mask_float if idx % (2 * self.attend_text_every_n_blocks) == 0
                                 else image_mask_float)
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=curr_mask,
                        temb=temb,
                    )

            conditioning = temb
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            return self.proj_out_2(hidden_states)

    wrapper = ExportWrapper()
    wrapper.eval()

    # 构造假输入 (固定形状)
    B, SEQ, CTX = 1, 66, 256
    dummy_inputs = (
        torch.randn(B, SEQ, 1536, dtype=torch.float16),       # hidden_states
        torch.randn(B, CTX, 2048, dtype=torch.float16),       # encoder_hidden_states
        torch.zeros(B, dtype=torch.int32),                     # timestep
        torch.zeros(B, CTX, dtype=torch.float16),              # image_mask as float
        torch.ones(B, CTX, dtype=torch.float16) * -10000.0,   # non_image_mask as float
    )

    # CPU 导出 (避免 NPU 算子干扰 ONNX trace)
    wrapper = wrapper.cpu().float()  # ONNX 导出建议用 fp32 权重
    dummy_inputs = tuple(t.cpu().float() for t in dummy_inputs)

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        opset_version=opset_version,
        input_names=[
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "image_attention_mask",
            "non_image_attention_mask",
        ],
        output_names=["output"],
        dynamic_axes={
            "hidden_states": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "timestep": {0: "batch"},
            "image_attention_mask": {0: "batch"},
            "non_image_attention_mask": {0: "batch"},
            "output": {0: "batch"},
        },
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,
        do_constant_folding=True,
    )

    print(f"ONNX exported to {output_path}")
```

##### 验证 ONNX

```bash
# 检查算子和图结构
python3 -c "
import onnx
model = onnx.load('dit_model.onnx')
onnx.checker.check_model(model)
print(f'图节点数: {len(model.graph.node)}')
print(f'输入: {[i.name for i in model.graph.input]}')
print(f'输出: {[o.name for o in model.graph.output]}')

# 检查是否有不支持的算子
unsupported = set()
for node in model.graph.node:
    if node.op_type in ['If', 'Loop', 'Scan']:
        unsupported.add(node.op_type)
print(f'不支持的算子: {unsupported or \"无\"}')"
```

### 3.3 ONNX 算子清洗

导出的 ONNX 可能包含 ATC 不直接支持的算子，需要清洗：

#### 3.3.1 Diffusers 的 baddbmm 路径

Diffusers 的 `Attention.get_attention_scores` 使用 `torch.baddbmm`：

```python
# attention.py:432-438
attention_scores = torch.baddbmm(
    baddbmm_input,       # attention_mask or empty
    query,               # [B*H, 66, 48]
    key.transpose(-1,-2),# [B*H, 48, 256]
    beta=beta,
    alpha=self.scale,
)
```

ONNX trace 后 `baddbmm` 会展开为 `MatMul → Mul(scale) → Add(mask)`，这些是标准 ONNX 算子，ATC 原生支持。

#### 3.3.2 可能的问题算子

| ONNX 算子 | ATC 支持 | 备注 |
|-----------|---------|------|
| MatMul, Add, Mul, Div | ✓ | 原生支持 |
| Softmax | ✓ | 支持 |
| LayerNormalization | ✓ | 需 ONNX opset ≥ 17 |
| Reshape, Transpose, Squeeze, Unsqueeze | ✓ | 原生支持 |
| SiLU (SigmoidLinearUnit) | ✓ | opset ≥ 14 (即 `x·sigmoid(x)`) |
| Where (替代 bool mask) | ✓ | 支持 |
| `torch.baddbmm` → MatMul+Mul+Add | ✓ | 展开为基本算子 |
| GELU (GaussianErrorLinearUnit) | can 支持 | FFN 激活函数 |
| `F.silu` → SiLU | ✓ | 原生支持 |

**结论**: 当前 DiT 使用的全部算子都在 ATC 支持范围内，预计不需要额外清洗。

#### 3.3.3 如果遇到不支持的算子

```bash
# 使用 onnx-simplifier 简化图
pip install onnx-simplifier
python3 -m onnxsim dit_model.onnx dit_model_sim.onnx

# 自定义清洗脚本
python3 scripts/atc/clean_onnx.py --input dit_model.onnx --output dit_model_clean.onnx
```

### 3.4 ATC 编译

#### 3.4.1 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 确认 ATC 可用
atc --version

# 确认 SOC 版本
npu-smi info -t board -i 0 | grep "Chip"
# 预期输出: Ascend 310P3
```

#### 3.4.2 编译命令

```bash
atc \
    --model=dit_model.onnx \
    --framework=5 \
    --output=dit_310p3_fp16 \
    --soc_version=Ascend310P3 \
    --input_shape="hidden_states:1,66,1536;encoder_hidden_states:1,256,2048;timestep:1;image_attention_mask:1,256;non_image_attention_mask:1,256" \
    --input_format=ND \
    --input_fp16_nodes="hidden_states;encoder_hidden_states;image_attention_mask;non_image_attention_mask" \
    --output_type=FP16 \
    --op_precision_mode=allow_fp32_to_fp16 \
    --precision_mode=allow_fp32_to_fp16 \
    --log=info \
    --out_nodes="output" \
    --enable_small_channel=1 \
    --insert_op_conf=aipp_dit.cfg \
    --singleop=compile \
    2>&1 | tee atc_compile.log
```

参数说明：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--framework=5` | ONNX | ATC 框架编号 |
| `--soc_version` | Ascend310P3 | 目标芯片 |
| `--input_shape` | 固定形状 | 初期用固定 shape 获得最优性能 |
| `--input_fp16_nodes` | 所有浮点输入 | 输入保持 FP16，避免额外转换 |
| `--output_type` | FP16 | 输出精度 |
| `--precision_mode` | allow_fp32_to_fp16 | 允许 FP32→FP16 转换 |
| `--enable_small_channel=1` | ON | 对小通道 (48) 的 attention head 优化 |
| `--op_precision_mode` | allow_fp32_to_fp16 | 与 precision_mode 协同 |

#### 3.4.3 常见编译错误处理

| 错误 | 原因 | 解决 |
|------|------|------|
| `E10001: Check graph fail` | ONNX 图不符合 ATC 要求 | 用 onnx-simplifier 简化；检查是否有动态控制流 |
| `E10002: Unsupported op type X` | 算子不支持 | 用 ONNX opset 13 或 14 重新导出；或用等价算子替换 |
| `E19010: No parser is registered for This op` | 某算子无 NPU 实现 | 将该算子保留在 CPU 执行 (见 3.6) |
| `E10008: Shape inference fail` | 形状推导失败 | 固定所有维度，避免动态 shape |
| `E14999: Compile timeout` | 图过大 | 尝试方案 B (单层导出) |

### 3.5 推理集成

#### 3.5.1 OM 模型加载和推理

```python
# scripts/atc/dit_om_runner.py

import torch
import torch_npu
import numpy as np

class DiTOMRunner:
    """在 PyTorch 推理管线中加载和运行 OM 格式的 DiT 模型."""

    def __init__(self, om_path: str, device_id: int = 0):
        import acl
        self.acl = acl
        self.device_id = device_id

        # 初始化 ACL
        ret = acl.init()
        assert ret == 0, f"ACL init failed: {ret}"
        ret = acl.rt.set_device(device_id)
        assert ret == 0, f"ACL set_device failed: {ret}"

        # 加载模型
        self.model_id, _ = acl.mdl.load_from_file(om_path)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        assert ret == 0, f"ACL get_desc failed: {ret}"

        # 创建输入输出 dataset
        self.input_dataset, self.output_dataset = acl.mdl.create_dataset()

    def run(self, hidden_states, encoder_hidden_states, timestep,
            image_mask, non_image_mask):
        """
        运行 DiT OM 模型推理.

        Args:
            hidden_states: torch.Tensor [1, 66, 1536] float16 on NPU
            encoder_hidden_states: torch.Tensor [1, 256, 2048] float16 on NPU
            timestep: int
            image_mask: torch.Tensor [1, 256] float16 on NPU
            non_image_mask: torch.Tensor [1, 256] float16 on NPU

        Returns:
            torch.Tensor [1, 66, 1024] float16
        """
        # 将 NPU tensor 转为 numpy (零拷贝)
        inputs = [
            hidden_states.cpu().numpy(),
            encoder_hidden_states.cpu().numpy(),
            np.array([timestep], dtype=np.int32),
            image_mask.cpu().numpy(),
            non_image_mask.cpu().numpy(),
        ]

        # 创建 ACL buffer
        for i, inp in enumerate(inputs):
            buffer_size = inp.nbytes
            buffer, ret = acl.rt.malloc(buffer_size, acl.const.ACL_MEMCPY_HOST_TO_DEVICE)
            ret = acl.rt.memcpy(buffer, buffer_size, inp, buffer_size,
                                acl.const.ACL_MEMCPY_HOST_TO_DEVICE)

            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)

        # 分配输出 buffer
        output_size = 1 * 66 * 1024 * 2  # FP16
        out_buffer, _ = acl.rt.malloc(output_size, acl.const.ACL_MEMCPY_HOST_TO_DEVICE)
        out_data_buffer = acl.create_data_buffer(out_buffer, output_size)
        acl.mdl.add_dataset_buffer(self.output_dataset, out_data_buffer)

        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)

        # 取回输出
        out_np = np.empty((1, 66, 1024), dtype=np.float16)
        acl.rt.memcpy(out_np.ctypes.data, output_size, out_buffer, output_size,
                      acl.const.ACL_MEMCPY_DEVICE_TO_HOST)

        return torch.from_numpy(out_np).to(device="npu:0")

    def __del__(self):
        self.acl.mdl.unload(self.model_id)
        self.acl.mdl.destroy_desc(self.model_desc)
        self.acl.finalize()
```

#### 3.5.2 集成到现有推理管线

```python
# 修改 gr00t/model/gr00t_n1d6/gr00t_n1d6.py 中 get_action_with_features

class Gr00tN1d6ActionHead(nn.Module):
    def __init__(self, config, use_om_dit: bool = False, om_path: str = None):
        # ... 原有初始化 ...
        if use_om_dit and om_path:
            self._om_runner = DiTOMRunner(om_path)
            self._use_om = True
        else:
            self._om_runner = None
            self._use_om = False

    def _run_dit(self, hidden_states, encoder_hidden_states, timestep,
                 image_mask, backbone_attention_mask):
        if self._use_om:
            # 预计算 float mask (替代 bool mask)
            image_mask_float = torch.where(
                image_mask & backbone_attention_mask,
                torch.zeros_like(image_mask, dtype=torch.float16),
                torch.full_like(image_mask, -10000.0, dtype=torch.float16),
            )
            non_image_mask_float = torch.where(
                (~image_mask) & backbone_attention_mask,
                torch.zeros_like(image_mask, dtype=torch.float16),
                torch.full_like(image_mask, -10000.0, dtype=torch.float16),
            )
            return self._om_runner.run(
                hidden_states, encoder_hidden_states,
                timestep.item(), image_mask_float, non_image_mask_float,
            )
        else:
            # 原有 PyTorch eager 路径
            return self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
```

### 3.6 备选：单层 Block 导出方案

如果全模型导出遇到问题（编译超时、内存不足），降级为单层 block 导出：

```bash
# 分别导出 cross-attn block 和 self-attn block
atc --model=cross_attn_block.onnx --framework=5 \
    --output=cross_attn_block --soc_version=Ascend310P3 \
    --input_shape="hidden_states:1,66,1536;encoder_hidden_states:1,256,2048;temb:1,1536;attn_mask:1,256" \
    --input_fp16_nodes="hidden_states;encoder_hidden_states;temb;attn_mask" \
    --output_type=FP16

atc --model=self_attn_block.onnx --framework=5 \
    --output=self_attn_block --soc_version=Ascend310P3 \
    --input_shape="hidden_states:1,66,1536;temb:1,1536" \
    --input_fp16_nodes="hidden_states;temb" \
    --output_type=FP16
```

在 Python 层循环 32 次调用 OM 模型。额外开销估算：32×2×30μs ≈ 2ms 的 Python→ACL dispatch 开销。

### 3.7 备选：torch_npu 融合优化（轻量级替代）

如果 ATC 编译链条过长，可以先尝试直接用 `torch_npu` 融合大算子：

```python
# 替换 DiT 中的 BasicTransformerBlock 手动 attention
import torch_npu

# 对 cross-attention block
# 将 AdaLayerNorm + Linear + reshape 融合
hidden_states, encoder_hidden_states, output = torch_npu.npu_fused_attention(
    query=normed_q, key=normed_k, value=normed_v,
    head_num=32,  # attention heads
)
```

这个方案实现成本低（1-2天），但效果不如完整 ATC 编译。

---

## 4. 开发步骤与时间线

### Phase 1: ONNX 导出验证 (1-2天)

```
□ 编写 ExportableAlternateVLDiT 包装模块
□ 在 CPU 上 (TORCH_DEVICE_BACKEND_AUTOLOAD=0) 导出 ONNX
□ onnx.checker 验证图结构
□ 对比 ONNX 推理输出与 PyTorch 推理输出 (数值精度 ±1e-3)
```

### Phase 2: ATC 编译 (1天)

```
□ 编写 ATC 编译脚本
□ 尝试全模型编译 (可能多次迭代修复不支持的算子)
□ 若失败则降级为单层 Block 方案
□ 生成 OM 文件
```

### Phase 3: 推理集成 (2-3天)

```
□ 编写 DiTOMRunner (ACL Python 封装)
□ 修改 get_action_with_features 集成 OM
□ 端到端精度验证 (OM vs PyTorch 输出 actions 一致)
□ Benchmark: 对比 OM vs eager 延迟
```

### Phase 4: 优化迭代 (2-3天)

```
□ 性能未达标: 分析 ATC 日志，调整编译选项
□ 尝试 FP16/INT8 混合精度编译
□ 调整算子系统配置 (L1/L2 buffer 分配)
□ 最终 benchmark 报告
```

---

## 5. 风险与限制

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| ONNX 图过大编译失败 | 中 | 必须降级方案B | 准备单层导出脚本 |
| diffusers Attention 算子不兼容 | 低 | 需定制 ONNX export | 替换为等价 matmul+softmax 实现 |
| OM 输出精度下降 | 中 | 动作质量下降 | FP32 部分保留 + 精度对比测试 |
| bool mask 转换引入额外开销 | 低 | 1-2ms 增加 | Python 侧预计算 float mask |
| 手工 SDPA 仍是性能瓶颈 | 高 | ATC 收益低于预期 | 后续必须上 Ascend C |

**最大的限制**: ATC 不能消除手工 SDPA attention 的核心瓶颈——FP32 attention scores 矩阵（2.16MB）的 HBM 读写。即使 ATC 融合了相邻算子，这个中间矩阵依然要落 HBM。只有 Ascend C 自定义 kernel （online softmax in L1 buffer）才能真正解决这个问题。

---

## 6. 验收标准

```
□ ONNX 模型通过 onnx.checker 验证
□ ONNX vs PyTorch 输出差异 < 1e-3 (FP16)
□ OM 模型编译成功，atc 无错误
□ OM vs PyTorch 输出差异 < 1e-2 (FP16)
□ DiT 单步延迟 < 50ms (当前 ~63ms)
□ 端到端延迟 < 230ms (当前 ~304ms)
□ 动作精度无损 (输出 action 向量差异 < 1%)
```

---

## 附录 A: 关键文件路径

| 文件 | 用途 |
|------|------|
| `gr00t/model/modules/dit.py` | DiT / AlternateVLDiT / BasicTransformerBlock 定义 |
| `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | Action head + 去噪循环 |
| `gr00t/configs/model/gr00t_n1d6.py` | 模型配置 (维度、层数等) |
| `gr00t/model/modules/embodiment_conditioned_mlp.py` | State Encoder / Action Encoder / Action Decoder |
| `gr00t/policy/gr00t_policy.py` | 推理入口 (若使用完整 pipeline) |

## 附录 B: 参考命令速查

```bash
# ONNX 导出 (在 NPU 机器上)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/atc/export_dit_onnx.py

# ONNX 验证
python3 -c "
import onnx
m = onnx.load('dit_model.onnx')
onnx.checker.check_model(m)
print(f'Nodes: {len(m.graph.node)}, Opset: {m.opset_import[0].version}')
"

# ATC 编译
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/atc/compile_dit.sh

# 精度验证
python3 scripts/atc/verify_precision.py \
    --pytorch_model /root/models/GR00T-N1.6-3B-FP16 \
    --om_model output/dit_310p3_fp16.om

# Benchmark
python3 scripts/profiling/latency_profiler.py \
    --mode steps --num_warmup 5 --num_runs 50
```

## 附录 C: 算子映射表

| PyTorch 算子 | ONNX 算子 (opset 14) | ATC 支持 | 备注 |
|-------------|---------------------|---------|------|
| `F.linear` | `MatMul + Add` | ✓ | |
| `torch.baddbmm` | `MatMul + Mul + Add` | ✓ | 展开为 3 个基本算子 |
| `F.softmax` | `Softmax` | ✓ | |
| `F.layer_norm` | `LayerNormalization` | ✓ | opset≥17 |
| `F.silu(x)` | `Mul(x, Sigmoid(x))` | ✓ | opset≥14 |
| `F.gelu` | `Gelu` | ✓ | 或展开为 erf |
| `torch.chunk` | `Slice × 2` | ✓ | |
| `torch.cat` | `Concat` | ✓ | |
| `torch.where` | `Where` | ✓ | |
| `nn.Embedding` | `Gather` | ✓ | Position embedding |
| `+` / `*` (tensor op) | `Add` / `Mul` | ✓ | |
| `tensor[:, None]` | `Unsqueeze` | ✓ | |
