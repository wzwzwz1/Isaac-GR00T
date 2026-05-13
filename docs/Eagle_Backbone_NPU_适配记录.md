# Eagle Backbone 310P3 NPU 适配记录

## 背景

GR00T-N1.6 的 Eagle Backbone（SigLIP2 视觉编码器 + Qwen3 LLM）最初被认为无法在 310P3 NPU 上运行，原因是 `torch.polar` 复数运算不支持。经过实验发现，通过将复数运算替换为等价的实值运算，Backbone 可以完整运行。

## 实验过程

### 阶段 1：初步结论（错误）

```
RuntimeError: Current settings do not support Complex dtype.
```

错误发生在 SigLIP2 的 RoPE (Rotary Position Embedding) 计算中：

```python
# modeling_siglip2.py:754
x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # 创建复数 tensor
y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
freqs_cis = torch.cat([x_cis, y_cis], dim=0)  # 复数 tensor

# 后续在 attention 中使用复数乘法
xq_ = torch.view_as_complex(xq.float().view(...))
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # 复数乘法
```

初步判断：310P3 NPU 不支持复数 dtype，Backbone 无法运行。这与昇腾官方文档中 `torch.polar` 支持 fp32 的描述一致——`torch.polar` **本身**可以 fallback 到 CPU 执行，但产生的**复数 tensor** 在后续 NPU 算子中无法使用。

### 阶段 2：关键洞察

复数形式的 RoPE 可以**完全用实值运算等价替代**。RoPE 的数学原理：

```
复数形式：
  x_cis = cos(θ) + i·sin(θ)    （cis 表示法）
  x_rope = x * x_cis             （复数乘法）

等价实值形式：
  (x_r + i·x_i) × (cos + i·sin)
  = (x_r·cos - x_i·sin) + i·(x_r·sin + x_i·cos)
```

因此复数操作链 `polar → complex multiply → view_as_real` 可以替换为纯实值操作 `cos/sin → real multiply → stack`，**完全不涉及复数 dtype**。

### 阶段 3：补丁实现

#### 补丁 1：RoPE 预计算 — 返回实值 cos/sin 而非复数

```python
# 原始（复数）
x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
freqs_cis = torch.cat([x_cis, y_cis], dim=0)  # [N+M, C/4] complex
return freqs_cis

# 补丁（实值）
x_cos, x_sin = torch.cos(x_freqs), torch.sin(x_freqs)
y_cos, y_sin = torch.cos(y_freqs), torch.sin(y_freqs)
freqs_cos = torch.cat([x_cos.unsqueeze(-1), y_cos.unsqueeze(-1)], dim=-1)
freqs_sin = torch.cat([x_sin.unsqueeze(-1), y_sin.unsqueeze(-1)], dim=-1)
freqs_cos = freqs_cos.reshape(self.max_height, self.max_width, -1)
freqs_sin = freqs_sin.reshape(self.max_height, self.max_width, -1)
return torch.stack([freqs_cos, freqs_sin], dim=-1)  # [H,W,C/4,2] float
```

#### 补丁 2：RoPE 应用 — 实值乘法替代复数乘法

```python
# 原始（复数乘法）
freqs_cis = freqs_cis.unsqueeze(-2)
xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)

# 补丁（实值乘法）
cos = freqs_cis[..., 0]  # [..., head_dim/4]
sin = freqs_cis[..., 1]
cos = cos.unsqueeze(-2)
sin = sin.unsqueeze(-2)
xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(-2)
xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(-2)
```

### 阶段 4：其他连锁补丁

RoPE 修复后，又遇到了一系列由 Flash Attention 不可用导致的错误：

| 问题 | 文件 | 修复 |
|------|------|------|
| `_flash_supports_window_size` 未定义 | `modeling_siglip2.py` | flash_attn 不可用时设为 False |
| `flash_attn_varlen_func` None 被调用 | `modeling_siglip2.py` | 添加 if/else 分支 |
| `F.scaled_dot_product_attention` 报 `can not cast format` | 全局 | monkey-patch 为手工 matmul+softmax+matmul |
| `initializer_range` 缺失 | `modeling_eagle3_vl.py` | `getattr(config, 'initializer_range', 0.02)` |
| `VideoInput` 不存在 | `processing_eagle3_vl.py` | 从 import 中移除 |
| PEP 604 语法 `str \| None` | 25+ .py 文件 | 加 `from __future__ import annotations` |

### 阶段 5：推理通过

```
✓ Backbone: torch.Size([1, 4, 2048]) in 69207ms
```

首次运行 69 秒（含大量 CPU fallback 和首次编译开销），但 Backbone 完整跑通。

## 最终性能

经过预热后的 Backbone 延迟约 10-20 秒（16 层 LLM 手工 SDPA）。对于 RoboCasa 评估场景：

| 组件 | 延迟 | 说明 |
|------|------|------|
| Backbone（首次） | ~70s | 预计算特征 |
| Backbone（预热后） | ~10-20s | CPU fallback 减少 |
| DiT 单步 (OM) | 28ms | 模型推理 |
| DiT 4步 (OM) | 112ms | 主要耗时 |
| Action Head | ~130ms | 前向推理 |
| **Action Head 总延迟** | **~130ms** | 不包括 Backbone |

## 结论

**310P3 NPU 可以运行 Eagle Backbone**，但需要 2 个关键补丁：

1. **RoPE 复数→实值**：数学等价替换，不依赖 NPU 对复数 dtype 的支持
2. **Flash Attention → 手工 SDPA**：flash_attn 在 ARM64 不可用，用 monkey-patch 降级

这不是 NPU 的硬件限制，而是 **软件层面的兼容性问题**。所有涉及复数 dtype 的运算都有实值等价形式，只是需要手动替换。

## 涉及的补丁文件

| 文件 | 补丁内容 |
|------|---------|
| `modeling_siglip2.py` (cached + local) | RoPE 复数→实值, flash_attn stub, SDPA fallback |
| `modeling_eagle3_vl.py` (cached + local) | flash_attn 断言移除, initializer_range 默认值 |
| `processing_eagle3_vl.py` (cached + local) | VideoInput 移除, unused_kwargs 容错 |
| `image_processing_eagle3_vl_fast.py` (cached + local) | BASE_IMAGE_PROCESSOR 导入移除 |
| `eagle_backbone.py` (local) | flash_attn/bf16 断言移除, sdpa fallback |
| `gr00t_n1d6.py` (local) | collator try/except |
| 25+ .py files | `from __future__ import annotations` |
