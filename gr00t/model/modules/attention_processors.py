"""
Ascend NPU 自定义 Attention Processor.

在 310P3 上使用 npu_prompt_flash_attention 替换手工 SDPA,
在 910B 上使用 npu_fusion_attention (通过 diffusers AttnProcessorNPU),
在其他设备上保持默认 AttnProcessor2_0.

用法:
    from gr00t.model.modules.attention_processors import set_npu_attention_processors
    set_npu_attention_processors(dit_model)
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# NPU 检测
# ═══════════════════════════════════════════════════════════════════════

def _get_npu_device_type() -> Optional[str]:
    """检测当前 NPU 型号. 返回 'ascend310p3', 'ascend910b', 或 None."""
    override = os.environ.get("GR00T_FORCE_ATTN_PROCESSOR")
    if override:
        return override

    try:
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            return None
        name = torch.npu.get_device_name(0).lower()
        if "310p3" in name or "310p" in name:
            return "ascend310p3"
        if "910" in name:
            return "ascend910b"
        return None
    except Exception:
        return None


def _is_310p3() -> bool:
    return _get_npu_device_type() == "ascend310p3"


def _is_910b() -> bool:
    return _get_npu_device_type() == "ascend910b"


# ═══════════════════════════════════════════════════════════════════════
# 310P3 Attention Processor
# ═══════════════════════════════════════════════════════════════════════

class AttnProcessorNPU310P3:
    """
    使用 npu_prompt_flash_attention 的 Attention Processor.

    模仿 diffusers AttnProcessor2_0 的接口, 但将 F.scaled_dot_product_attention
    替换为 torch_npu.npu_prompt_flash_attention (310P3 唯一可用的融合 attention op).

    输入布局: BNSD (Batch, NumHeads, SeqLen, HeadDim)
    支持: 交叉注意力 (Q seq != KV seq) 和自注意力
    """

    def __init__(self):
        self._disabled = os.environ.get("GR00T_DISABLE_NPU_PFA", "0") == "1"
        if self._disabled:
            print("[AttnProcessorNPU310P3] Disabled by GR00T_DISABLE_NPU_PFA=1")

    def __call__(
        self,
        attn,  # diffusers.models.attention_processor.Attention
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """与 AttnProcessor2_0.__call__ 签名兼容."""

        # ── 前处理 (同 AttnProcessor2_0) ──
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # ── QKV 投影 ──
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # ── Reshape to BNSD ──
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ── 调用融合 attention (预切片策略绕过 310P3 mask 限制) ──
        hidden_states = self._fused_attention_with_slice(
            query, key, value, attn, attention_mask
        )

        # ── 输出投影 (同 AttnProcessor2_0) ──
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _can_slice_mask(
        self, attention_mask: Optional[torch.Tensor], kv_seq_len: int
    ) -> str:
        """
        判断 mask 是否可以通过预切片 K/V 来替代.

        返回:
            "none"         — mask 为空, 不需要切片
            "slice"        — 可以切片, indices 存储为 self._slice_indices
            "fallback"     — 复杂 mask, 无法切片
        """
        if attention_mask is None:
            return "none"

        # 情况: 2D bool mask [B, S_kv] — 可以尝试切片
        if attention_mask.dim() == 2 and attention_mask.dtype == torch.bool:
            row = attention_mask[0]
            # 找 True 的连续区间
            first_true = None
            last_true = None
            for i, val in enumerate(row.cpu().tolist()):
                if val:
                    if first_true is None:
                        first_true = i
                    last_true = i

            if first_true is not None:
                # 检查连续性 (中间不能有 False)
                contiguous = all(
                    row[j] for j in range(first_true, last_true + 1)
                )
                if contiguous:
                    self._slice_start = first_true
                    self._slice_end = last_true + 1
                    return "slice"

        # 复杂 mask 或无 True 元素
        return "fallback"

    def _fused_attention_with_slice(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        使用预切片策略调用 npu_prompt_flash_attention.

        310P3 限制:
        - PFA 在 Q seq != KV seq 时不支持 attention mask
        - PFA 在小 seq (如 17) 时也不支持 mask (对齐问题)

        策略:
        1. 无 mask → 直接 PFA
        2. 简单 bool mask (连续 True 区间) → 预切片 K/V → PFA 无 mask
        3. 复杂 mask → fallback 到 manual SDPA
        """
        if self._disabled or query.dtype not in (torch.float16,):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False,
            )

        kv_seq_len = key.shape[2]
        action = self._can_slice_mask(attention_mask, kv_seq_len)

        # 情况 3: 复杂 mask, fallback
        if action == "fallback":
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False,
            )

        # 情况 2: 预切片 K/V
        if action == "slice":
            key = key[:, :, self._slice_start:self._slice_end, :]
            value = value[:, :, self._slice_start:self._slice_end, :]

        # 情况 1 & 2: 调用 PFA (无 mask)
        try:
            import torch_npu
            scale = float(1.0 / math.sqrt(query.shape[-1]))
            return torch_npu.npu_prompt_flash_attention(
                query, key, value,
                num_heads=attn.heads,
                scale_value=scale,
                input_layout="BNSD",
                pre_tokens=2147483647,
                next_tokens=2147483647,
                sparse_mode=0,
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            if not hasattr(self, "_fallback_warned"):
                print(f"[AttnProcessorNPU310P3] PFA failed: {e}, falling back to manual SDPA")
                self._fallback_warned = True
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False,
            )


# ═══════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════

def set_npu_attention_processors(model) -> None:
    """
    为 DiT / ActionHead 模型中的所有 Attention 层设置 NPU 优化处理器.

    参数:
        model: DiT, AlternateVLDiT, 或 Gr00tN1d6ActionHead 实例

    行为:
        - Ascend 310P3: 使用 AttnProcessorNPU310P3 (npu_prompt_flash_attention)
        - Ascend 910B:  使用 diffusers AttnProcessorNPU (npu_fusion_attention)
        - 其他设备:      保持默认处理器不变
    """
    from diffusers.models.attention_processor import Attention

    device_type = _get_npu_device_type()

    if device_type == "ascend310p3":
        processor = AttnProcessorNPU310P3()
        label = "AttnProcessorNPU310P3 (npu_prompt_flash_attention)"
    elif device_type == "ascend910b":
        # 使用 diffusers 内置的 910B 处理器
        from diffusers.models.attention_processor import AttnProcessorNPU
        processor = AttnProcessorNPU()
        label = "AttnProcessorNPU (npu_fusion_attention)"
    else:
        print("[Attention] Not on Ascend NPU, keeping default processor")
        return

    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, Attention):
            module.set_processor(processor)
            count += 1

    print(f"[Attention] Applied {label} to {count} Attention layers")


def get_attention_processor_info() -> dict:
    """返回当前 attention 处理器配置信息."""
    return {
        "device_type": _get_npu_device_type(),
        "processor": ("AttnProcessorNPU310P3" if _is_310p3()
                      else "AttnProcessorNPU" if _is_910b()
                      else "Default (AttnProcessor2_0)"),
        "disabled": os.environ.get("GR00T_DISABLE_NPU_PFA", "0") == "1",
    }
