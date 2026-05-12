#!/usr/bin/env python3
"""
Phase 1: 将 GR00T AlternateVLDiT 导出为 ONNX.

关键适配:
  - mask 预转为 float (bool 不兼容 ONNX)
  - timestep int64 → int32
  - 在 CPU 上导出 (避免 NPU 算子干扰 trace)
  - 固定形状: B=1, Seq=51 (1 state + 50 actions), Ctx=256
  - SDPA monkey-patch 确保 trace 看到 matmul+softmax+matmul

用法:
    python3 scripts/atc/export_dit_onnx.py \
        --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
        --output atc_output/dit_model.onnx
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# SDPA Monkey-Patch (导出时确保 trace 到 matmul+softmax+matmul)
# ═══════════════════════════════════════════════════════════════════════

def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
    d_k = query.shape[-1]
    s = scale if scale is not None else 1.0 / (d_k ** 0.5)
    scores = torch.matmul(query, key.transpose(-2, -1)) * s
    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        elif attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        scores = scores + attn_mask
    attn_w = F.softmax(scores, dim=-1)
    return torch.matmul(attn_w, value)


# ═══════════════════════════════════════════════════════════════════════
# Exportable 包装器
# ═══════════════════════════════════════════════════════════════════════

class ExportableAlternateVLDiT(torch.nn.Module):
    """
    将 AlternateVLDiT 包装为纯 Tensor 输入/输出的可导出模块.

    mask 预处理为 float:
      - image_attention_mask: [B, ctx] float, 0=attend, -10000=mask
      - non_image_attention_mask: [B, ctx] float
    这两个 mask 在 Python 侧预计算后传入，避免 ONNX 中的 bool 类型问题.
    """

    def __init__(self, dit_model):
        super().__init__()
        self.timestep_encoder = dit_model.timestep_encoder
        self.transformer_blocks = dit_model.transformer_blocks
        self.norm_out = dit_model.norm_out
        self.proj_out_1 = dit_model.proj_out_1
        self.proj_out_2 = dit_model.proj_out_2
        self.config = dit_model.config
        self.attend_text_every_n_blocks = dit_model.attend_text_every_n_blocks

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                image_attn_mask_float, non_image_attn_mask_float):
        """
        Args:
            hidden_states:              [B, Seq, 1536]  float16
            encoder_hidden_states:      [B, Ctx, 2048]   float16
            timestep:                   [B]              int32
            image_attn_mask_float:      [B, Ctx]         float16 (0=attend, -10000=mask)
            non_image_attn_mask_float:  [B, Ctx]         float16

        Returns:
            [B, Seq, 1024] float16
        """
        temb = self.timestep_encoder(timestep)

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # self-attention
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # cross-attention — 使用预计算的 float mask
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_mask = non_image_attn_mask_float
                else:
                    curr_mask = image_attn_mask_float

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    temb=temb,
                )

        # Output head
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


# ═══════════════════════════════════════════════════════════════════════
# 导出逻辑
# ═══════════════════════════════════════════════════════════════════════

def export_dit_to_onnx(model, output_path: str, opset: int = 14,
                       batch: int = 1, seq_len: int = 51, ctx_len: int = 256):
    """
    将 GR00T AlternateVLDiT 导出为 ONNX.

    Args:
        model:      Gr00tN1d6 模型 (已加载权重, 在 NPU 或 CPU 上)
        output_path: ONNX 输出路径
        opset:       ONNX opset version
        batch, seq_len, ctx_len: 固定形状参数
    """
    dit = model.action_head.model  # AlternateVLDiT

    wrapper = ExportableAlternateVLDiT(dit)
    wrapper.eval()

    # ── 构造假输入 ──
    inner_dim = dit.config.num_attention_heads * dit.config.attention_head_dim  # 1536
    cross_dim = dit.config.cross_attention_dim  # 2048

    dummy = (
        torch.randn(batch, seq_len, inner_dim, dtype=torch.float16),     # hidden_states
        torch.randn(batch, ctx_len, cross_dim, dtype=torch.float16),    # encoder_hidden_states
        torch.zeros(batch, dtype=torch.int32),                            # timestep
        torch.zeros(batch, ctx_len, dtype=torch.float16),                # image_mask (0=attend)
        torch.full((batch, ctx_len), -10000.0, dtype=torch.float16),    # non_image_mask (-10000=mask)
    )

    # ── 移到 CPU (避免 NPU 算子干扰 ONNX trace) ──
    wrapper = wrapper.cpu().float()
    dummy = tuple(t.cpu().float() if t.dtype != torch.int32 else t.cpu() for t in dummy)
    # 注意: timestep 保持 int32 (index 类型), 不转 float

    print(f"[ONNX Export] opset={opset}, B={batch}, Seq={seq_len}, Ctx={ctx_len}")
    print(f"[ONNX Export] wrapper params: {sum(p.numel() for p in wrapper.parameters()):,}")

    # ── Monkey-patch SDPA 确保 trace 展开 ──
    _orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _manual_sdpa

    try:
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            opset_version=opset,
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
        print(f"[ONNX Export] ✓ Saved to {output_path}")
    finally:
        F.scaled_dot_product_attention = _orig_sdpa

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════════════════

def verify_onnx(onnx_path: str):
    """加载并验证 ONNX 模型结构."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        ops = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1

        print(f"\n[ONNX Verify] ✓ Model valid")
        print(f"  Total nodes: {len(model.graph.node):,}")
        print(f"  Inputs:      {[i.name for i in model.graph.input]}")
        print(f"  Outputs:     {[o.name for o in model.graph.output]}")
        print(f"  Opset:       {model.opset_import[0].version}")
        print(f"  Top ops:     {sorted(ops.items(), key=lambda x: x[1], reverse=True)[:15]}")

        # 检查不支持的控制流算子
        unsupported = [n for n in model.graph.node
                       if n.op_type in ('If', 'Loop', 'Scan', 'DynamicQuantizeLinear')]
        if unsupported:
            print(f"  ⚠ Unsupported ops: {[n.op_type for n in unsupported]}")
        else:
            print(f"  ✓ No unsupported control flow ops found")

        return True
    except ImportError:
        print("[ONNX Verify] onnx package not installed, skipping verification")
        return True
    except Exception as e:
        print(f"[ONNX Verify] ✗ Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Export GR00T DiT to ONNX")
    parser.add_argument("--model_path", type=str,
                        default="/home/wangzhe/models/GR00T-N1.6-3B-FP16")
    parser.add_argument("--output", type=str,
                        default="atc_output/dit_model.onnx")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=51,
                        help="Sequence length (1 state + action_horizon)")
    parser.add_argument("--ctx_len", type=int, default=256)
    parser.add_argument("--verify_only", type=str, default=None,
                        help="Only verify an existing ONNX file")
    args = parser.parse_args()

    if args.verify_only:
        verify_onnx(args.verify_only)
        return

    # ── 加载模型 ──
    sys.path.insert(0, '.')
    import gr00t.model  # noqa
    from transformers import AutoModel

    print(f"[Setup] Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()

    c = model.config
    print(f"[Setup] action_horizon={c.action_horizon}, "
          f"action_dim={c.max_action_dim}, "
          f"inner_dim={c.diffusion_model_cfg['num_attention_heads'] * c.diffusion_model_cfg['attention_head_dim']}")

    # ── 导出 ──
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    export_dit_to_onnx(model, args.output, opset=args.opset,
                       batch=args.batch, seq_len=args.seq_len,
                       ctx_len=args.ctx_len)

    # ── 验证 ──
    verify_onnx(args.output)


if __name__ == "__main__":
    main()
