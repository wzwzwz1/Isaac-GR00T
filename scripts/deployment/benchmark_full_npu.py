#!/usr/bin/env python3
"""
Full pipeline benchmark for GR00T-N1.6 on NPU using DrivingSDK patches.
Tests backbone (Eagle VLM) + action head (DiT) end-to-end.

Usage:
    source /home/wangzhe/venvs/activate-4.51.sh
    python scripts/deployment/benchmark_full_npu.py --num_iterations 20 --warmup 5
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch_npu

# ============================================================================
# Monkey-patches for transformers 4.51 compatibility
# ============================================================================
import transformers.image_processing_utils_fast as ipuf
import transformers.image_utils as iu
import transformers.processing_utils as pu

if not hasattr(ipuf, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
    ipuf.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = "Base class for fast image processors."
if not hasattr(ipuf, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
    ipuf.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = "Preprocess method."
if not hasattr(iu, "VideoInput"):
    iu.VideoInput = iu.ImageInput
if not hasattr(iu, "make_batched_videos"):
    iu.make_batched_videos = lambda x: x


# =========================================================================
# Layer 1: 310P3 compatibility patches (REQUIRED for 310P3, not needed on 910B)
# These fix ops that 310P3 CANN kernels don't support at all.
# =========================================================================
def apply_310p3_compat_patches():
    """Apply 310P3 compatibility workarounds (unfold, complex, SDPA, dtype)."""
    print("[310P3] Applying compatibility patches...")

    # 1. F.unfold (used by SigLIP2 window attention) -> manual reshape
    _orig_unfold = torch.nn.functional.unfold
    def _npu_unfold(input, kernel_size, stride=1, padding=0, dilation=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if stride == kernel_size and padding == 0 and dilation == 1:
            B, C, H, W = input.shape
            kh, kw = kernel_size
            H_out, W_out = H // kh, W // kw
            x = input.reshape(B, C, H_out, kh, W_out, kw)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            return x.reshape(B, C * kh * kw, H_out * W_out)
        return _orig_unfold(input, kernel_size, dilation, padding, stride)
    torch.nn.functional.unfold = _npu_unfold
    print("[310P3]   F.unfold -> manual reshape")

    # 2. F.scaled_dot_product_attention -> manual matmul+softmax
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _npu_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, **kwargs):
        B, H, Lq, D = query.shape
        Lk = key.shape[-2]
        if scale is None:
            scale = 1.0 / (D ** 0.5)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal and attn_mask is None:
            causal_mask = torch.triu(torch.ones(Lq, Lk, device=query.device,
                dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask
        attn_weights = torch.softmax(attn_weights, dim=-1).to(query.dtype)
        return torch.matmul(attn_weights, value)
    torch.nn.functional.scaled_dot_product_attention = _npu_sdpa
    print("[310P3]   F.scaled_dot_product_attention -> manual matmul")

    # 3. Complex ops (torch.polar/view_as_complex/view_as_real)
    # Used by SigLIP2 2D RoPE
    def _npu_polar(abs_, angle):
        return torch.stack([abs_ * torch.cos(angle), abs_ * torch.sin(angle)], dim=-1)
    def _npu_view_as_complex(x):
        return x.reshape(list(x.shape[:-1]) + [-1, 2])
    def _npu_view_as_real(x):
        return x.flatten(-2)

    torch.polar = _npu_polar
    torch.view_as_complex = _npu_view_as_complex
    torch.view_as_real = _npu_view_as_real

    # Also patch SigLIP2's _apply_rope_2d to avoid complex multiply
    def _npu_apply_rope_2d(xq, xk, freqs_cis):
        cos_ = freqs_cis[..., 0].unsqueeze(-2)
        sin_ = freqs_cis[..., 1].unsqueeze(-2)
        xq_shape = xq.shape
        xq_ = xq.float().reshape(*xq_shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xq_shape[:-1], -1, 2)
        a, b = xq_[..., 0], xq_[..., 1]
        xq_out = torch.stack([a*cos_ - b*sin_, a*sin_ + b*cos_], dim=-1).flatten(-2).to(xq.dtype)
        a, b = xk_[..., 0], xk_[..., 1]
        xk_out = torch.stack([a*cos_ - b*sin_, a*sin_ + b*cos_], dim=-1).flatten(-2).to(xk.dtype)
        return xq_out, xk_out

    try:
        import importlib
        siglip2_mod = importlib.import_module(
            "transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2")
        siglip2_mod._apply_rope_2d = _npu_apply_rope_2d
        print("[310P3]   SigLIP2 _apply_rope_2d -> real arithmetic")
    except Exception:
        print("[310P3]   SigLIP2 RoPE: using global torch.polar patch")

    print("[310P3]   torch.polar/view_as_complex/view_as_real -> real arithmetic")
    print()

# =========================================================================
# Layer 2: DrivingSDK NPU optimization patches (op substitution, toggleable)
# =========================================================================
def apply_drivingsdk_patches():
    """Apply DrivingSDK op substitution patches (RMSNorm, RoPE, FlashAttn)."""
    from transformers.models.qwen3 import modeling_qwen3

    print("[DrivingSDK] Applying op substitution patches...")

    # P1: RMSNorm -> npu_rms_norm
    def _npu_rmsnorm_forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight,
                                       epsilon=self.variance_epsilon)[0]

    if hasattr(modeling_qwen3, "Qwen3RMSNorm"):
        modeling_qwen3.Qwen3RMSNorm.forward = _npu_rmsnorm_forward
        print("[DrivingSDK] P1: Qwen3RMSNorm -> npu_rms_norm")
    else:
        print("[DrivingSDK] P1: SKIP — Qwen3RMSNorm not found")

    # P2: RoPE -> npu_rotary_mul
    def _npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (torch_npu.npu_rotary_mul(q, cos, sin),
                torch_npu.npu_rotary_mul(k, cos, sin))

    if hasattr(modeling_qwen3, "apply_rotary_pos_emb"):
        modeling_qwen3.apply_rotary_pos_emb = _npu_apply_rotary_pos_emb
        print("[DrivingSDK] P2: apply_rotary_pos_emb -> npu_rotary_mul")
    else:
        print("[DrivingSDK] P2: SKIP — apply_rotary_pos_emb not found")

    # P3: flash_attn_func -> npu_fusion_attention
    # NOTE: npu_fusion_attention kernel is NOT available on 310P3.
    # This patch is loaded but the fused kernel will fail at runtime.
    # The 310P3 SDPA patch (Layer 1) catches actual attention calls.
    _attn_cache = {}
    def _get_attn_mask(device):
        if device not in _attn_cache:
            _attn_cache[device] = torch.triu(
                torch.ones([2048, 2048], device=device), diagonal=1).bool()
        return _attn_cache[device]
    sparse_mode = int(os.getenv("NPU_FA2_SPARSE_MODE", "3"))

    try:
        from transformers.integrations import npu_flash_attention
        def _npu_fa_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
            keep_prob = 1.0 - dropout_p
            hn = q.shape[2]
            if not causal:
                return torch_npu.npu_fusion_attention(
                    q, k, v, hn, "BSND", keep_prob=keep_prob, scale=softmax_scale)[0]
            return torch_npu.npu_fusion_attention(
                q, k, v, hn, "BSND", keep_prob=keep_prob,
                scale=softmax_scale, atten_mask=_get_attn_mask(q.device),
                sparse_mode=sparse_mode)[0]
        npu_flash_attention.flash_attn_func = _npu_fa_func
        print("[DrivingSDK] P3: flash_attn_func -> npu_fusion_attention (WILL FAIL on 310P3)")
    except Exception as e:
        print(f"[DrivingSDK] P3: SKIP — {e}")

    # P4: AttnProcessor2_0 -> NPU attn
    # NOTE: This is our own manual matmul version (DrivingSDK original also uses
    # npu_fusion_attention which doesn't work on 310P3). Still applied because
    # the default SDPA path is already patched by Layer 1.
    try:
        from diffusers.models import attention_processor as ap
        from diffusers.models.attention_processor import Attention

        if hasattr(ap, "AttnProcessor2_0"):
            def _npu_attn_call(
                self, attn: Attention, hidden_states: torch.Tensor,
                encoder_hidden_states=None, attention_mask=None, temb=None,
                *args, **kwargs,
            ) -> torch.Tensor:
                residual = hidden_states
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)
                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    B, C, H, W = hidden_states.shape
                    hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)
                elif input_ndim == 5:
                    B, C, H, W, F = hidden_states.shape
                    hidden_states = hidden_states.view(B, C, H * W, F)
                    hidden_states = hidden_states.permute(0, 3, 2, 1).reshape(B * F, H * W, C)

                B, S, _ = hidden_states.shape
                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                scale = head_dim ** -0.5

                query = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(B, -1, attn.heads, head_dim).transpose(1, 2)

                attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
                if attention_mask is not None:
                    if attention_mask.ndim == 2:
                        attention_mask = attention_mask[:, None, None, :]
                    elif attention_mask.ndim == 3:
                        attention_mask = attention_mask[:, None, :, :]
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.softmax(attn_weights, dim=-1).to(query.dtype)
                hidden_states = torch.matmul(attn_weights, value)
                hidden_states = hidden_states.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)

                hidden_states = attn.to_out[0](hidden_states)
                if len(attn.to_out) > 1:
                    hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)
                elif input_ndim == 5:
                    hidden_states = hidden_states.reshape(B, F, H * W, C)
                    hidden_states = hidden_states.permute(0, 3, 2, 1).reshape(B, C, H, W, F)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states

            ap.AttnProcessor2_0.__call__ = _npu_attn_call
            print("[DrivingSDK] P4: AttnProcessor2_0 -> manual matmul (310P3 compat)")
    except Exception as e:
        print(f"[DrivingSDK] P4: SKIP — {e}")

    print()


def load_model_and_processor(model_path: str, device_id: int = 0):
    """Load GR00T-N1.6 model and processor."""
    import gr00t.model  # noqa: F401
    from transformers import AutoModel, AutoProcessor

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)

    device = torch.device(f"npu:{device_id}")
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    dit_params = sum(p.numel() for p in model.action_head.model.parameters())
    print(f"Model on {device} | {total_params/1e9:.1f}B params (DiT: {dit_params/1e6:.0f}M)")
    print(f"  Backbone layers: {len(model.backbone.model.language_model.model.layers)}")
    print(f"  DiT layers: {model.config.diffusion_model_cfg['num_layers']}")
    print(f"  Denoising steps: {model.config.num_inference_timesteps}")
    print()
    return model, processor


def create_synthetic_observation(processor, embodiment_tag="gr1"):
    """Create a synthetic observation matching the expected format."""
    configs = processor.get_modality_configs()
    mods = configs[embodiment_tag]

    obs = {}

    # Video: (T, H, W, C) uint8 per camera
    obs["video"] = {}
    for vk in mods["video"].modality_keys:
        obs["video"][vk] = np.random.randint(0, 255,
            size=(1, 1, 256, 256, 3), dtype=np.uint8)

    # State: (T, D) float32 — correct dimensions per key
    obs["state"] = {}
    for sk in mods["state"].modality_keys:
        dim = 7  # default
        if "hand" in sk:
            dim = 6
        elif "waist" in sk:
            dim = 3
        obs["state"][sk] = np.random.randn(1, 1, dim).astype(np.float32)

    # Language
    obs["language"] = {}
    for lk in mods["language"].modality_keys:
        obs["language"][lk] = [["do something"]]

    return obs


def benchmark_full_pipeline(processor, obs, n_iter=20, warmup=5):
    """Benchmark the full pipeline: processor + backbone + action_head."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    print("=" * 60)
    print("Full Pipeline Benchmark (processor + backbone + action_head)")
    print("=" * 60)

    # Create policy once, reuse across iterations
    device = torch.device(f"npu:{torch_npu.npu.current_device()}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag("gr1"),
        model_path="/home/wangzhe/models/GR00T-N1.6-3B",
        device=device,
        strict=False,
    )

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = policy.get_action(obs)
    torch_npu.npu.synchronize()

    times = []
    for i in range(n_iter):
        torch_npu.npu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.get_action(obs)
        torch_npu.npu.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  Min:    {np.min(times):.1f} ms")
    print(f"  Max:    {np.max(times):.1f} ms")
    print(f"  P90:    {np.percentile(times, 90):.1f} ms")
    print(f"  Freq:   {1000/np.median(times):.2f} Hz")
    return times


def benchmark_backbone(model, processor, obs, n_iter=20, warmup=5):
    """Benchmark only the Eagle backbone."""
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.data.embodiment_tags import EmbodimentTag

    print("=" * 60)
    print("Eagle Backbone Benchmark")
    print("=" * 60)

    # Prepare a single processed batch
    modality_configs = processor.get_modality_configs()["gr1"]
    language_key = modality_configs["language"].modality_keys[0]

    # Set processor to eval mode (avoids requiring actions)
    processor.eval()

    # Build processed input once
    step_data = VLAStepData(
        images={"ego_view_bg_crop_pad_res256_freq20": obs["video"]["ego_view_bg_crop_pad_res256_freq20"][0]},
        states={k: v[0] for k, v in obs["state"].items()},
        actions={},
        text=obs["language"][language_key][0][0],
        embodiment=EmbodimentTag("gr1"),
    )
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
    processed = processor(messages)
    collated = processor.collator([processed])["inputs"]

    # Move to device
    import tree
    def to_npu(x):
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                return x.to(model.device, dtype=next(model.parameters()).dtype)
            return x.to(model.device)
        return x
    collated = tree.map_structure(to_npu, collated)

    backbone_inputs = model.backbone.prepare_input(collated)
    backbone_inputs = tree.map_structure(to_npu, backbone_inputs.data)

    from transformers.feature_extraction_utils import BatchFeature
    backbone_inputs = BatchFeature(data=backbone_inputs)

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model.backbone(backbone_inputs)
    torch_npu.npu.synchronize()

    times = []
    for i in range(n_iter):
        torch_npu.npu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.backbone(backbone_inputs)
        torch_npu.npu.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  Min:    {np.min(times):.1f} ms")
    print(f"  Max:    {np.max(times):.1f} ms")
    print(f"  P90:    {np.percentile(times, 90):.1f} ms")
    return times


def report_memory():
    """Report NPU memory usage."""
    for i in range(torch_npu.npu.device_count()):
        mi = torch_npu.npu.mem_get_info(i)
        total = mi[1] / 1024**3
        used = (mi[1] - mi[0]) / 1024**3
        free = mi[0] / 1024**3
        print(f"  NPU:{i} total={total:.1f}GB used={used:.1f}GB free={free:.1f}GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/wangzhe/models/GR00T-N1.6-3B")
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--no_drivingsdk", action="store_true",
                        help="Disable DrivingSDK patches (only 310P3 compat)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ["LOGURU_LEVEL"] = "WARNING"

    print("=" * 80)
    print("GR00T-N1.6 FULL PIPELINE NPU BENCHMARK")
    print("=" * 80)
    print(f"Device: NPU:{args.device_id} ({torch_npu.npu.get_device_name(0)})")
    print(f"Iterations: {args.num_iterations}, Warmup: {args.warmup}")
    print(f"310P3 compat patches:  ALWAYS ON (F.unfold, complex, SDPA, dtype)")
    print(f"DrivingSDK patches:    {'OFF' if args.no_drivingsdk else 'ON'} (RMSNorm, RoPE, FlashAttn, AttnProc)")
    print()

    torch.npu.set_device(args.device_id)

    # Layer 1: 310P3 compat (always needed on 310P3)
    apply_310p3_compat_patches()

    # Layer 2: DrivingSDK op substitutions (toggleable)
    if not args.no_drivingsdk:
        apply_drivingsdk_patches()

    # Load model + processor
    print("Loading model...")
    print("-" * 40)
    t0 = time.perf_counter()
    model, processor = load_model_and_processor(args.model_path, args.device_id)
    print(f"Load time: {time.perf_counter() - t0:.1f}s")

    print("NPU Memory:")
    report_memory()

    # Create synthetic observation
    obs = create_synthetic_observation(processor, "gr1")
    print(f"\nSynthetic observation: gr1 tag")
    print(f"  video: {list(obs['video'].keys())}")
    print(f"  state: {list(obs['state'].keys())}")
    print(f"  images: {obs['video']['ego_view_bg_crop_pad_res256_freq20'].shape}")

    # Benchmark backbone (uses raw model.backbone directly)
    gc.collect()
    torch_npu.npu.empty_cache()
    bb_no_patch_times = benchmark_backbone(model, processor, obs,
                                            args.num_iterations, args.warmup)

    # Benchmark full pipeline (uses Gr00tPolicy wrapper)
    # Note: Gr00tPolicy loads model from path again - we accept the double memory
    # to get a clean, complete pipeline measurement
    gc.collect()
    torch_npu.npu.empty_cache()
    # Delete model to free memory before policy loads its own copy
    del model
    torch_npu.npu.empty_cache()
    full_times = benchmark_full_pipeline(processor, obs,
                                          args.num_iterations, args.warmup)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Backbone (Eagle VLM):     {np.median(bb_no_patch_times):.1f} ms")
    print(f"  Full pipeline:             {np.median(full_times):.1f} ms")
    action_head_derived = np.median(full_times) - np.median(bb_no_patch_times)
    print(f"  Action head (derived):     {action_head_derived:.1f} ms")
    print(f"  Frequency:                 {1000/np.median(full_times):.2f} Hz")
    print(f"  Patches:                   {'OFF' if args.no_drivingsdk else 'ON'}")

    print("\nMemory after benchmark:")
    report_memory()


if __name__ == "__main__":
    main()
