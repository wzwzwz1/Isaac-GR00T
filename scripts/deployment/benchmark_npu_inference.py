#!/usr/bin/env python3
"""
NPU Inference Benchmark for GR00T-N1.6 using DrivingSDK optimizations.

Applies NPU-specific patches (npu_rms_norm, npu_rotary_mul, npu_fusion_attention)
from the Ascend DrivingSDK project and benchmarks the model on Ascend NPU.

Usage:
    python scripts/deployment/benchmark_npu_inference.py \
        --model_path /home/wangzhe/models/GR00T-N1.6-3B \
        --num_iterations 50 --warmup 10
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch_npu

# ============================================================================
# Monkey-patches for transformers 4.57+ compatibility
# ============================================================================
import transformers.image_processing_utils_fast as ipuf
import transformers.image_utils as iu
import transformers.processing_utils as pu

if not hasattr(ipuf, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
    ipuf.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = "Base class for fast image processors."
if not hasattr(ipuf, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
    ipuf.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = (
        "Preprocess method for fast image processors."
    )
if not hasattr(iu, "VideoInput"):
    iu.VideoInput = iu.ImageInput
if not hasattr(iu, "make_batched_videos"):
    iu.make_batched_videos = lambda x: x


def apply_drivingsdk_patches():
    """Apply NPU optimization patches from DrivingSDK GR00T-N1.6 example."""

    # --- RMSNorm patch: replace with npu_rms_norm ---
    from transformers.models.qwen3 import modeling_qwen3

    def rmsnorm_forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    if hasattr(modeling_qwen3, "Qwen3RMSNorm"):
        modeling_qwen3.Qwen3RMSNorm.forward = rmsnorm_forward
        print("[Patch] Qwen3RMSNorm -> npu_rms_norm")
    else:
        print("[Patch] WARNING: Qwen3RMSNorm not found, skipping RMSNorm patch")

    # --- RoPE patch: replace with npu_rotary_mul ---
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
        return q_embed, k_embed

    if hasattr(modeling_qwen3, "apply_rotary_pos_emb"):
        modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb
        print("[Patch] apply_rotary_pos_emb -> npu_rotary_mul")
    else:
        print("[Patch] WARNING: apply_rotary_pos_emb not found, skipping RoPE patch")

    # --- FlashAttention patch ---
    _attn_mask_npu_cache = {}

    def _get_attn_mask_npu(device):
        if device not in _attn_mask_npu_cache:
            _attn_mask_npu_cache[device] = torch.triu(
                torch.ones([2048, 2048], device=device), diagonal=1
            ).bool()
        return _attn_mask_npu_cache[device]

    sparse_mode = int(os.getenv("NPU_FA2_SPARSE_MODE", "3"))

    # Patch flash_attn_func
    try:
        from transformers.integrations import npu_flash_attention

        def npu_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
            keep_prob = 1.0 - dropout_p
            head_num = q.shape[2]
            if not causal:
                output = torch_npu.npu_fusion_attention(
                    q, k, v, head_num, "BSND", keep_prob=keep_prob, scale=softmax_scale,
                )[0]
            else:
                attn_mask = _get_attn_mask_npu(q.device)
                output = torch_npu.npu_fusion_attention(
                    q, k, v, head_num, "BSND", keep_prob=keep_prob,
                    scale=softmax_scale, atten_mask=attn_mask, sparse_mode=sparse_mode,
                )[0]
            return output

        npu_flash_attention.flash_attn_func = npu_flash_attn_func
        print("[Patch] flash_attn_func -> npu_fusion_attention")
    except ImportError:
        print("[Patch] WARNING: npu_flash_attention not found")

    # Patch flash_attn_varlen_func
    try:
        def npu_flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
        ):
            keep_prob = 1.0 - dropout_p
            head_num = q.shape[1]
            if not causal:
                output = torch_npu.npu_fusion_attention(
                    q, k, v, head_num, pse=None, atten_mask=None,
                    scale=softmax_scale, keep_prob=keep_prob, input_layout="TND",
                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                )[0]
            else:
                attn_mask = _get_attn_mask_npu(q.device)
                output = torch_npu.npu_fusion_attention(
                    q, k, v, head_num, pse=None, padding_mask=None,
                    atten_mask=attn_mask, scale=softmax_scale, keep_prob=keep_prob,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                    sparse_mode=sparse_mode,
                )[0]
            return output

        npu_flash_attention.flash_attn_varlen_func = npu_flash_attn_varlen_func
        print("[Patch] flash_attn_varlen_func -> npu_fusion_attention")
    except Exception:
        pass

    # Also patch modeling_flash_attention_utils
    try:
        from transformers import modeling_flash_attention_utils as mfau
        if hasattr(mfau, "flash_attn_func"):
            mfau.flash_attn_func = npu_flash_attn_func
        if hasattr(mfau, "flash_attn_varlen_func"):
            mfau.flash_attn_varlen_func = npu_flash_attn_varlen_func
        print("[Patch] modeling_flash_attention_utils flash_attn -> npu_fusion_attention")
    except ImportError:
        pass

    # --- NPU Attention patch for DiT ---
    # npu_fusion_attention / FlashAttentionScore kernel may not be available
    # on all NPU configurations (e.g., 310P3 with certain CANN versions).
    # We implement attention using basic matmul+softmax which works everywhere.
    try:
        import torch.nn.functional as F
        from diffusers.models import attention_processor as ap
        from diffusers.models.attention_processor import Attention

        if hasattr(ap, "AttnProcessor2_0"):
            def _npu_attn_call(
                self,
                attn: Attention,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                temb: torch.Tensor | None = None,
                *args,
                **kwargs,
            ) -> torch.Tensor:
                """NPU-compatible attention using basic matmul+softmax operations."""
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                elif input_ndim == 5:
                    batch_size, channel, height, width, frames = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width, frames)
                    hidden_states = hidden_states.permute(0, 3, 2, 1).reshape(batch_size * frames, height * width, channel)

                batch_size, sequence_length, _ = hidden_states.shape

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                # QKV projection
                query = attn.to_q(hidden_states)
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                # Manual multi-head attention using basic ops
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                scale = head_dim ** -0.5

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Attention scores: Q @ K^T
                attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

                if attention_mask is not None:
                    # Adjust mask shape for multi-head format
                    if attention_mask.ndim == 2:
                        attention_mask = attention_mask[:, None, None, :]
                    elif attention_mask.ndim == 3:
                        attention_mask = attention_mask[:, None, :, :]
                    attn_weights = attn_weights + attention_mask

                attn_weights = torch.softmax(attn_weights, dim=-1).to(query.dtype)

                # Weighted sum
                hidden_states = torch.matmul(attn_weights, value)

                # Reshape back
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

                # Output projection
                hidden_states = attn.to_out[0](hidden_states)
                if len(attn.to_out) > 1:
                    hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                elif input_ndim == 5:
                    hidden_states = hidden_states.reshape(batch_size, frames, height * width, channel)
                    hidden_states = hidden_states.permute(0, 3, 2, 1)
                    hidden_states = hidden_states.reshape(batch_size, channel, height, width, frames)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states

            ap.AttnProcessor2_0.__call__ = _npu_attn_call
            print("[Patch] AttnProcessor2_0 -> manual matmul attention (NPU-compatible)")
    except Exception as e:
        print(f"[Patch] WARNING: Failed to patch AttnProcessor2_0: {e}")

    print("[Patch] All DrivingSDK patches applied.\n")


def load_model_on_npu(model_path: str):
    """Load GR00T-N1.6 model on NPU."""
    import gr00t.model  # noqa: F401 - register model
    from transformers import AutoModel

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16)

    # Move to NPU
    device = torch.device("npu:0")
    model = model.to(device)
    model.eval()

    # Collect model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = model.config
    print(f"Model loaded on NPU:0")
    print(f"  Type: {type(model).__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  DiT layers: {config.diffusion_model_cfg['num_layers']}")
    print(f"  DiT heads: {config.diffusion_model_cfg['num_attention_heads']}")
    print(f"  Action horizon: {config.action_horizon}")
    print(f"  Inference timesteps: {config.num_inference_timesteps}")
    print(f"  Backbone embedding dim: {config.backbone_embedding_dim}")
    print(f"  Input embedding dim: {config.input_embedding_dim}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print()

    return model


def create_synthetic_inputs(model, batch_size=1):
    """
    Create synthetic inputs matching the model's expected input format.
    We create backbone_output and action_input directly, bypassing the
    Eagle VLM processor (which requires a compatible transformers version).
    """
    config = model.config
    device = next(model.parameters()).device
    # Use the DiT's dtype (float16) since the action head operates in fp16
    dtype = model.action_head.model.transformer_blocks[0].attn1.to_q.weight.dtype

    # Backbone output: what Eagle VLM produces
    # - backbone_features: [B, seq_len, backbone_embedding_dim]
    # - backbone_attention_mask: [B, seq_len]
    # - image_mask: [B, seq_len]
    seq_len = 512  # typical VLM output sequence length
    backbone_features = torch.randn(batch_size, seq_len, config.backbone_embedding_dim,
                                    device=device, dtype=dtype)
    backbone_attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    image_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    # Set some positions as image tokens
    image_mask[:, :50] = True

    from transformers.feature_extraction_utils import BatchFeature
    backbone_output = BatchFeature(data={
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask,
        "image_mask": image_mask,
    })

    # Action input
    state_horizon = 1  # single state step
    state_dim = model.action_head.state_encoder.layer1.W.shape[1]  # e.g., 128
    state = torch.randn(batch_size, state_horizon, state_dim, device=device, dtype=dtype)
    embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

    action_input = BatchFeature(data={
        "state": state,
        "embodiment_id": embodiment_id,
    })

    return backbone_output, action_input


def benchmark_backbone(model, backbone_inputs, num_iterations=50, warmup=10):
    """Benchmark only the backbone (Eagle VLM) forward pass."""
    print(f"\n{'='*60}")
    print("Benchmark: Backbone (VLM)")
    print(f"{'='*60}")

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model.backbone(backbone_inputs)
    torch_npu.npu.synchronize()

    times = []
    for i in range(num_iterations):
        torch_npu.npu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.backbone(backbone_inputs)
        torch_npu.npu.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  P90:    {np.percentile(times, 90):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    return times


def benchmark_action_head(model, backbone_output, action_input, num_iterations=50, warmup=10):
    """Benchmark the action head (including DiT with N denoising steps)."""
    denoising_steps = model.config.num_inference_timesteps
    print(f"\n{'='*60}")
    print(f"Benchmark: Action Head (DiT, {denoising_steps} denoising steps)")
    print(f"{'='*60}")

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model.action_head.get_action(backbone_output, action_input)
    torch_npu.npu.synchronize()

    times = []
    for i in range(num_iterations):
        # Create fresh random noise at each step (as in real inference)
        torch_npu.npu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.action_head.get_action(backbone_output, action_input)
        torch_npu.npu.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  P90:    {np.percentile(times, 90):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    print(f"  Per denoising step: {np.median(times) / denoising_steps:.2f} ms")
    return times


def benchmark_dit_only(model, backbone_output, action_input, num_iterations=50, warmup=10):
    """Benchmark a single DiT forward pass (one denoising step)."""
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import BatchFeature

    # Prepare features once (like in _encode_features + get_action_with_features)
    action_head = model.action_head
    config = action_head.config

    with torch.inference_mode():
        features = action_head._encode_features(backbone_output, action_input)
        backbone_features = features.backbone_features
        state_features = features.state_features
        embodiment_id = action_input.embodiment_id

        batch_size = backbone_features.shape[0]
        device = backbone_features.device
        dtype = backbone_features.dtype

        actions = torch.randn(
            size=(batch_size, config.action_horizon, action_head.action_dim),
            dtype=dtype, device=device,
        )
        t_discretized = 0
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device,
        )
        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)

        if config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_embs = backbone_features

    print(f"\n{'='*60}")
    print("Benchmark: Single DiT Forward Pass")
    print(f"{'='*60}")

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
    torch_npu.npu.synchronize()

    times = []
    for i in range(num_iterations):
        torch_npu.npu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        torch_npu.npu.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    times = np.array(times)
    print(f"  Input shape: sa_embs={list(sa_embs.shape)}, vl_embs={list(vl_embs.shape)}")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  P90:    {np.percentile(times, 90):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    return times


def benchmark_memory(model):
    """Report NPU memory usage."""
    print(f"\n{'='*60}")
    print("NPU Memory Usage")
    print(f"{'='*60}")
    for i in range(torch_npu.npu.device_count()):
        mem_info = torch_npu.npu.mem_get_info(i)
        total = mem_info[1] / (1024 ** 3)
        used = (mem_info[1] - mem_info[0]) / (1024 ** 3)
        free = mem_info[0] / (1024 ** 3)
        print(f"  NPU:{i} - Total: {total:.1f} GB, Used: {used:.1f} GB, Free: {free:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GR00T-N1.6 inference on Ascend NPU")
    parser.add_argument("--model_path", type=str, default="/home/wangzhe/models/GR00T-N1.6-3B",
                        help="Path to the model checkpoint")
    parser.add_argument("--num_iterations", type=int, default=50,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--apply_patches", action="store_true", default=True,
                        help="Apply DrivingSDK NPU patches")
    parser.add_argument("--no_patches", action="store_true",
                        help="Skip DrivingSDK NPU patches (run vanilla)")
    parser.add_argument("--device_id", type=int, default=0,
                        help="NPU device ID")
    args = parser.parse_args()

    # Suppress verbose warnings
    warnings.filterwarnings("ignore")
    os.environ["LOGURU_LEVEL"] = "WARNING"

    print("=" * 80)
    print("GR00T-N1.6 NPU INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Batch size: {args.batch_size}")
    print(f"NPU device: {args.device_id}")
    print(f"NPU devices available: {torch_npu.npu.device_count()}")
    print(f"NPU device name: {torch_npu.npu.get_device_name(args.device_id)}")

    # Set NPU device
    torch.npu.set_device(args.device_id)

    apply_patches_flag = args.apply_patches and not args.no_patches

    # Step 1: Apply DrivingSDK patches (if enabled)
    if apply_patches_flag:
        print(f"\n{'='*60}")
        print("Applying DrivingSDK NPU patches...")
        print(f"{'='*60}")
        apply_drivingsdk_patches()
    else:
        print("\n[Skipping NPU patches - running vanilla PyTorch on NPU]")

    # Step 2: Load model
    print(f"{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    model_load_start = time.perf_counter()
    model = load_model_on_npu(args.model_path)
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model load time: {model_load_time:.2f}s")

    benchmark_memory(model)

    # Step 3: Create synthetic inputs
    print(f"\n{'='*60}")
    print("Creating Synthetic Inputs")
    print(f"{'='*60}")
    backbone_output, action_input = create_synthetic_inputs(model, args.batch_size)
    print(f"  backbone_features shape: {backbone_output.backbone_features.shape}")
    print(f"  state shape: {action_input.state.shape}")
    print(f"  embodiment_id: {action_input.embodiment_id}")

    # Need to move inputs through prepare_input for proper device/dtype handling
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import BatchFeature
    import tree

    def _to_npu(x):
        if torch.is_floating_point(x) or x.dtype in (torch.int64, torch.int32, torch.long):
            return x.to(model.device)
        elif isinstance(x, torch.Tensor) and x.dtype == torch.bool:
            return x.to(model.device)
        return x

    backbone_output_data = tree.map_structure(_to_npu, backbone_output.data)
    action_input_data = tree.map_structure(_to_npu, action_input.data)
    backbone_output = BatchFeature(data=backbone_output_data)
    action_input = BatchFeature(data=action_input_data)

    # Step 4: Run benchmarks
    gc.collect()
    torch_npu.npu.empty_cache()

    # 4a: Single DiT forward pass
    dit_times = benchmark_dit_only(
        model, backbone_output, action_input, args.num_iterations, args.warmup,
    )

    # 4b: Action head (full diffusion: 4 denoising steps)
    gc.collect()
    torch_npu.npu.empty_cache()

    action_head_times = benchmark_action_head(
        model, backbone_output, action_input, args.num_iterations, args.warmup,
    )

    # Summary
    denoising_steps = model.config.num_inference_timesteps
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Model loading:                 {model_load_time:.2f}s")
    print(f"  Dit params:                    {sum(p.numel() for p in model.action_head.model.parameters()):,}")
    print(f"  Denoising steps:               {denoising_steps}")
    print(f"  Single DiT forward (median):   {np.median(dit_times):.2f} ms")
    print(f"  Action Head total (median):    {np.median(action_head_times):.2f} ms")
    print(f"  Per-denoising-step (derived):  {np.median(action_head_times)/denoising_steps:.2f} ms")
    print(f"  Frequency:                     {1000/np.median(action_head_times):.2f} Hz")
    print()
    print("COMPARISON CHECK:")
    print(f"  action_head / denoising_steps = {np.median(action_head_times)/denoising_steps:.2f} ms")
    print(f"  single_dit_forward            = {np.median(dit_times):.2f} ms")
    print(f"  Overhead:                     {np.median(action_head_times)/denoising_steps - np.median(dit_times):.2f} ms per step")

    benchmark_memory(model)


if __name__ == "__main__":
    main()
