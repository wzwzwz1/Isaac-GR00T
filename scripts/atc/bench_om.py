#!/usr/bin/env python3
"""
Phase 3: 加载 ATC 编译的 OM 模型并进行推理 Benchmark.

对比:
  - PyTorch eager (手工 SDPA)
  - ATC OM 模型

用法:
    python3 scripts/atc/bench_om.py \
        --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
        --om_path atc_output/dit_310p3_fp16.om \
        --num_warmup 5 --num_runs 30
"""

import argparse
import ctypes
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# OM 模型运行器 (ACL 封装)
# ═══════════════════════════════════════════════════════════════════════

class DiTOMRunner:
    """使用 ACL 加载和运行 OM 格式的 DiT 模型."""

    def __init__(self, om_path: str, device_id: int = 0):
        import acl
        self.acl = acl

        # ACL 可能已被 torch_npu 初始化, 错误码 100002 表示重复初始化
        init_ret = acl.init()
        if init_ret != 0 and init_ret != 100002:  # 100002 = already initialized
            raise RuntimeError(f"ACL init failed: {init_ret}")
        ret = acl.rt.set_device(device_id)
        if ret != 0:
            raise RuntimeError(f"ACL set_device failed: {ret}")

        self.model_id, _ = acl.mdl.load_from_file(om_path)
        self._owns_acl = (init_ret == 0)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        assert ret == 0, f"get_desc failed: {ret}"

        # 获取输入输出数量和大小
        self.num_inputs = acl.mdl.get_num_inputs(self.model_desc)
        self.num_outputs = acl.mdl.get_num_outputs(self.model_desc)

        self.input_sizes = []
        for i in range(self.num_inputs):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            self.input_sizes.append(size)

        self.output_sizes = []
        for i in range(self.num_outputs):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            self.output_sizes.append(size)

        print(f"[OM Runner] Loaded: {om_path}")
        print(f"[OM Runner] Inputs: {self.num_inputs}, sizes={self.input_sizes}")
        print(f"[OM Runner] Outputs: {self.num_outputs}, sizes={self.output_sizes}")

    def run(self, inputs: list) -> list:
        """
        运行 OM 模型推理.

        Args:
            inputs: list of numpy arrays on host

        Returns:
            list of numpy arrays (output tensors on host)
        """
        acl = self.acl

        # ACL memory kind constants (from C API)
        H2D = 1   # ACL_MEMCPY_HOST_TO_DEVICE
        D2H = 2   # ACL_MEMCPY_DEVICE_TO_HOST
        MALLOC_NORMAL = 2  # ACL_MEM_MALLOC_NORMAL_ONLY

        input_dataset = acl.mdl.create_dataset()
        input_buffers = []
        for i, inp_np in enumerate(inputs):
            buf, ret = acl.rt.malloc(self.input_sizes[i], MALLOC_NORMAL)
            if ret != 0:
                raise RuntimeError(f"malloc input {i} failed: {ret}")
            acl.rt.memcpy(buf, self.input_sizes[i],
                          inp_np.ctypes.data_as(ctypes.c_void_p).value,
                          inp_np.nbytes, H2D)
            data_buf = acl.create_data_buffer(buf, self.input_sizes[i])
            acl.mdl.add_dataset_buffer(input_dataset, data_buf)
            input_buffers.append(buf)

        output_dataset = acl.mdl.create_dataset()
        output_buffers = []
        for i, out_size in enumerate(self.output_sizes):
            buf, ret = acl.rt.malloc(out_size, MALLOC_NORMAL)
            if ret != 0:
                raise RuntimeError(f"malloc output {i} failed: {ret}")
            data_buf = acl.create_data_buffer(buf, out_size)
            acl.mdl.add_dataset_buffer(output_dataset, data_buf)
            output_buffers.append(buf)

        ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        if ret != 0:
            raise RuntimeError(f"ACL execute failed: {ret}")

        outputs = []
        for i, (buf, out_size) in enumerate(zip(output_buffers, self.output_sizes)):
            out_np = np.empty(out_size, dtype=np.uint8)
            acl.rt.memcpy(out_np.ctypes.data_as(ctypes.c_void_p).value,
                          out_size, buf, out_size, D2H)
            outputs.append(out_np)

        for buf in input_buffers + output_buffers:
            acl.rt.free(buf)
        acl.mdl.destroy_dataset(input_dataset)
        acl.mdl.destroy_dataset(output_dataset)

        return outputs

    def __del__(self):
        try:
            self.acl.mdl.unload(self.model_id)
            self.acl.mdl.destroy_desc(self.model_desc)
            if self._owns_acl:
                self.acl.finalize()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════

def build_om_inputs(hidden_states, encoder_hidden_states, timestep,
                    image_mask_float, non_image_mask_float):
    """将 PyTorch NPU tensors 转为 numpy arrays (H2H)."""
    def _to_np(t, dtype=None):
        x = t.detach().cpu().numpy()
        return x.astype(dtype) if dtype else x

    inputs = [
        _to_np(hidden_states, np.float16),
        _to_np(encoder_hidden_states, np.float16),
        np.array(timestep.detach().cpu(), dtype=np.int32),
        _to_np(image_mask_float, np.float16),
        _to_np(non_image_mask_float, np.float16),
    ]
    return inputs


def parse_om_output(raw_outputs, expected_shape):
    """将 OM 原始输出字节转为 numpy tensor."""
    out_bytes = raw_outputs[0]
    return np.frombuffer(out_bytes, dtype=np.float16).reshape(expected_shape)


def main():
    parser = argparse.ArgumentParser(description="Benchmark OM vs PyTorch DiT")
    parser.add_argument("--model_path", type=str,
                        default="/home/wangzhe/models/GR00T-N1.6-3B-FP16")
    parser.add_argument("--om_path", type=str,
                        default="/home/wangzhe/Isaac-GR00T/atc_output/dit_310p3_fp16.om")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["pytorch", "om", "both"])
    args = parser.parse_args()

    sys.path.insert(0, '.')
    import gr00t.model  # noqa
    from transformers import AutoModel, BatchFeature

    # ── 加载 PyTorch 模型 ──
    print(f"[Setup] Loading PyTorch model...")
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    model.to(device='npu:0', dtype=torch.float16)

    c = model.config
    ah = model.action_head
    dit = ah.model
    device = next(model.parameters()).device

    B, SEQ, CTX = 1, 51, 256

    # ── 构造输入 ──
    state = torch.randn(B, 1, c.max_state_dim, dtype=torch.float16, device=device)
    bb_feat = torch.randn(B, CTX, c.backbone_embedding_dim, dtype=torch.float16, device=device)
    bb_am = torch.ones(B, CTX, dtype=torch.bool, device=device)
    img_mask = torch.zeros(B, CTX, dtype=torch.bool, device=device)
    img_mask[:, :CTX//2] = True

    bb_out = BatchFeature({'backbone_features': bb_feat,
                           'backbone_attention_mask': bb_am,
                           'image_mask': img_mask})
    act_in = BatchFeature({'state': state,
                           'embodiment_id': torch.zeros(B, dtype=torch.long, device=device)})

    # ── 预计算 encode ──
    feats = ah._encode_features(bb_out, act_in)
    vl_emb = feats.backbone_features
    st_feat = feats.state_features
    emb_id = act_in.embodiment_id

    # ── 预计算 OM 输入 ──
    img_am = img_mask & bb_am
    non_img_am = (~img_mask) & bb_am
    img_mask_float = torch.where(img_am, torch.zeros_like(img_mask, dtype=torch.float16),
                                  torch.full_like(img_mask, -10000.0, dtype=torch.float16))
    non_img_mask_float = torch.where(non_img_am, torch.zeros_like(img_mask, dtype=torch.float16),
                                      torch.full_like(img_mask, -10000.0, dtype=torch.float16))

    om_inputs_static = [img_mask_float, non_img_mask_float, vl_emb]

    # ── SDPA monkey-patch ──
    _orig_sdpa = F.scaled_dot_product_attention
    def manual_attn(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False):
        d_k = query.shape[-1]
        s = scale if scale is not None else 1.0 / (d_k ** 0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * s
        if attn_mask is not None:
            if attn_mask.dim() == 3: attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2: attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            scores = scores + attn_mask
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, value)
    F.scaled_dot_product_attention = manual_attn

    # ════════════════════════════════════════════════════════════
    # Benchmark: PyTorch Eager
    # ════════════════════════════════════════════════════════════

    if args.mode in ("pytorch", "both"):
        print(f"\n[Bench PyTorch] warmup={args.num_warmup}, runs={args.num_runs}...")
        for _ in range(args.num_warmup):
            actions = torch.randn((B, c.action_horizon, c.max_action_dim),
                                  dtype=torch.float16, device=device)
            dt_val = 1.0 / ah.num_inference_timesteps
            for t_idx in range(ah.num_inference_timesteps):
                t_cont = t_idx / float(ah.num_inference_timesteps)
                t_disc = int(t_cont * ah.num_timestep_buckets)
                ts = torch.full((B,), fill_value=t_disc, device=device)
                act_feat = ah.action_encoder(actions, ts, emb_id)
                if c.add_pos_embed:
                    pids = torch.arange(act_feat.shape[1], dtype=torch.long, device=device)
                    act_feat = act_feat + ah.position_embedding(pids).unsqueeze(0)
                sa_embs = torch.cat((st_feat, act_feat), dim=1)

                torch.npu.synchronize()
                m_out = dit(hidden_states=sa_embs, encoder_hidden_states=vl_emb,
                           timestep=ts, image_mask=img_mask,
                           backbone_attention_mask=bb_am)
                torch.npu.synchronize()

                pred_v = ah.action_decoder(m_out, emb_id)[:, -c.action_horizon:]
                actions = actions + dt_val * pred_v

        pytorch_times = []
        for run in range(args.num_runs):
            actions = torch.randn((B, c.action_horizon, c.max_action_dim),
                                  dtype=torch.float16, device=device)
            dt_val = 1.0 / ah.num_inference_timesteps
            for t_idx in range(ah.num_inference_timesteps):
                t_cont = t_idx / float(ah.num_inference_timesteps)
                t_disc = int(t_cont * ah.num_timestep_buckets)
                ts = torch.full((B,), fill_value=t_disc, device=device)
                act_feat = ah.action_encoder(actions, ts, emb_id)
                if c.add_pos_embed:
                    pids = torch.arange(act_feat.shape[1], dtype=torch.long, device=device)
                    act_feat = act_feat + ah.position_embedding(pids).unsqueeze(0)
                sa_embs = torch.cat((st_feat, act_feat), dim=1)

                torch.npu.synchronize()
                t0 = time.perf_counter()
                m_out = dit(hidden_states=sa_embs, encoder_hidden_states=vl_emb,
                           timestep=ts, image_mask=img_mask,
                           backbone_attention_mask=bb_am)
                torch.npu.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                pytorch_times.append(elapsed)

                pred_v = ah.action_decoder(m_out, emb_id)[:, -c.action_horizon:]
                actions = actions + dt_val * pred_v

        arr = np.array(pytorch_times)
        print(f"  PyTorch DiT step: mean={np.mean(arr):.1f}ms  p95={np.percentile(arr,95):.1f}ms  "
              f"min={np.min(arr):.1f}ms")
        pytorch_mean = np.mean(arr)
    else:
        pytorch_mean = None

    # ════════════════════════════════════════════════════════════
    # Benchmark: OM model
    # ════════════════════════════════════════════════════════════

    if args.mode in ("om", "both"):
        om_runner = DiTOMRunner(args.om_path)

        # Warmup OM
        print(f"\n[Bench OM] warmup={args.num_warmup}, runs={args.num_runs}...")
        for _ in range(args.num_warmup):
            actions = torch.randn((B, c.action_horizon, c.max_action_dim),
                                  dtype=torch.float16, device=device)
            dt_val = 1.0 / ah.num_inference_timesteps
            for t_idx in range(ah.num_inference_timesteps):
                t_cont = t_idx / float(ah.num_inference_timesteps)
                t_disc = int(t_cont * ah.num_timestep_buckets)
                ts = torch.full((B,), fill_value=t_disc, device=device)

                act_feat = ah.action_encoder(actions, ts, emb_id)
                if c.add_pos_embed:
                    pids = torch.arange(act_feat.shape[1], dtype=torch.long, device=device)
                    act_feat = act_feat + ah.position_embedding(pids).unsqueeze(0)
                sa_embs = torch.cat((st_feat, act_feat), dim=1)

                om_inputs = build_om_inputs(sa_embs, vl_emb, ts,
                                             om_inputs_static[0],
                                             om_inputs_static[1])
                raw = om_runner.run(om_inputs)
                m_out_np = parse_om_output(raw, (1, SEQ, 1024))

        # Benchmark OM
        om_times = []
        om_ref_output = None
        for run in range(args.num_runs):
            actions = torch.randn((B, c.action_horizon, c.max_action_dim),
                                  dtype=torch.float16, device=device)
            dt_val = 1.0 / ah.num_inference_timesteps
            for t_idx in range(ah.num_inference_timesteps):
                t_cont = t_idx / float(ah.num_inference_timesteps)
                t_disc = int(t_cont * ah.num_timestep_buckets)
                ts = torch.full((B,), fill_value=t_disc, device=device)

                act_feat = ah.action_encoder(actions, ts, emb_id)
                if c.add_pos_embed:
                    pids = torch.arange(act_feat.shape[1], dtype=torch.long, device=device)
                    act_feat = act_feat + ah.position_embedding(pids).unsqueeze(0)
                sa_embs = torch.cat((st_feat, act_feat), dim=1)

                om_inputs = build_om_inputs(sa_embs, vl_emb, ts,
                                             om_inputs_static[0],
                                             om_inputs_static[1])

                torch.npu.synchronize()
                t0 = time.perf_counter()
                raw = om_runner.run(om_inputs)
                torch.npu.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                om_times.append(elapsed)

                m_out_np = parse_om_output(raw, (1, SEQ, 1024))
                if run == 0 and t_idx == 0:
                    om_ref_output = m_out_np.copy()

        arr_om = np.array(om_times)
        om_mean = np.mean(arr_om)
        print(f"  OM DiT step:     mean={om_mean:.1f}ms  p95={np.percentile(arr_om,95):.1f}ms  "
              f"min={np.min(arr_om):.1f}ms")

        if pytorch_mean:
            print(f"\n  Speedup (OM vs PyTorch): {pytorch_mean/om_mean:.2f}x")

    # ── 精度验证 ──
    if args.mode in ("both",) and om_ref_output is not None:
        # Run one PyTorch forward with same inputs for precision comparison
        actions = torch.randn((B, c.action_horizon, c.max_action_dim),
                              dtype=torch.float16, device=device)
        ts = torch.zeros(B, dtype=torch.long, device=device)
        act_feat = ah.action_encoder(actions, ts, emb_id)
        if c.add_pos_embed:
            pids = torch.arange(act_feat.shape[1], dtype=torch.long, device=device)
            act_feat = act_feat + ah.position_embedding(pids).unsqueeze(0)
        sa_embs = torch.cat((st_feat, act_feat), dim=1)

        torch.npu.synchronize()
        pt_out = dit(hidden_states=sa_embs, encoder_hidden_states=vl_emb,
                     timestep=ts, image_mask=img_mask,
                     backbone_attention_mask=bb_am)
        torch.npu.synchronize()
        pt_out_np = pt_out.detach().cpu().float().numpy()
        om_out_fp32 = om_ref_output.astype(np.float32)

        max_diff = np.abs(pt_out_np - om_out_fp32).max()
        mean_diff = np.abs(pt_out_np - om_out_fp32).mean()
        print(f"\n[Precision] max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    F.scaled_dot_product_attention = _orig_sdpa


if __name__ == "__main__":
    main()
