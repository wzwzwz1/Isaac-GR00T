#!/usr/bin/env python3
"""
昇腾 NPU Attention 实现对比 Benchmark.

对比以下 attention 实现的延迟和显存:
1. 手动实现 (当前方案): matmul + softmax + matmul
2. torch.nn.functional.scaled_dot_product_attention (如果有 NPU 支持)
3. torch_npu.npu_fused_attention (如果可用)
4. 分块 attention (chunked, 减少显存峰值)

用于评估用 Ascend C 自定义算子的潜在收益。

用法:
    python scripts/profiling/attention_bench.py --batch 1 --seq_len 66 --ctx_len 256 --hidden 1536
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _is_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def _sync():
    if _is_npu():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn: Callable, warmup: int = 5, repeat: int = 20) -> Dict[str, float]:
    """运行并计时一个函数，返回统计."""
    # 预热
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


# ─── Attention 实现 ─────────────────────────────────────────────────

def manual_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """当前方案: 标准 attention 的手工实现."""
    d_k = Q.shape[-1]
    scale = 1.0 / (d_k ** 0.5)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        attn_scores = attn_scores + mask
    attn_weights = F.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, V)


def sdpa_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """使用 PyTorch 内置 SDPA (可能在 NPU 上有问题)."""
    try:
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    except Exception as e:
        # fallback to manual
        return manual_attention(Q, K, V, mask)


def npu_fused_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """使用 torch_npu 融合 attention (如果可用)."""
    try:
        import torch_npu  # noqa: F811
        # npu_fused_attention 接口可能与标准不同
        # 尝试直接调用
        return torch_npu.npu_fused_attention(Q, K, V)
    except (ImportError, AttributeError, RuntimeError):
        return manual_attention(Q, K, V)


def chunked_manual_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: Optional[torch.Tensor] = None, chunk_size: int = 32,
) -> torch.Tensor:
    """分块 attention: 减少显存峰值, 适合 NPU 小 L1 buffer."""
    B, num_heads, seq_len, d_k = Q.shape
    _, _, ctx_len, _ = K.shape
    scale = 1.0 / (d_k ** 0.5)

    # 分块处理 seq_len 维度 (Q 的维度)
    output = torch.zeros_like(Q)
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        Q_chunk = Q[:, :, i:end_i, :]  # [B, H, chunk, d]

        # Q*K^T per chunk
        attn_scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale  # [B, H, chunk, ctx]

        if mask is not None:
            mask_chunk = mask[:, :, i:end_i, :] if mask.dim() == 4 else mask
            attn_scores = attn_scores + mask_chunk

        attn_weights = F.softmax(attn_scores, dim=-1)
        output[:, :, i:end_i, :] = torch.matmul(attn_weights, V)

    return output


# ─── Benchmark ──────────────────────────────────────────────────────

def run_benchmark(
    batch: int = 1,
    num_heads: int = 32,
    seq_len: int = 66,
    ctx_len: int = 256,
    d_k: int = 48,
    dtype: torch.dtype = torch.float16,
    device: str = "npu",
    npu_timeline: bool = False,
):
    """运行完整的 attention benchmark."""

    if device == "npu" and not _is_npu():
        print("NPU 不可用，改为 CUDA/CPU")
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)

    # 构造 Q, K, V
    Q = torch.randn(batch, num_heads, seq_len, d_k, dtype=dtype, device=dev)
    K = torch.randn(batch, num_heads, ctx_len, d_k, dtype=dtype, device=dev)
    V = torch.randn(batch, num_heads, ctx_len, d_k, dtype=dtype, device=dev)

    print(f"\n{'='*60}")
    print(f"  Attention Benchmark — {device.upper()}")
    print(f"  形状: Q={list(Q.shape)} K={list(K.shape)} V={list(V.shape)}")
    print(f"  dtype={dtype}")
    print(f"{'='*60}\n")

    implementations = {
        "manual (matmul+softmax+matmul)": lambda: manual_attention(Q, K, V),
        "torch SDPA": lambda: sdpa_attention(Q, K, V),
        "chunked manual (chunk=32)": lambda: chunked_manual_attention(Q, K, V, chunk_size=32),
    }

    if device == "npu":
        implementations["npu_fused_attention"] = lambda: npu_fused_attention(Q, K, V)

    results = {}
    for name, fn in implementations.items():
        try:
            stats = _time_ms(fn, warmup=5, repeat=20)
            results[name] = stats
            print(f"  {name:<35s}  {stats['mean']:7.2f}ms  "
                  f"(p95={stats['p95']:6.2f}ms, min={stats['min']:5.2f}ms)")
        except Exception as e:
            print(f"  {name:<35s}  ERROR: {e}")

    # 对比 manual 的加速比
    if "manual (matmul+softmax+matmul)" in results:
        baseline = results["manual (matmul+softmax+matmul)"]["mean"]
        print(f"\n  --- 相对加速比 (vs manual) ---")
        for name, stats in results.items():
            if name != "manual (matmul+softmax+matmul)":
                speedup = baseline / stats["mean"]
                print(f"  {name:<35s}  {speedup:.2f}x")

    # 估算了整个 DiT 的 attention 总开销
    print(f"\n  --- 估算 DiT 总 Attention 开销 (32层 x 4步) ---")
    if "manual (matmul+softmax+matmul)" in results:
        per_attn = results["manual (matmul+softmax+matmul)"]["mean"]
        # 每层 DiT: cross-attn layers have 1 attention, self-attn layers have 1 attention
        # 16 cross-attn + 16 self-attn per step, 4 steps
        total_attn_ms = per_attn * 16 * 4  # cross-attention only (self-attn is smaller)
        print(f"  单次 Cross-Attn: {per_attn:.2f}ms")
        print(f"  16层 Cross-Attn x 4步: {total_attn_ms:.0f}ms")
        print(f"  +16层 Self-Attn x 4步 (估计): {total_attn_ms * 0.3:.0f}ms")
        print(f"  DiT Attention 总计: {total_attn_ms * 1.3:.0f}ms")
        print(f"  (占总延迟 ~{total_attn_ms * 1.3 / 304 * 100:.0f}%, 基于 304ms 总延迟)")

    return results


# ─── 显存分析 ────────────────────────────────────────────────────────

def memory_analysis(
    batch: int = 1,
    num_heads: int = 32,
    seq_len: int = 66,
    ctx_len: int = 256,
    d_k: int = 48,
):
    """分析 attention 中间张量的显存占用."""
    print(f"\n{'='*60}")
    print(f"  Attention 显存分析")
    print(f"{'='*60}\n")

    fp16_bytes = 2

    # Q, K, V
    q_size = batch * num_heads * seq_len * d_k * fp16_bytes
    k_size = batch * num_heads * ctx_len * d_k * fp16_bytes
    v_size = batch * num_heads * ctx_len * d_k * fp16_bytes

    # 中间 attention scores: [B, H, seq, ctx] in float32 (softmax 需要)
    scores_size = batch * num_heads * seq_len * ctx_len * 4  # FP32

    # 输出
    out_size = batch * num_heads * seq_len * d_k * fp16_bytes

    # 计算各方案的总显存
    # 手工方案: QKV + scores(FP32) + output ≈ scores 是主要开销
    manual_peak = q_size + k_size + v_size + scores_size + out_size

    # 分块方案 (chunk=32): Q_chunk + K + V + scores_chunk(FP32) + out_chunk
    chunk = 32
    chunk_scores = batch * num_heads * chunk * ctx_len * 4
    chunked_peak = chunk * num_heads * d_k * fp16_bytes + k_size + v_size + chunk_scores + chunk * num_heads * d_k * fp16_bytes

    def fmt(b):
        if b > 1e9:
            return f"{b/1e9:.2f} GB"
        elif b > 1e6:
            return f"{b/1e6:.2f} MB"
        else:
            return f"{b/1e3:.2f} KB"

    print(f"  Q 显存:          {fmt(q_size)}")
    print(f"  K 显存:          {fmt(k_size)}")
    print(f"  V 显存:          {fmt(v_size)}")
    print(f"  Attention Scores: {fmt(scores_size)}  ← 主要瓶颈! (seq={seq_len} x ctx={ctx_len} x FP32)")
    print(f"  输出:             {fmt(out_size)}")
    print(f"")
    print(f"  手工方案峰值:     {fmt(manual_peak)}")
    print(f"  分块方案峰值:     {fmt(chunked_peak)} (chunk_size={chunk})")
    print(f"  峰值节省:         {fmt(manual_peak - chunked_peak)} ({(1 - chunked_peak/manual_peak)*100:.0f}%)")


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NPU Attention 实现 Benchmark"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=66,
                        help="action head 序列长度 (1 state + 16 action + padding)")
    parser.add_argument("--ctx_len", type=int, default=256,
                        help="backbone context 长度")
    parser.add_argument("--d_k", type=int, default=48,
                        help="attention head dimension")
    parser.add_argument("--device", type=str, default="npu",
                        choices=["npu", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--memory", action="store_true",
                        help="仅运行显存分析")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (自动持久化结果)")
    args = parser.parse_args()

    from scripts.profiling._persist import ResultCollector
    rc = ResultCollector("attention_bench", args.output_dir)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    results = {}
    if args.memory:
        memory_analysis(args.batch, args.num_heads, args.seq_len, args.ctx_len, args.d_k)
    else:
        results = run_benchmark(args.batch, args.num_heads, args.seq_len,
                                args.ctx_len, args.d_k, dtype, args.device)
        memory_analysis(args.batch, args.num_heads, args.seq_len, args.ctx_len, args.d_k)

    # Persist
    bench_data = {"config": {"batch": args.batch, "num_heads": args.num_heads,
                              "seq_len": args.seq_len, "ctx_len": args.ctx_len,
                              "d_k": args.d_k, "device": args.device, "dtype": args.dtype},
                   "results": results}
    rc.add_json("attention_bench", bench_data)
    rc.save_all()


if __name__ == "__main__":
    main()
