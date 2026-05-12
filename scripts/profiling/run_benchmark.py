#!/usr/bin/env python3
"""
GR00T-N1.6 昇腾NPU 一键Profiling脚本.
运行所有分析工具并持久化结果到 profiling_results/ 目录.

输出:
  profiling_results/
  ├── 00_environment.json        # 环境信息
  ├── 01_stage_breakdown.json    # 管线阶段耗时
  ├── 01_stage_breakdown.txt     # 管线阶段耗时(可读)
  ├── 02_dit_per_block.json      # DiT 32层逐Block耗时
  ├── 02_dit_per_block.txt       # DiT 32层逐Block耗时(可读)
  ├── 03_operator_breakdown.json # 按Block类型×算子类别
  ├── 03_operator_breakdown.txt  # 按Block类型×算子类别(可读)
  ├── 04_global_summary.json     # 全局算子汇总
  └── 04_global_summary.txt      # 全局算子汇总(可读)

用法:
    python3 scripts/profiling/run_benchmark.py \
        --model_path /home/wangzhe/models/GR00T-N1.6-3B-FP16 \
        --num_warmup 5 --num_runs 30 \
        --output_dir profiling_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.profiling._persist import ResultCollector, save_json, save_text


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _sync():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_table(headers: list, rows: list, col_widths: list = None) -> str:
    """ Generate a formatted ASCII table string. """
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
                      for i in range(len(headers))]
    lines = []
    # header
    hdr = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    # rows
    for row in rows:
        lines.append("".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths)))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 主 Profiling 逻辑
# ═══════════════════════════════════════════════════════════════════════

def collect_environment(output_dir: str) -> dict:
    """收集环境信息."""
    info = {
        "timestamp": now_str(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    try:
        import torch_npu
        info["torch_npu_version"] = torch_npu.__version__
        info["npu_available"] = torch.npu.is_available()
        info["npu_device_count"] = torch.npu.device_count()
        info["npu_device_name"] = torch.npu.get_device_name(0)
    except Exception:
        info["npu_available"] = False

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    save_json(info, os.path.join(output_dir, "00_environment.json"))
    return info


def run_stage_breakdown(model, bb_out, act_in, output_dir: str,
                        warmup: int, runs: int):
    """
    Step 1: 管线阶段耗时分解.
    测量 encode_features + 每一步denoising 的时间.
    """
    ah = model.action_head
    c = model.config
    B = 1
    device = next(model.parameters()).device
    steps = defaultdict(list)

    print(f"  [1/4] 管线阶段Profiling ({warmup}+{runs} runs)...")

    for run_idx in range(warmup + runs):
        torch.npu.synchronize()
        t_start = time.perf_counter()

        # Encode
        feats = ah._encode_features(bb_out, act_in)
        torch.npu.synchronize()
        t_encode = time.perf_counter()

        # Denoising loop
        vl_emb = feats.backbone_features
        st_feat = feats.state_features
        emb_id = act_in.embodiment_id

        actions = torch.randn((B, c.action_horizon, c.max_action_dim),
                              dtype=torch.float16, device=device)
        dt_val = 1.0 / ah.num_inference_timesteps

        denoise_times = []
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
            t_step = time.perf_counter()

            if c.use_alternate_vl_dit:
                m_out = ah.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_emb,
                    timestep=ts, image_mask=bb_out.image_mask,
                    backbone_attention_mask=bb_out.backbone_attention_mask)
            else:
                m_out = ah.model(hidden_states=sa_embs, encoder_hidden_states=vl_emb,
                                timestep=ts)

            torch.npu.synchronize()
            denoise_times.append((time.perf_counter() - t_step) * 1000)

            pred_v = ah.action_decoder(m_out, emb_id)[:, -c.action_horizon:]
            actions = actions + dt_val * pred_v

        torch.npu.synchronize()
        t_total = (time.perf_counter() - t_start) * 1000

        if run_idx >= warmup:
            steps['encode'].append((t_encode - t_start) * 1000)
            for i, d in enumerate(denoise_times):
                steps[f'denoise_step_{i}'].append(d)
            steps['total'].append(t_total)

    # ── 生成报告文本 ──
    total_mean = np.mean(steps['total'])
    lines = []
    lines.append(f"GR00T-N1.6 管线阶段耗时分解")
    lines.append(f"时间: {now_str()}")
    lines.append(f"action_horizon={c.action_horizon} action_dim={c.max_action_dim}")
    lines.append(f"warmup={warmup} runs={runs}")
    lines.append("")
    lines.append(f"{'Stage':<22s} {'Mean(ms)':>8s} {'Min(ms)':>8s} {'Max(ms)':>8s} "
                 f"{'P95(ms)':>8s} {'%':>6s}")
    lines.append(f"{'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    stage_order = ['encode'] + [f'denoise_step_{i}' for i in range(ah.num_inference_timesteps)] + ['total']
    json_data = {"timestamp": now_str(), "config": {
        "action_horizon": c.action_horizon, "action_dim": c.max_action_dim,
        "num_inference_timesteps": ah.num_inference_timesteps,
        "warmup": warmup, "runs": runs,
    }, "stages": {}}

    for key in stage_order:
        arr = np.array(steps[key])
        pct = np.mean(arr) / total_mean * 100 if total_mean > 0 else 0
        lines.append(f"  {key:<20s} {np.mean(arr):7.1f}  {np.min(arr):7.1f}  "
                     f"{np.max(arr):7.1f}  {np.percentile(arr, 95):7.1f}  {pct:5.1f}%")
        json_data["stages"][key] = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }

    denoise_total = sum(np.mean(steps[f'denoise_step_{i}'])
                        for i in range(ah.num_inference_timesteps))
    lines.append(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    lines.append(f"  {'Denoise 4步合计':<20s} {denoise_total:7.1f}ms  "
                 f"({denoise_total/total_mean*100:.0f}%)")

    text = "\n".join(lines)
    save_text(text, os.path.join(output_dir, "01_stage_breakdown.txt"))
    save_json(json_data, os.path.join(output_dir, "01_stage_breakdown.json"))

    print(f"    Action Head Total: {total_mean:.1f}ms | Denoise: {denoise_total:.1f}ms "
          f"({denoise_total/total_mean*100:.0f}%)")
    return steps


def run_per_block_breakdown(model, bb_out, act_in, output_dir: str,
                            warmup: int, runs: int):
    """
    Step 2: DiT 32层逐Block耗时.
    """
    ah = model.action_head
    dit = ah.model
    c = model.config
    B = 1
    device = next(model.parameters()).device

    print(f"  [2/4] DiT 逐Block Profiling ({warmup}+{runs} runs)...")

    # 安装钩子
    block_times = defaultdict(list)  # blk_idx -> [durations]
    hooks = []

    for blk_idx, block in enumerate(dit.transformer_blocks):
        def make_hooks(b_idx):
            s = [None]
            def pre_h(mod, inp): torch.npu.synchronize(); s[0] = time.perf_counter()
            def fwd_h(mod, inp, out):
                torch.npu.synchronize()
                block_times[b_idx].append((time.perf_counter() - s[0]) * 1000)
            return pre_h, fwd_h
        pre, fwd = make_hooks(blk_idx)
        hooks.append(block.register_forward_pre_hook(pre))
        hooks.append(block.register_forward_hook(fwd))

    # 运行
    feats = ah._encode_features(bb_out, act_in)
    vl_emb = feats.backbone_features
    st_feat = feats.state_features
    emb_id = act_in.embodiment_id

    for run_idx in range(warmup + runs):
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

            img_am = bb_out.image_mask & bb_out.backbone_attention_mask
            non_img_am = (~bb_out.image_mask) & bb_out.backbone_attention_mask
            temb = dit.timestep_encoder(ts)
            hidden = sa_embs.contiguous()
            enc_hidden = vl_emb.contiguous()

            for blk_idx, block in enumerate(dit.transformer_blocks):
                if blk_idx % 2 == 1:
                    hidden = block(hidden, attention_mask=None,
                                   encoder_hidden_states=None,
                                   encoder_attention_mask=None, temb=temb)
                else:
                    curr = (non_img_am if blk_idx % (2 * dit.attend_text_every_n_blocks) == 0
                            else img_am)
                    hidden = block(hidden, attention_mask=None,
                                   encoder_hidden_states=enc_hidden,
                                   encoder_attention_mask=curr, temb=temb)

            shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
            hidden = dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            m_out = dit.proj_out_2(hidden)
            pred_v = ah.action_decoder(m_out, emb_id)[:, -c.action_horizon:]
            actions = actions + dt_val * pred_v

    # 卸载钩子
    for h in hooks:
        h.remove()

    # ── 生成报告 ──
    lines = []
    lines.append(f"GR00T-N1.6 DiT 32层逐Block耗时")
    lines.append(f"时间: {now_str()}  |  action_horizon={c.action_horizon} action_dim={c.max_action_dim}")
    lines.append(f"数据为单次block forward耗时(ms), 跨{warmup}+{runs}次运行 × 4步取平均")
    lines.append("")
    lines.append(f"{'Block':<6s} {'Type':<24s} {'Mean(ms)':>9s} {'Std(ms)':>8s} "
                 f"{'Min(ms)':>8s} {'Max(ms)':>8s} {'P95(ms)':>8s}")
    lines.append(f"{'─'*6} {'─'*24} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    json_data = {"timestamp": now_str(), "config": {
        "action_horizon": c.action_horizon, "action_dim": c.max_action_dim,
        "warmup": warmup, "runs": runs,
    }, "blocks": {}}

    cross_total = 0.0
    self_total = 0.0

    for blk_idx in sorted(block_times.keys()):
        arr = np.array(block_times[blk_idx])
        btype = ("cross-attn (text)" if blk_idx % 4 == 0 else "cross-attn (image)"
                 if blk_idx % 2 == 0 else "self-attn")
        lines.append(f"  L{blk_idx:<4d} {btype:<24s} {np.mean(arr):8.3f}  "
                     f"{np.std(arr):7.3f}  {np.min(arr):7.3f}  "
                     f"{np.max(arr):7.3f}  {np.percentile(arr, 95):7.3f}")
        json_data["blocks"][str(blk_idx)] = {
            "type": btype,
            "mean": round(float(np.mean(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
        }
        if blk_idx % 2 == 0:
            cross_total += np.mean(arr)
        else:
            self_total += np.mean(arr)

    lines.append(f"  {'─'*6} {'─'*24} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    lines.append(f"  Cross-Attn (16层) 合计: {cross_total:.2f}ms")
    lines.append(f"  Self-Attn  (16层) 合计: {self_total:.2f}ms")
    lines.append(f"  32层总计: {cross_total + self_total:.2f}ms")

    text = "\n".join(lines)
    save_text(text, os.path.join(output_dir, "02_dit_per_block.txt"))
    save_json(json_data, os.path.join(output_dir, "02_dit_per_block.json"))

    print(f"    Cross-Attn: {cross_total:.1f}ms  |  Self-Attn: {self_total:.1f}ms  "
          f"|  Total: {cross_total+self_total:.1f}ms")


def run_operator_breakdown(model, bb_out, act_in, output_dir: str,
                           warmup: int, runs: int):
    """
    Step 3 & 4: 按Block类型 × 算子类别 + 全局汇总.
    """
    ah = model.action_head
    dit = ah.model
    c = model.config
    B = 1
    device = next(model.parameters()).device

    print(f"  [3/4] 逐算子分类 Profiling ({warmup}+{runs} runs)...")

    # 安装叶子模块钩子
    op_times = defaultdict(list)  # (block_idx, category) -> [durations]

    for blk_idx, block in enumerate(dit.transformer_blocks):
        for mod_name, module in block.named_modules():
            if mod_name == '':
                continue
            cat = _classify_op(mod_name, module)
            b = blk_idx

            def make_hooks(m_name, m_cat, m_blk):
                s = [None]
                def pre_h(mod, inp): torch.npu.synchronize(); s[0] = time.perf_counter()
                def fwd_h(mod, inp, out):
                    torch.npu.synchronize()
                    op_times[(m_blk, m_cat)].append((time.perf_counter() - s[0]) * 1000)
                return pre_h, fwd_h
            pre, fwd = make_hooks(mod_name, cat, b)
            module.register_forward_pre_hook(pre)
            module.register_forward_hook(fwd)

    # 运行
    feats = ah._encode_features(bb_out, act_in)
    vl_emb = feats.backbone_features
    st_feat = feats.state_features
    emb_id = act_in.embodiment_id

    for run_idx in range(warmup + runs):
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

            img_am = bb_out.image_mask & bb_out.backbone_attention_mask
            non_img_am = (~bb_out.image_mask) & bb_out.backbone_attention_mask
            temb = dit.timestep_encoder(ts)
            hidden = sa_embs.contiguous()
            enc_hidden = vl_emb.contiguous()

            for blk_idx, block in enumerate(dit.transformer_blocks):
                if blk_idx % 2 == 1:
                    hidden = block(hidden, attention_mask=None,
                                   encoder_hidden_states=None,
                                   encoder_attention_mask=None, temb=temb)
                else:
                    curr = (non_img_am if blk_idx % (2 * dit.attend_text_every_n_blocks) == 0
                            else img_am)
                    hidden = block(hidden, attention_mask=None,
                                   encoder_hidden_states=enc_hidden,
                                   encoder_attention_mask=curr, temb=temb)

            shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
            hidden = dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            _ = dit.proj_out_2(hidden)

    # ── 按Block类型聚合 ──
    cross_blks = [i for i in range(32) if i % 2 == 0]
    self_blks = [i for i in range(32) if i % 2 == 1]

    lines = []
    lines.append(f"GR00T-N1.6 DiT 逐算子类别耗时分解")
    lines.append(f"时间: {now_str()}  |  action_horizon={c.action_horizon} action_dim={c.max_action_dim}")
    lines.append(f"warmup={warmup} runs={runs} | 数据为 per-step 均值(ms)")
    lines.append("")

    json_data = {"timestamp": now_str(), "config": {
        "action_horizon": c.action_horizon, "action_dim": c.max_action_dim,
        "warmup": warmup, "runs": runs,
    }, "cross_attn": {}, "self_attn": {}, "global_summary": {}}

    for label, blk_list in [("Cross-Attn (16 blocks)", cross_blks),
                             ("Self-Attn  (16 blocks)", self_blks)]:
        cat_sums = defaultdict(float)
        for blk in blk_list:
            for (b, cat), durs in op_times.items():
                if b == blk:
                    cat_sums[cat] += np.mean(durs)

        total = sum(cat_sums.values())
        lines.append(f"  [{label}]  per-step total: {total:.1f}ms")
        lines.append(f"  {'Category':<25s} {'Ms/Step':>8s} {'%':>7s}  {'说明':<35s}")
        lines.append(f"  {'─'*25} {'─'*8} {'─'*7}  {'─'*35}")

        section_key = "cross_attn" if "Cross" in label else "self_attn"

        for cat, ms in sorted(cat_sums.items(), key=lambda x: x[1], reverse=True):
            pct = ms / total * 100 if total > 0 else 0
            bar = '█' * max(1, int(pct / 2))
            desc = _cat_description(cat)
            lines.append(f"  {cat:<25s} {ms:7.2f}  {pct:5.1f}%  {bar} {desc}")
            json_data[section_key][cat] = {
                "ms_per_step": round(float(ms), 2),
                "pct": round(float(pct), 1),
            }
        lines.append("")

    # ── 全局汇总 ──
    print(f"  [4/4] 全局算子汇总...")
    global_cat = defaultdict(float)
    for (b, cat), durs in op_times.items():
        if b >= 0:
            global_cat[cat] += np.sum(durs) / runs / 4  # per-step

    total = sum(global_cat.values())
    lines.append(f"  全局算子类别汇总 (All 32 blocks):")
    lines.append(f"  Per-Step DiT Total (hook-instrumented): {total:.1f}ms")
    lines.append(f"  (注: 钩子同步开销导致绝对值 > 真实耗时, 相对比例是准确的)")
    lines.append(f"  (真实DiT单步耗时请参考 01_stage_breakdown)")
    lines.append("")
    lines.append(f"  {'Category':<25s} {'PerStep(ms)':>11s} {'%':>7s}  {'累计%':>7s}")
    lines.append(f"  {'─'*25} {'─'*11} {'─'*7}  {'─'*7}")

    cum = 0.0
    for cat, ms in sorted(global_cat.items(), key=lambda x: x[1], reverse=True):
        pct = ms / total * 100 if total > 0 else 0
        cum += pct
        bar = '█' * max(1, int(pct / 3))
        lines.append(f"  {cat:<25s} {ms:10.2f}  {pct:5.1f}%  {cum:6.1f}%  {bar}")
        json_data["global_summary"][cat] = {
            "ms_per_step": round(float(ms), 2),
            "pct": round(float(pct), 1),
            "cum_pct": round(float(cum), 1),
        }

    # ── 推算真实耗时 ──
    lines.append("")
    lines.append(f"  {'─'*60}")
    lines.append(f"  真实耗时推算 (去钩子开销)")
    lines.append(f"  {'─'*60}")

    # 获取无钩子的真实单步耗时
    real_per_step = None
    stage_json_path = os.path.join(output_dir, "01_stage_breakdown.json")
    if os.path.exists(stage_json_path):
        with open(stage_json_path) as f:
            stage_data = json.load(f)
        denoise_means = [v["mean"] for k, v in stage_data["stages"].items()
                         if k.startswith("denoise_step")]
        if denoise_means:
            real_per_step = np.mean(denoise_means)

    if real_per_step:
        lines.append(f"  无钩子DiT单步耗时: {real_per_step:.1f}ms")
        lines.append(f"  钩子DiT单步耗时:   {total:.1f}ms")
        lines.append(f"  钩子开销:           {total - real_per_step:.1f}ms")
        lines.append("")
        lines.append(f"  {'Category':<25s} {'推算真实(ms)':>12s} {'%':>6s}")
        lines.append(f"  {'─'*25} {'─'*12} {'─'*6}")

        scale = real_per_step / total if total > 0 else 1.0
        for cat, ms in sorted(global_cat.items(), key=lambda x: x[1], reverse=True)[:10]:
            real_ms = ms * scale
            pct = real_ms / real_per_step * 100
            lines.append(f"  {cat:<25s} {real_ms:11.2f}  {pct:5.1f}%")

    text = "\n".join(lines)
    save_text(text, os.path.join(output_dir, "03_operator_breakdown.txt"))
    save_json(json_data, os.path.join(output_dir, "03_operator_breakdown.json"))


def _classify_op(mod_name: str, module) -> str:
    """根据模块名和类型分类算子."""
    n = mod_name.lower()
    if isinstance(module, torch.nn.Linear):
        if any(k in n for k in ['to_q', 'to_k', 'to_v', 'to_qkv']):
            return 'QKV Linear'
        if 'to_out' in n:
            return 'Out Projection'
        if any(k in n for k in ['gate_proj', 'up_proj', 'linear_1', 'linear_3', 'ff_in']):
            return 'MLP up/gate'
        if any(k in n for k in ['down_proj', 'linear_2']):
            return 'MLP down'
        if 'linear' in n or 'proj' in n:
            return 'MLP Linear'
        return 'Linear (other)'
    if isinstance(module, torch.nn.LayerNorm):
        return 'LayerNorm'
    if 'AdaLayerNorm' in type(module).__name__:
        return 'AdaNorm'
    if isinstance(module, torch.nn.Dropout):
        return 'Dropout'
    if isinstance(module, torch.nn.Embedding):
        return 'Position Embed'
    return 'Other'


def _cat_description(cat: str) -> str:
    return {
        'AdaNorm': 'AdaLayerNorm(SiLU+Linear投影+LayerNorm+scale/shift)',
        'LayerNorm': 'nn.LayerNorm',
        'QKV Linear': 'to_q / to_k / to_v 投影层',
        'Out Projection': 'to_out 输出线性层',
        'MLP up/gate': 'FFN gate+up 投影 (SwiGLU前半)',
        'MLP down': 'FFN down 投影 (SwiGLU后半)',
        'MLP Linear': 'FFN 内未细分的 Linear',
        'Linear (other)': '其他未分类 Linear',
        'Dropout': 'Dropout 层',
        'Position Embed': '位置编码 Embedding',
        'Other': '残差加法 + Attention函数调用(QK/Softmax/AV)等非模块操作',
    }.get(cat, '')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GR00T-N1.6 NPU 一键Profiling")
    parser.add_argument("--model_path", type=str,
                        default="/home/wangzhe/models/GR00T-N1.6-3B-FP16")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="profiling_results")
    parser.add_argument("--skip_steps", type=str, default="",
                        help="跳过步骤: 1=stage, 2=block, 3=operator")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  GR00T-N1.6 NPU Profiling Suite")
    print(f"  输出目录: {output_dir.absolute()}")
    print(f"  warmup={args.num_warmup}  runs={args.num_runs}")
    print(f"{'='*70}\n")

    # ── 环境信息 ──
    env = collect_environment(str(output_dir))
    print(f"  设备: {env.get('npu_device_name', 'N/A')}  "
          f"torch={env.get('torch_version', 'N/A')}\n")

    # ── 加载模型 ──
    print("  加载模型...")
    import gr00t.model  # noqa
    from transformers import AutoModel, BatchFeature

    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    model.to(device='npu:0', dtype=torch.float16)

    # SDPA monkey-patch
    _orig_sdpa = F.scaled_dot_product_attention
    def manual_attn(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False):
        d_k = query.shape[-1]
        s = scale if scale is not None else 1.0 / (d_k ** 0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * s
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, value)
    F.scaled_dot_product_attention = manual_attn

    c = model.config
    device = next(model.parameters()).device

    # 构造输入
    B, CTX = 1, 256
    state = torch.randn(B, 1, c.max_state_dim, dtype=torch.float16, device=device)
    bb_feat = torch.randn(B, CTX, c.backbone_embedding_dim, dtype=torch.float16, device=device)
    bb_am = torch.ones(B, CTX, dtype=torch.bool, device=device)
    img_mask = torch.zeros(B, CTX, dtype=torch.bool, device=device)
    img_mask[:, :CTX // 2] = True

    bb_out = BatchFeature({
        'backbone_features': bb_feat,
        'backbone_attention_mask': bb_am,
        'image_mask': img_mask,
    })
    act_in = BatchFeature({
        'state': state,
        'embodiment_id': torch.zeros(B, dtype=torch.long, device=device),
    })

    skip = set(args.skip_steps.split(",")) if args.skip_steps else set()

    # Step 1: 管线阶段
    if "1" not in skip:
        run_stage_breakdown(model, bb_out, act_in, str(output_dir),
                            args.num_warmup, args.num_runs)
    else:
        print("  [1/4] 跳过")

    # Step 2: 逐Block
    if "2" not in skip:
        run_per_block_breakdown(model, bb_out, act_in, str(output_dir),
                                args.num_warmup, args.num_runs)
    else:
        print("  [2/4] 跳过")

    # Step 3&4: 算子分类 + 全局汇总
    if "3" not in skip:
        run_operator_breakdown(model, bb_out, act_in, str(output_dir),
                               args.num_warmup, args.num_runs)
    else:
        print("  [3/4] 跳过")

    # 清理
    F.scaled_dot_product_attention = _orig_sdpa

    print(f"\n{'='*70}")
    print(f"  Profiling 完成! 结果保存在: {output_dir.absolute()}")
    print(f"{'='*70}")
    print(f"\n  {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name}  ({size:,} bytes)")
    print()


if __name__ == "__main__":
    main()
