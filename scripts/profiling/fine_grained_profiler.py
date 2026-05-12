#!/usr/bin/env python3
"""
GR00T DiT 细粒度逐算子 Profiler.

按层、按算子类别精确分解 DiT 推理耗时:
  - QKV Linear          (Q/K/V 投影层)
  - Attention QK        (Q·K^T)
  - Attention Softmax
  - Attention AV        (AttnWeights·V)
  - Output Projection   (to_out)
  - MLP (gate+up+down)  (FeedForward 三线性层)
  - LayerNorm / AdaNorm
  - Position Embedding
  - Timestep Encoding
  - Output Head         (norm_out + proj_out)
  - Residual Add        (残差加法)
  - Cast / dtype convert
  - Transpose / Reshape / Contiguous
  - Dropout

输出三层树形结构:
  DiT Step N
    ├─ Input Prep (pos_embed + concat)
    ├─ Timestep Encoder
    ├─ Block 0  (cross_attn, text)
    │   ├─ AdaLayerNorm
    │   ├─ QKV Linear        [to_q, to_k, to_k(context)]
    │   ├─ Attention QK
    │   ├─ Attention Softmax
    │   ├─ Attention AV
    │   ├─ Output Projection
    │   ├─ Residual Add ×2
    │   ├─ LayerNorm
    │   ├─ MLP Linear         [gate, up, down]
    │   └─ Transpose/Reshape
    ├─ Block 1  (self_attn)
    │   └─ ...
    ├─ Block 2  (cross_attn, image)
    │   └─ ...
    ... (共 32 blocks)
    └─ Output Head

用法:
    python scripts/profiling/fine_grained_profiler.py \
        --model_path /root/models/GR00T-N1.6-3B-FP16 \
        --num_warmup 5 --num_runs 10
"""

from __future__ import annotations

import argparse
import json
import time
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════
# NPU/CUDA 兼容
# ═══════════════════════════════════════════════════════════════════════

def _is_npu() -> bool:
    try:
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except Exception:
        return False

def _sync():
    if _is_npu():
        torch.npu.synchronize()
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════
# 算子分类体系
# ═══════════════════════════════════════════════════════════════════════

class OpCategory:
    """所有可追踪算子类别的标签."""

    # ── 投影层 (来自 nn.Module hook) ──
    QKV_LINEAR       = "Linear (QKV)"           # to_q / to_k / to_v
    OUT_PROJ         = "Linear (Output)"         # to_out
    MLP_UP_GATE      = "Linear (MLP up+gate)"    # FFN gate + up projection (SwiGLU)
    MLP_DOWN         = "Linear (MLP down)"       # FFN down projection
    TIMESTEP_EMBED   = "Linear (Timestep embed)" # TimestepEmbedding 内部线性层
    OUT_HEAD_LINEAR  = "Linear (Output Head)"    # proj_out_1, proj_out_2
    STATE_ENCODER    = "Linear (State Encoder)"  # CategorySpecificMLP
    ACTION_ENCODER   = "Linear (Action Encoder)" # MultiEmbodimentActionEncoder
    ACTION_DECODER   = "Linear (Action Decoder)" # CategorySpecificMLP
    LLN_LINEAR       = "Linear (Other)"

    # ── 归一化层 ──
    ADA_NORM         = "AdaNorm"                 # AdaLayerNorm / AdaLayerNormZero
    LAYER_NORM       = "LayerNorm"               # nn.LayerNorm

    # ── Attention 计算 (来自函数 monkey-patch) ──
    ATTN_QK          = "Attention·QK"            # Q @ K^T
    ATTN_SOFTMAX     = "Attention·Softmax"       # F.softmax(scores)
    ATTN_AV          = "Attention·AV"            # attn @ V

    # ── 张量操作 (来自函数 monkey-patch) ──
    RESIDUAL_ADD     = "Residual Add"            # h = h + attn/ffn_out (通过 operator.add 追踪)
    TRANSPOSE        = "Transpose"               # .transpose(-2,-1), .permute()
    RESHAPE          = "Reshape"                 # .reshape(), .view()
    CONTIGUOUS       = "Contiguous"              # .contiguous()
    CAST             = "Cast"                    # .to(dtype=...)
    CONCAT           = "Concat"                  # torch.cat
    SOFTMAX          = "Softmax (other)"         # 非 attention 的 softmax
    SILU_GELU        = "SiLU / GELU"             # 激活函数

    # ── Embedding ──
    POS_EMBED        = "Position Embedding"      # nn.Embedding

    # ── Dropout ──
    DROPOUT          = "Dropout"

    # ── 其他 ──
    OTHER            = "Other"


# ═══════════════════════════════════════════════════════════════════════
# 细粒度 Profiler 核心
# ═══════════════════════════════════════════════════════════════════════

class FineGrainedProfiler:
    """
    递归 hook + 函数 monkey-patch 的细粒度 Profiler.

    工作流程:
      1. install() - 递归注册所有 nn.Module 的前后向 hook
      2. patch_torch_ops() - monkey-patch torch 关键函数
      3. run()  - 多次执行推理
      4. aggregate() - 汇总统计
      5. print_tree() - 输出树形延迟报告
    """

    def __init__(self, warmup: int = 5):
        self.warmup = warmup
        self._run = 0
        self._recording = False

        # 避免双重计数: module 内部调用的 matmul/softmax 等不应被函数 patch 重复记录
        # module_depth > 0 表示当前处于某个已 hook 模块的 forward 内部
        self._module_depth: int = 0

        # trace 数据: List[op_record]
        self._all_records: List[dict] = []

        self._current_block_idx: int = -1
        self._current_step_idx: int = -1

        self._event_stack: List[dict] = []

        self._module_hooks: List[tuple] = []
        self._patches: List[tuple] = []

        self._module_category: Dict[str, str] = {}

    # ─── 模块钩子 ────────────────────────────────────────────────

    def _classify_module(self, full_name: str, module: torch.nn.Module) -> str:
        """根据模块名和类型自动分类."""
        name_lower = full_name.lower()

        # QKV 投影
        if any(k in name_lower for k in ['to_q', 'to_k', 'to_v', 'to_qkv']):
            return OpCategory.QKV_LINEAR

        # Attention Output 投影
        if 'to_out' in name_lower and 'linear' in name_lower:
            return OpCategory.OUT_PROJ

        # MLP: FeedForward 内部
        if isinstance(module, torch.nn.Linear):
            if 'ff.' in name_lower or 'feedforward' in name_lower or name_lower.endswith('.ff'):
                # 区分: linear_1/linear_3 = gate+up, linear_2 = down
                if any(k in name_lower for k in ['linear_1', 'linear_3', 'net.0.proj']):
                    return OpCategory.MLP_UP_GATE
                elif any(k in name_lower for k in ['linear_2', 'net.2']):
                    return OpCategory.MLP_DOWN
                return OpCategory.MLP_UP_GATE

            # Timestep encoder 内部
            if 'timestep' in name_lower or 'time_proj' in name_lower:
                return OpCategory.TIMESTEP_EMBED

            # Output head
            if any(k in name_lower for k in ['proj_out', 'norm_out']):
                return OpCategory.OUT_HEAD_LINEAR

            # State / Action encoder/decoder
            if 'state_encoder' in name_lower or 'action_decoder' in name_lower:
                return OpCategory.ACTION_DECODER if 'decoder' in name_lower else OpCategory.STATE_ENCODER
            if 'action_encoder' in name_lower:
                return OpCategory.ACTION_ENCODER

            return OpCategory.LLN_LINEAR

        # 归一化层
        if isinstance(module, torch.nn.LayerNorm):
            if 'ada' in name_lower or 'norm1' in name_lower:
                return OpCategory.ADA_NORM
            return OpCategory.LAYER_NORM

        # Embedding
        if isinstance(module, torch.nn.Embedding):
            return OpCategory.POS_EMBED

        # Dropout
        if isinstance(module, torch.nn.Dropout):
            return OpCategory.DROPOUT

        # SiLU/GELU 激活 (作为独立模块时)
        if any(t in str(type(module)).lower() for t in ['silu', 'gelu', 'relu', 'swish']):
            return OpCategory.SILU_GELU

        return OpCategory.OTHER

    def install(self, model: torch.nn.Module, root_name: str = ""):
        """递归为 model 及其所有子模块安装 pre/post hook."""
        self._module_hooks = []

        for name, module in model.named_modules():
            # 跳过根模块本身
            if name == "":
                continue

            full_name = f"{root_name}.{name}" if root_name else name
            category = self._classify_module(full_name, module)
            self._module_category[full_name] = category

            profiler_ref = self

            def make_hooks(module_name, cat):
                pre_start = [None]

                def pre_hook(mod, inp):
                    profiler_ref._module_depth += 1
                    if not profiler_ref._recording:
                        return
                    _sync()
                    pre_start[0] = time.perf_counter()

                    block_idx = profiler_ref._current_block_idx
                    step_idx = profiler_ref._current_step_idx

                    # 尝试从 name 中提取 block index
                    for part in module_name.split('.'):
                        if 'transformer_blocks' in part or 'transformer_block' in part:
                            # try to parse the numeric index
                            sub_parts = module_name.split('.')
                            for i, sp in enumerate(sub_parts):
                                if sp in ('transformer_blocks', 'transformer_block'):
                                    try:
                                        block_idx = int(sub_parts[i + 1])
                                    except (IndexError, ValueError):
                                        pass
                                    break
                            break

                    profiler_ref._event_stack.append({
                        "name": module_name,
                        "category": cat,
                        "block_idx": block_idx,
                        "step_idx": step_idx,
                        "start": pre_start[0],
                    })

                def fwd_hook(mod, inp, out):
                    profiler_ref._module_depth -= 1
                    if not profiler_ref._recording:
                        return
                    _sync()
                    ts = time.perf_counter()

                    # 从栈中找到匹配的 pre 事件
                    evt = None
                    for i in range(len(profiler_ref._event_stack) - 1, -1, -1):
                        if profiler_ref._event_stack[i]["name"] == module_name:
                            evt = profiler_ref._event_stack.pop(i)
                            break

                    if evt is None:
                        return

                    duration = (ts - evt["start"]) * 1000.0
                    profiler_ref._all_records.append({
                        "name": module_name,
                        "category": cat,
                        "block_idx": evt["block_idx"],
                        "step_idx": evt["step_idx"],
                        "duration_ms": duration,
                        "type": "module",
                    })

                return pre_hook, fwd_hook

            pre, fwd = make_hooks(full_name, category)
            pre_handle = module.register_forward_pre_hook(pre)
            fwd_handle = module.register_forward_hook(fwd)
            self._module_hooks.append((module, pre_handle, fwd_handle))

    # ─── 函数 Monkey-Patch ───────────────────────────────────────

    def patch_torch_ops(self):
        """
        Monkey-patch torch 关键函数以追踪非 nn.Module 操作.
        包括: torch.matmul, F.softmax, tensor.transpose 等.
        """
        profiler_ref = self
        original_fns = {}

        def _make_patched(name, category, orig_fn):
            def patched(*args, **kwargs):
                if not profiler_ref._recording:
                    return orig_fn(*args, **kwargs)
                if profiler_ref._module_depth > 0:
                    return orig_fn(*args, **kwargs)
                _sync()
                t0 = time.perf_counter()
                result = orig_fn(*args, **kwargs)
                _sync()
                duration = (time.perf_counter() - t0) * 1000.0
                profiler_ref._all_records.append({
                    "name": name,
                    "category": category,
                    "block_idx": profiler_ref._current_block_idx,
                    "step_idx": profiler_ref._current_step_idx,
                    "duration_ms": duration,
                    "type": "function",
                })
                return result
            return patched

        # 需要 patch 的函数列表
        targets = []

        # matmul / bmm
        targets.append(("torch.matmul", OpCategory.ATTN_QK, torch, "matmul"))
        targets.append(("torch.bmm", OpCategory.ATTN_QK, torch, "bmm"))
        targets.append(("torch.baddbmm", OpCategory.ATTN_QK, torch, "baddbmm"))

        # softmax
        targets.append(("F.softmax", OpCategory.ATTN_SOFTMAX, F, "softmax"))

        # transpose / permute
        targets.append(("tensor.transpose", OpCategory.TRANSPOSE, torch.Tensor, "transpose"))
        targets.append(("tensor.permute", OpCategory.TRANSPOSE, torch.Tensor, "permute"))

        # reshape / view
        targets.append(("tensor.reshape", OpCategory.RESHAPE, torch.Tensor, "reshape"))
        targets.append(("tensor.view", OpCategory.RESHAPE, torch.Tensor, "view"))

        # contiguous
        targets.append(("tensor.contiguous", OpCategory.CONTIGUOUS, torch.Tensor, "contiguous"))

        # concat
        targets.append(("torch.cat", OpCategory.CONCAT, torch, "cat"))

        # cast
        # Note: tensor.to is complex, can be to(device), to(dtype), to(other_tensor)
        # Only wrap for dtype conversions
        original_to = torch.Tensor.to
        def patched_to(self, *args, **kwargs):
            if not profiler_ref._recording or profiler_ref._module_depth > 0:
                return original_to(self, *args, **kwargs)
            is_dtype_conv = (
                (len(args) > 0 and isinstance(args[0], torch.dtype)) or
                ('dtype' in kwargs)
            )
            if is_dtype_conv:
                _sync()
                t0 = time.perf_counter()
                result = original_to(self, *args, **kwargs)
                _sync()
                duration = (time.perf_counter() - t0) * 1000.0
                profiler_ref._all_records.append({
                    "name": "tensor.to(dtype)",
                    "category": OpCategory.CAST,
                    "block_idx": profiler_ref._current_block_idx,
                    "step_idx": profiler_ref._current_step_idx,
                    "duration_ms": duration,
                    "type": "function",
                })
                return result
            return original_to(self, *args, **kwargs)
        torch.Tensor.to = patched_to
        self._patches.append(("torch.Tensor.to", original_to, patched_to))

        for name, category, obj, attr in targets:
            original = getattr(obj, attr)
            patched = _make_patched(name, category, original)
            setattr(obj, attr, patched)
            self._patches.append((f"{obj.__name__}.{attr}" if hasattr(obj, '__name__') else name, original, patched))

        # softmax——区分 attention 内部和外部的 softmax
        # 在 transformer block 内部调用的是 attention softmax, 否则是 other softmax
        # 这里简化处理：F.softmax 统一标为 ATTN_SOFTMAX
        # (实际 DiT 中只有 attention 内部使用 softmax)

        # SiLU / GELU: F.silu, F.gelu
        for fn_name, cat in [("silu", OpCategory.SILU_GELU), ("gelu", OpCategory.SILU_GELU)]:
            if hasattr(F, fn_name):
                original = getattr(F, fn_name)
                patched = _make_patched(f"F.{fn_name}", cat, original)
                setattr(F, fn_name, patched)
                self._patches.append((f"F.{fn_name}", original, patched))

    # ─── 运行时控制 ──────────────────────────────────────────────

    def set_step(self, step_idx: int):
        self._current_step_idx = step_idx

    def set_block(self, block_idx: int):
        self._current_block_idx = block_idx

    def start_recording(self):
        self._recording = True

    def stop_recording(self):
        self._recording = False

    def begin_run(self):
        self._run += 1
        if self._run > self.warmup:
            self._recording = True
        self._current_block_idx = -1
        self._current_step_idx = -1

    def end_run(self):
        self._recording = False

    def uninstall(self):
        """恢复所有 monkey-patch 和 module hook."""
        for module, pre, fwd in self._module_hooks:
            pre.remove()
            fwd.remove()
        self._module_hooks.clear()

        for name, original_fn, _patched_fn in self._patches:
            if name == "torch.Tensor.to":
                torch.Tensor.to = original_fn
            else:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    obj_name, attr = parts
                    if obj_name == "F":
                        setattr(torch.nn.functional, attr, original_fn)
                    elif obj_name == "torch":
                        setattr(torch, attr, original_fn)
                    elif "tensor" in obj_name:
                        setattr(torch.Tensor, attr, original_fn)
        self._patches.clear()

    # ─── 汇总统计 ────────────────────────────────────────────────

    def aggregate(self) -> Dict[str, list]:
        """按 (step_idx, block_idx, category) 聚合所有记录."""

        # 聚合键: step_idx, block_idx, category
        groups: Dict[Tuple, List[float]] = defaultdict(list)

        for rec in self._all_records:
            step = rec["step_idx"]
            block = rec["block_idx"]
            cat = rec["category"]
            groups[(step, block, cat)].append(rec["duration_ms"])

        return {
            "groups": dict(groups),
            "total_records": len(self._all_records),
            "num_runs": self._run - self.warmup,
        }

    # ─── 树形输出 ────────────────────────────────────────────────

    def print_tree(self, total_latency_ms: float = None):
        """
        打印三层树形报告:
          DiT Step N (total ms)
            ├─ Block M (cross_attn/text) ...
            │   ├─ Category A: xx ms
            │   ├─ Category B: xx ms
            │   ...
            └─ ...
        """
        aggregated = self.aggregate()
        groups = aggregated["groups"]
        num_runs = max(aggregated["num_runs"], 1)

        if not groups:
            print("[FineGrainedProfiler] 无数据: 请确认已运行足够轮次 (warmup+num_runs)")
            return

        # ── 第一步: 按 step_idx, block_idx 组织 ──
        step_data: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        for (step, block, cat), durations in groups.items():
            avg_ms = np.mean(durations) if durations else 0
            step_data[step][block][cat] += avg_ms

        # 分离 step-specific 数据和非 step 数据 (block_idx==-1)
        global_cats: Dict[str, float] = defaultdict(float)
        for (step, block, cat), durations in groups.items():
            if block == -1 and step == -1:
                global_cats[cat] += np.mean(durations)

        # ── 输出 ──
        device_info = _get_device_name_safe()
        print(f"\n{'='*100}")
        print(f"  GR00T DiT 细粒度逐算子时延分解 ({device_info})")
        print(f"  Fine-Grained Per-Operator Latency Breakdown")
        print(f"  统计运行次数: {num_runs}")
        print(f"{'='*100}")

        for step_idx in sorted(step_data.keys()):
            blocks = step_data[step_idx]

            # 计算该 step 的总时间
            step_total = sum(
                sum(cat_times.values()) for cat_times in blocks.values()
            )
            if step_total == 0:
                continue

            print(f"\n  ╔══ DiT Step {step_idx}  [{step_total:.1f} ms] ═══════════════════════════════════════")

            for block_idx in sorted(blocks.keys()):
                block_cats = blocks[block_idx]
                block_total = sum(block_cats.values())

                # 确定 block 类型
                if block_idx % 2 == 1:
                    block_type = "self-attn"
                else:
                    # cross-attn: 区分 text / image
                    # 实际取决于 attend_text_every_n_blocks, 这里简化
                    if block_idx % 4 == 0:
                        block_type = "cross-attn (text tokens)"
                    else:
                        block_type = "cross-attn (image tokens)"

                print(f"  ┌── Block {block_idx:<3d} ({block_type:>25s})  [{block_total:.1f} ms] " +
                      f"{'─'*40}")

                # 按类别排序输出 (耗时大的优先)
                sorted_cats = sorted(block_cats.items(), key=lambda x: x[1], reverse=True)

                # 定义显示顺序组
                cat_groups = [
                    ("Head", ["AdaNorm", "LayerNorm"]),
                    ("QKV",  ["Linear (QKV)"]),
                    ("Attention", ["Attention·QK", "Attention·Softmax", "Attention·AV"]),
                    ("Output", ["Linear (Output)"]),
                    ("FFN",   ["Linear (MLP up+gate)", "Linear (MLP down)", "SiLU / GELU"]),
                    ("Tensor", ["Transpose", "Reshape", "Contiguous", "Dropout", "Residual Add", "Cast", "Concat"]),
                    ("Other", []),
                ]

                shown = set()
                for group_name, group_cats in cat_groups:
                    for cat, ms in sorted_cats:
                        if cat in shown:
                            continue
                        if group_name == "Other" or cat in group_cats:
                            shown.add(cat)
                            # 计算该类别占本 block 的百分比
                            pct = ms / block_total * 100 if block_total > 0 else 0
                            bar = _bar(pct)
                            print(f"  │   ├── {cat:<30s} {ms:8.2f}ms ({pct:5.1f}%) {bar}")

                # 如果还有未分类的
                for cat, ms in sorted_cats:
                    if cat not in shown:
                        pct = ms / block_total * 100 if block_total > 0 else 0
                        bar = _bar(pct)
                        print(f"  │   ├── {cat:<30s} {ms:8.2f}ms ({pct:5.1f}%) {bar}")

                print(f"  └{'─'*89}")

            # Step 级别的汇总
            print(f"  ◆ Step {step_idx} 汇总: {step_total:.1f} ms")

        # ── 全局汇总 (所有 steps 的所有 blocks 叠加) ──
        print(f"\n{'='*100}")
        print(f"  跨 Step 全局算子类别汇总 (All Steps × All Blocks)")
        print(f"{'='*100}")
        print(f"  {'Category':<30s} {'Total(ms)':>10s} {'PerStep(ms)':>12s} {'PerBlock(ms)':>13s} {'%':>7s}")

        all_categories: Dict[str, List[float]] = defaultdict(list)
        for (step, block, cat), durations in groups.items():
            if step >= 0:  # 只统计 step 内部数据
                all_categories[cat].extend(durations)

        grand_total = sum(np.sum(v) / num_runs for v in all_categories.values())
        num_steps = len([s for s in step_data.keys()]) or 1
        num_blocks_per_step = max(
            len(step_data[s]) for s in step_data.keys()
        ) or 32

        cat_summary = []
        for cat, durations in all_categories.items():
            total = np.sum(durations) / num_runs
            per_step = total / num_steps
            per_block = total / (num_steps * num_blocks_per_step)
            pct = total / grand_total * 100 if grand_total > 0 else 0
            cat_summary.append((cat, total, per_step, per_block, pct))

        cat_summary.sort(key=lambda x: x[1], reverse=True)

        for cat, total, per_step, per_block, pct in cat_summary:
            bar = _bar(pct)
            print(f"  {cat:<30s} {total:9.2f}  {per_step:11.2f}  {per_block:12.2f}  {pct:5.1f}% {bar}")

        print(f"  {'─'*30} {'──────────'} {'────────────'} {'─────────────'} {'───────'}")
        print(f"  {'TOTAL':<30s} {grand_total:9.2f} ms")

        # ── 关键发现 ──
        self._print_insights(cat_summary, grand_total)

    def _print_insights(self, cat_summary, grand_total):
        """基于数据分析输出关键发现."""
        cat_map = {c: (t, p) for c, t, _, _, p in cat_summary}

        print(f"\n{'='*100}")
        print(f"  📊 关键发现 / Key Insights")
        print(f"{'='*100}")

        # 1. Attention 总开销
        attn_total = sum(v for c, v, _, _, _ in cat_summary
                         if c in [OpCategory.ATTN_QK, OpCategory.ATTN_SOFTMAX, OpCategory.ATTN_AV])
        if attn_total > 0:
            pct = attn_total / grand_total * 100
            print(f"  ◆ Attention (QK+Softmax+AV) 总计: {attn_total:.1f}ms ({pct:.1f}%)")
            for sub in [OpCategory.ATTN_QK, OpCategory.ATTN_SOFTMAX, OpCategory.ATTN_AV]:
                if sub in cat_map:
                    print(f"     └─ {sub}: {cat_map[sub][0]:.1f}ms ({cat_map[sub][1]:.1f}%)")

        # 2. Linear 投影总开销
        linear_total = sum(v for c, v, _, _, _ in cat_summary
                           if "Linear" in c)
        if linear_total > 0:
            print(f"  ◆ 所有 Linear 投影总计: {linear_total:.1f}ms ({linear_total/grand_total*100:.1f}%)")

        # 3. 归一化总开销
        norm_total = sum(v for c, v, _, _, _ in cat_summary
                         if "Norm" in c or "AdaNorm" in c)
        if norm_total > 0:
            print(f"  ◆ 归一化层总计: {norm_total:.1f}ms ({norm_total/grand_total*100:.1f}%)")

        # 4. 张量整形开销
        tensor_total = sum(v for c, v, _, _, _ in cat_summary
                          if c in [OpCategory.TRANSPOSE, OpCategory.RESHAPE,
                                   OpCategory.CONTIGUOUS, OpCategory.CAST])
        if tensor_total > 0:
            print(f"  ◆ 张量整形/搬运 (Transpose/Reshape/Contiguous/Cast): {tensor_total:.1f}ms ({tensor_total/grand_total*100:.1f}%)")

        # 5. 优化建议
        print(f"\n  💡 优化优先级分析:")
        print(f"  {'─'*80}")

        # 按耗时占比给出针对性建议
        if attn_total > 0 and attn_total / grand_total > 0.3:
            print(f"  P0: Attention 占 {attn_total/grand_total*100:.0f}% → Ascend C 自定义 fused attention kernel")
        linear_total = sum(v for c, v, _, _, _ in cat_summary if "Linear" in c)
        if linear_total > 0 and linear_total / grand_total > 0.2:
            print(f"  P1: Linear 投影占 {linear_total/grand_total*100:.0f}% → ATC 融合 QKV/MLP 为单 kernel")
        tensor_total = sum(v for c, v, _, _, _ in cat_summary
                          if c in [OpCategory.TRANSPOSE, OpCategory.RESHAPE, OpCategory.CONTIGUOUS, OpCategory.CAST])
        if tensor_total > 0 and tensor_total / grand_total > 0.05:
            print(f"  P2: 张量整形占 {tensor_total/grand_total*100:.0f}% → 消除冗余 contiguous/reshape, 用 npu_fused 减少中间张量")


def _bar(pct: float, width: int = 20) -> str:
    filled = max(1, int(pct / 100 * width))
    return "█" * filled


def _get_device_name_safe() -> str:
    try:
        import torch
        if _is_npu():
            return f"Ascend {torch.npu.get_device_name(0)}"
        elif torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


# ═══════════════════════════════════════════════════════════════════════
# 与 ActionHead 推理管线的集成
# ═══════════════════════════════════════════════════════════════════════

class ActionHeadProfiler:
    """
    ActionHead 完整推理管线的精细 Profiler.

    封装了 running 管理, 自动设置 step/block context,
    并统计 encode + action_decoder 等 DiT 之外的开销.
    """

    def __init__(self, model, warmup: int = 5, num_runs: int = 10):
        self.model = model
        self.warmup = warmup
        self.num_runs = num_runs
        self.profiler = FineGrainedProfiler(warmup=warmup)

        # 安装钩子
        dit = model.action_head.model
        self.profiler.install(dit, root_name="dit")

        # 也安装 action_head 级别的重要模块 (encoder/decoder)
        self.profiler.install(
            model.action_head, root_name="action_head"
        )

        # Monkey-patch torch ops
        self.profiler.patch_torch_ops()

    def run(self, backbone_output, action_input):
        """运行完整管线并记录所有 op 耗时."""
        ah = self.model.action_head

        for run_idx in range(self.warmup + self.num_runs):
            self.profiler.begin_run()
            _sync()

            # ── Encode features ──
            self.profiler.set_step(-1)  # pre-denoise
            features = ah._encode_features(backbone_output, action_input)

            # ── Denoising loop ──
            vl_embeds = features.backbone_features
            state_features = features.state_features
            embodiment_id = action_input.embodiment_id

            B = vl_embeds.shape[0]
            device = vl_embeds.device
            actions = torch.randn(
                (B, ah.config.action_horizon, ah.action_dim),
                dtype=vl_embeds.dtype, device=device,
            )
            dt = 1.0 / ah.num_inference_timesteps

            for t_idx in range(ah.num_inference_timesteps):
                self.profiler.set_step(t_idx)
                t_cont = t_idx / float(ah.num_inference_timesteps)
                t_discretized = int(t_cont * ah.num_timestep_buckets)
                timesteps_tensor = torch.full((B,), fill_value=t_discretized, device=device)

                # Action encoder (不属于 DiT, 单独计时)
                action_features = ah.action_encoder(actions, timesteps_tensor, embodiment_id)
                if ah.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = ah.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                sa_embs = torch.cat((state_features, action_features), dim=1)

                # ── DiT forward ──
                # 手动遍历 32 层，为每个 block 设置 context
                # 使用 alternate_vl_dit 的逻辑逐层调用
                dit = ah.model
                temb = dit.timestep_encoder(timesteps_tensor)
                hidden_states = sa_embs.contiguous()
                encoder_hidden_states = vl_embeds.contiguous()

                image_mask = backbone_output.image_mask
                bam = backbone_output.backbone_attention_mask
                image_attn_mask = image_mask & bam
                non_image_attn_mask = (~image_mask) & bam

                for blk_idx, block in enumerate(dit.transformer_blocks):
                    self.profiler.set_block(blk_idx)

                    if blk_idx % 2 == 1:
                        hidden_states = block(
                            hidden_states, attention_mask=None,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None, temb=temb,
                        )
                    else:
                        curr_mask = (non_image_attn_mask
                                     if blk_idx % (2 * dit.attend_text_every_n_blocks) == 0
                                     else image_attn_mask)
                        hidden_states = block(
                            hidden_states, attention_mask=None,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=curr_mask, temb=temb,
                        )

                # Output head
                conditioning = temb
                shift, scale = torch.nn.functional.silu(conditioning).chunk(2, dim=1) if hasattr(dit, 'proj_out_1') else (None, None)
                if shift is not None:
                    shift, scale = dit.proj_out_1(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
                    hidden_states = dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                    model_output = dit.proj_out_2(hidden_states)
                else:
                    model_output = hidden_states

                pred = ah.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -ah.action_horizon:]
                actions = actions + dt * pred_velocity

            self.profiler.end_run()
            _sync()

        self.profiler.print_tree()

    def cleanup(self):
        self.profiler.uninstall()


# ═══════════════════════════════════════════════════════════════════════
# 快速内联 API: 在已有推理代码中插入
# ═══════════════════════════════════════════════════════════════════════

def quick_profile_action_head(model, backbone_output, action_input,
                              warmup: int = 5, num_runs: int = 10):
    """
    快速内联 Profiler —— 在已有的推理代码中调用一次即可.

    用法:
        from scripts.profiling.fine_grained_profiler import quick_profile_action_head

        profiler = quick_profile_action_head(model, bb_out, act_in, warmup=5, num_runs=10)
        # 自动输出树形报告
    """
    p = ActionHeadProfiler(model, warmup=warmup, num_runs=num_runs)
    try:
        p.run(backbone_output, action_input)
    finally:
        p.cleanup()
    return p


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GR00T DiT 细粒度逐算子 Profiler"
    )
    parser.add_argument("--model_path", type=str,
                        default="/root/models/GR00T-N1.6-3B-FP16")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--output_json", type=str, default=None,
                        help="导出 JSON 详细数据")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=66)
    parser.add_argument("--ctx_len", type=int, default=256)
    args = parser.parse_args()

    import torch
    import gr00t.model  # noqa: F401
    from transformers import AutoModel
    from transformers.feature_extraction_utils import BatchFeature

    print(f"设备: {_get_device_name_safe()}")
    print(f"加载模型: {args.model_path}")

    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    model.to(dtype=torch.float16)
    device = next(model.parameters()).device
    print(f"模型设备: {device}")

    # 构造假输入
    B, SEQ, CTX = args.batch_size, args.seq_len, args.ctx_len
    backbone_dim = model.config.backbone_embedding_dim
    max_state_dim = model.config.max_state_dim

    # 构造 backbone features (模拟 Eagle Backbone 输出)
    backbone_features = torch.randn(B, CTX, backbone_dim, dtype=torch.float16, device=device)
    backbone_attention_mask = torch.ones(B, CTX, dtype=torch.bool, device=device)
    # 前半部分是 image tokens, 后半部分是 text tokens
    image_mask = torch.zeros(B, CTX, dtype=torch.bool, device=device)
    image_mask[:, :CTX//2] = True   # 前一半是 image

    backbone_output = BatchFeature({
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask,
        "image_mask": image_mask,
    })

    state = torch.randn(B, max_state_dim, dtype=torch.float16, device=device)
    embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

    action_input = BatchFeature({
        "state": state,
        "embodiment_id": embodiment_id,
    })

    print(f"\n预热 {args.num_warmup} + 计时 {args.num_runs} 次...")
    ActionHeadProfiler(model, warmup=args.num_warmup,
                       num_runs=args.num_runs).run(backbone_output, action_input)

    if args.output_json:
        # 导出详细的逐记录数据
        print(f"TODO: JSON export to {args.output_json}")
        print("(Profiler 实例需要从 run() 中返回以支持 export)")


if __name__ == "__main__":
    main()
