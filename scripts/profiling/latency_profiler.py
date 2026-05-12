#!/usr/bin/env python3
"""
GR00T-N1.6 昇腾NPU 推理时延分析工具.

通过 torch 的 register_forward_pre_hook / register_forward_hook
对整个推理管线进行分层计时，输出各模块的延迟统计和热力分布。

用法:
    python scripts/profiling/latency_profiler.py \
        --model_path /root/models/GR00T-N1.6-3B-FP16 \
        --num_warmup 5 --num_runs 20

    # 或用于已有推理脚本中:
    from scripts.profiling.latency_profiler import Profiler, profile_model
    profiler = profile_model(model)
    output = model.get_action(inputs)
    profiler.print_stats()
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


# ─── NPU / CUDA 兼容 ────────────────────────────────────────────────

def _is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def _sync_device():
    """同步设备，确保之前的异步操作全部完成后再计时."""
    if _is_npu_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_device_name() -> str:
    if _is_npu_available():
        return f"Ascend {torch.npu.get_device_name(0)}"
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


# ─── 计时记录 ────────────────────────────────────────────────────────

class TimingRecord:
    """单次计时的完整记录."""

    __slots__ = (
        "name", "start_ms", "end_ms", "duration_ms",
        "self_duration_ms", "children",
    )

    def __init__(self, name: str):
        self.name = name
        self.start_ms: float = 0.0
        self.end_ms: float = 0.0
        self.duration_ms: float = 0.0
        self.self_duration_ms: float = 0.0  # 减去子节点后的纯耗时
        self.children: List[TimingRecord] = []


class TimingStats:
    """多次运行的聚合统计."""

    def __init__(self, name: str, depth: int = 0):
        self.name = name
        self.depth = depth
        self.durations_ms: List[float] = []     # 总耗时(含子节点)
        self.self_durations_ms: List[float] = []  # 纯耗时(不含子节点)
        self.count: int = 0

    def add(self, record: TimingRecord):
        self.durations_ms.append(record.duration_ms)
        self.self_durations_ms.append(record.self_duration_ms)
        self.count += 1

    def summary(self) -> dict:
        if not self.durations_ms:
            return {"name": self.name, "count": 0}
        arr = np.array(self.durations_ms)
        self_arr = np.array(self.self_durations_ms)
        return {
            "name": self.name,
            "depth": self.depth,
            "count": self.count,
            "total_ms__mean": float(np.mean(arr)),
            "total_ms__std": float(np.std(arr)),
            "total_ms__min": float(np.min(arr)),
            "total_ms__max": float(np.max(arr)),
            "total_ms__p50": float(np.percentile(arr, 50)),
            "total_ms__p95": float(np.percentile(arr, 95)),
            "total_ms__p99": float(np.percentile(arr, 99)),
            "self_ms__mean": float(np.mean(self_arr)),
            "self_ms__p95": float(np.percentile(self_arr, 95)),
            "self_ms__sum": float(np.sum(self_arr)),
        }


# ─── 核心 Profiler ──────────────────────────────────────────────────

class Profiler:
    """分层模型 Profiler.

    通过 register_forward_pre_hook / register_forward_hook 为每个子模块
    安装计时钩子，在整个推理过程中记录每个模块的耗时。
    """

    def __init__(self, warmup: int = 5):
        self.warmup = warmup
        self._run_count = 0
        self._trace: List[TimingRecord] = []       # 每次 run 的根 trace
        self._current_trace: Optional[TimingRecord] = None
        self._stack: List[TimingRecord] = []
        self._hooks: List[tuple] = []              # (module, pre_handle, fwd_handle)
        self._event_stack: List[Tuple[str, float]] = []  # (name, start_ts)
        self._stats: Dict[str, TimingStats] = {}   # name → aggregated stats
        self._name_counts: Dict[str, int] = defaultdict(int)

    # ── 钩子安装 ──────────────────────────────────────────────────

    def install_on(self, module: torch.nn.Module, name_prefix: str = ""):
        """递归为 module 及其所有子模块安装计时钩子."""
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            self._install_hooks(child, full_name)
            self.install_on(child, full_name)

    def _install_hooks(self, module: torch.nn.Module, name: str):
        profiler = self

        def pre_hook(mod, inp):
            _sync_device()
            ts = time.perf_counter()
            profiler._event_stack.append((name, ts))

        def fwd_hook(mod, inp, out):
            _sync_device()
            ts = time.perf_counter()
            if profiler._event_stack:
                evt_name, start_ts = profiler._event_stack.pop()
                # 名字对不上说明嵌套出了问题，容错处理
                if evt_name != name:
                    # 可能是嵌套调用，尝试找到匹配的
                    for i in range(len(profiler._event_stack) - 1, -1, -1):
                        if profiler._event_stack[i][0] == name:
                            evt_name, start_ts = profiler._event_stack.pop(i)
                            break
                duration = (ts - start_ts) * 1000.0
                profiler._record(name, duration)

        pre_handle = module.register_forward_pre_hook(pre_hook)
        fwd_handle = module.register_forward_hook(fwd_hook)
        self._hooks.append((module, pre_handle, fwd_handle))

    def _record(self, name: str, duration_ms: float):
        """记录一次模块耗时并维护层级关系."""
        if self._run_count < self.warmup:
            return

        # 找父节点: 栈顶且 depth 等于当前层级
        # 简化处理: 直接用 name 层级关系重建树
        rec = TimingRecord(name)
        rec.duration_ms = duration_ms

        # 根据 name 层级找到父节点
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name = parts[0]
        else:
            parent_name = None

        self._trace.append((name, parent_name, rec))

    # ── 运行管理 ──────────────────────────────────────────────────

    def start_run(self):
        """开始一次推理计时的上下文管理器入口."""
        self._event_stack.clear()
        self._trace.clear()

    def end_run(self):
        """结束一次推理计时."""
        self._run_count += 1
        if self._run_count <= self.warmup:
            return

        # 建立树结构
        root = self._build_tree()
        if root is not None:
            self._accumulate(root)

    @contextmanager
    def run(self):
        """with profiler.run(): 上下文中运行的单次推理."""
        self.start_run()
        yield
        self.end_run()

    def _build_tree(self) -> Optional[TimingRecord]:
        """根据 name 层级关系构建调用树,计算 self duration."""
        if not self._trace:
            return None

        # 按 name 中的 '.' 数量确定深度
        # 按记录顺序重建父子关系
        nodes: Dict[str, TimingRecord] = {}
        for name, parent_name, rec in self._trace:
            nodes[name] = rec
            rec.children = []
            if parent_name and parent_name in nodes:
                nodes[parent_name].children.append(rec)

        # 找出根节点(没有父节点的)
        all_parents = {p for _, p, _ in self._trace if p is not None}
        roots = [rec for name, _, rec in self._trace if name not in all_parents]

        if not roots:
            return None

        # 如果有多个根, 创建人造根
        if len(roots) == 1:
            root = roots[0]
        else:
            root = TimingRecord("__total__")
            root.children = roots
            root.duration_ms = sum(r.duration_ms for r in roots)

        # 计算 self_duration (总耗时 - 直接子节点耗时之和)
        def _compute_self(node: TimingRecord):
            children_sum = 0.0
            for child in node.children:
                _compute_self(child)
                children_sum += child.duration_ms
            node.self_duration_ms = max(0.0, node.duration_ms - children_sum)

        _compute_self(root)
        return root

    def _accumulate(self, root: TimingRecord):
        """将一次 trace 累加到统计中."""

        def _walk(node: TimingRecord, depth: int):
            if node.name not in self._stats:
                self._stats[node.name] = TimingStats(node.name, depth)
            self._stats[node.name].add(node)
            for child in node.children:
                _walk(child, depth + 1)

        _walk(root, 0)

    # ── 卸载 ──────────────────────────────────────────────────────

    def uninstall(self):
        """移除所有已安装的钩子."""
        for mod, pre_h, fwd_h in self._hooks:
            pre_h.remove()
            fwd_h.remove()
        self._hooks.clear()

    # ── 结果输出 ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, TimingStats]:
        return self._stats

    def print_stats(self, sort_by: str = "self_ms__mean", top_n: int = 30):
        """打印延迟统计表."""
        if not self._stats:
            print("[Profiler] 没有统计数据 (warmup 可能还未结束)")
            return

        entries = [(name, s.summary()) for name, s in self._stats.items()]
        entries.sort(key=lambda x: x[1].get(sort_by, 0), reverse=True)
        entries = entries[:top_n]

        header = (
            f"{'Module':<55s} {'Self(ms)':>8s} {'Total(ms)':>8s} "
            f"{'%':>6s} {'P95(ms)':>8s} {'Count':>6s}"
        )
        print(f"\n{'='*len(header)}")
        print(f"  GR00T-N1.6 推理延迟热力统计 (设备: {_get_device_name()})")
        print(f"{'='*len(header)}")
        print(header)
        print("-" * len(header))

        total_ms = sum(s.summary()["self_ms__sum"] for _, s in self._stats.items())
        if total_ms == 0:
            return

        for name, info in entries:
            self_pct = info["self_ms__sum"] / total_ms * 100 if total_ms > 0 else 0
            # 缩进表示层级
            indent = "  " * info.get("depth", 0)
            display_name = indent + name.split(".")[-1] if info.get("depth", 0) > 0 else name
            # 截断过长的名字
            if len(display_name) > 52:
                display_name = "..." + display_name[-49:]

            print(
                f"{display_name:<55s} {info['self_ms__mean']:7.2f}  "
                f"{info['total_ms__mean']:7.2f}  {self_pct:5.1f}% "
                f"{info['self_ms__p95']:7.2f}  {info['count']:5d}  "
            )

        print("-" * len(header))
        print(f"  Total self time: {total_ms:.2f} ms")
        print()

    def print_hotspots(self, top_n: int = 15):
        """以热力排名方式输出 Top-N 耗时步骤."""
        if not self._stats:
            print("[Profiler] 没有统计数据")
            return

        entries = []
        for name, stats in self._stats.items():
            s = stats.summary()
            entries.append((name, s))

        entries.sort(key=lambda x: x[1]["self_ms__p95"], reverse=True)
        entries = entries[:top_n]

        max_name = max(len(e["name"]) for _, e in entries)

        print(f"\n  🔥 Top-{top_n} 耗时热点 (按 P95 self time 排序)")
        print(f"  {'Rank':<5s} {'Module':<{max_name + 2}s} {'Self P95':>10s}  {'Self Avg':>10s}  {'Total P95':>10s}")
        print(f"  {'-'*5} {'-'*(max_name+2)} {'-'*10} {'-'*10} {'-'*10}")

        for rank, (name, info) in enumerate(entries, 1):
            bar_len = int(info["self_ms__p95"] / entries[0][1]["self_ms__p95"] * 20) if entries[0][1]["self_ms__p95"] > 0 else 0
            bar = "█" * bar_len
            print(
                f"  {rank:<5d} {name:<{max_name + 2}s} "
                f"{info['self_ms__p95']:8.2f}ms  {info['self_ms__mean']:8.2f}ms  "
                f"{info['total_ms__p95']:8.2f}ms  {bar}"
            )
        print()

    def export_json(self, filepath: str):
        """导出统计数据为 JSON."""
        data = {
            "device": _get_device_name(),
            "num_runs": max(
                (s.count for s in self._stats.values()), default=0
            ),
            "modules": {
                name: stats.summary() for name, stats in self._stats.items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Profiler] 统计已导出到 {filepath}")

    def to_dataframe(self):
        """返回 pandas DataFrame (如果可用)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas 未安装, 无法导出 DataFrame")

        records = []
        for name, stats in self._stats.items():
            s = stats.summary()
            records.append(s)
        return pd.DataFrame(records)


# ─── 便捷 API ───────────────────────────────────────────────────────

def profile_model(model: torch.nn.Module, warmup: int = 5) -> Profiler:
    """一键为模型安装 profiler 并返回."""
    profiler = Profiler(warmup=warmup)
    profiler.install_on(model, name_prefix="model")
    return profiler


def profile_with_manual_steps(
    model,
    input_builder,
    warmup: int = 5,
    num_runs: int = 20,
    npu_timeline: bool = False,
) -> Profiler:
    """
    手动分步骤的精细 Profiler.

    将推理管线拆解为关键步骤并分别计时:
    - backbone: Eagle 视觉+语言编码
    - encode_features: VLLN + state encoder
    - denoising_step_{0..N}: DiT 去噪循环每一步
    - decode_action: action decoder
    """
    profiler = ManualStepProfiler(warmup=warmup, npu_timeline=npu_timeline)

    for _ in range(warmup + num_runs):
        inputs = input_builder()
        profiler.run_one(model, inputs)

    profiler.print_stats()
    return profiler


class ManualStepProfiler:
    """分步骤的精细 Profiler，比 hook-based 更精确地反映管线阶段.

    将推理分为:
    1. backbone - 视觉编码 + 语言模型
    2. encode  - VLLN + state_encoder
    3. denoise step 0..3 - 每个 DiT forward
    4. total   - 端到端
    """

    def __init__(self, warmup: int = 5, npu_timeline: bool = False):
        self.warmup = warmup
        self.npu_timeline = npu_timeline
        self._run = 0
        self.steps: Dict[str, List[float]] = defaultdict(list)

    def _time(self) -> float:
        _sync_device()
        return time.perf_counter()

    def run_one(self, model, inputs: dict):
        """执行一次计时推理."""
        self._run += 1

        t_start = self._time()

        # Step 1: Prepare inputs
        t0 = self._time()
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        t_prep = self._time()

        # Step 2: Backbone forward
        backbone_output = model.backbone(**backbone_inputs)
        t_backbone = self._time()

        # Step 3: Encode features (VLLN + state encoder)
        features = model.action_head._encode_features(backbone_output, action_inputs)
        t_encode = self._time()

        # Step 4: Denoising loop - each step separately
        vl_embeds = features.backbone_features
        state_features = features.state_features
        embodiment_id = action_inputs.embodiment_id
        device = vl_embeds.device
        batch_size = vl_embeds.shape[0]

        actions = torch.randn(
            size=(batch_size, model.action_head.config.action_horizon,
                  model.action_head.action_dim),
            dtype=vl_embeds.dtype, device=device,
        )
        dt = 1.0 / model.action_head.num_inference_timesteps

        denoise_steps = []
        for t_idx in range(model.action_head.num_inference_timesteps):
            t_cont = t_idx / float(model.action_head.num_inference_timesteps)
            t_discretized = int(t_cont * model.action_head.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = model.action_head.action_encoder(
                actions, timesteps_tensor, embodiment_id
            )
            if model.action_head.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                pos_embs = model.action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            t_step_start = self._time()

            if model.action_head.config.use_alternate_vl_dit:
                model_output = model.action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = model.action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )

            t_step_end = self._time()
            denoise_steps.append(t_step_end - t_step_start)

            pred = model.action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -model.action_head.action_horizon:]
            actions = actions + dt * pred_velocity

        t_end = self._time()

        if self._run <= self.warmup:
            return

        self.steps["01__prepare_input"].append((t_prep - t_start) * 1000)
        self.steps["02__backbone"].append((t_backbone - t_prep) * 1000)
        self.steps["03__encode_features"].append((t_encode - t_backbone) * 1000)
        for i, d in enumerate(denoise_steps):
            self.steps[f"04__denoise_step_{i}"].append(d * 1000)
        self.steps["05__total_end_to_end"].append((t_end - t_start) * 1000)

    def print_stats(self):
        if not self.steps:
            print("[ManualStepProfiler] 无数据")
            return

        print(f"\n{'='*80}")
        print(f"  GR00T-N1.6 推理管线阶段耗时分解 ({_get_device_name()})")
        print(f"  Inference Pipeline Stage Breakdown")
        print(f"{'='*80}")
        print(f"  {'Stage':<30s} {'Avg(ms)':>8s} {'Min(ms)':>8s} "
              f"{'Max(ms)':>8s} {'P95(ms)':>8s} {'%':>6s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

        total_avg = np.mean(self.steps.get("05__total_end_to_end", [0]) or [0])
        if total_avg == 0:
            total_avg = sum(np.mean(v) for v in self.steps.values() if v)

        stage_order = [
            "01__prepare_input",
            "02__backbone",
            "03__encode_features",
        ]
        stage_order += [k for k in sorted(self.steps) if k.startswith("04__denoise")]
        stage_order.append("05__total_end_to_end")

        for key in stage_order:
            values = self.steps.get(key, [])
            if not values:
                continue
            arr = np.array(values)
            pct = np.mean(arr) / total_avg * 100 if total_avg > 0 else 0
            label = key.replace("__", " ").replace("_", " ")
            print(
                f"  {label:<30s} {np.mean(arr):7.2f}  {np.min(arr):7.2f}  "
                f"{np.max(arr):7.2f}  {np.percentile(arr, 95):7.2f}  {pct:5.1f}%"
            )

        # 汇总 denoise 步骤
        denoise_values = []
        for k, v in self.steps.items():
            if k.startswith("04__denoise"):
                denoise_values.extend(v)
        if denoise_values:
            arr = np.array(denoise_values)
            print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
            pct = np.sum(arr) / (total_avg * len(self.steps.get("05__total_end_to_end", [1]))) * 100
            print(
                f"  {'denoise ALL STEPS (total)':<30s} {np.sum(arr):7.2f}  "
                f"{'':>8s} {'':>8s} {'':>8s}  {pct:5.1f}%"
            )
        print()


# ─── DiT 层级分析 ───────────────────────────────────────────────────

class DiTLayerProfiler:
    """单独分析 DiT 32 层 transformer 的每层耗时.

    用于精确定位 DiT 内部哪些层是热点.
    """

    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self._run = 0
        self.layer_times: Dict[int, List[float]] = defaultdict(list)
        self.block_type: Dict[int, str] = {}  # "cross_attn" or "self_attn"
        self._hooks = []

    def install_on(self, dit_model):
        """为 DiT 的每个 transformer block 安装计时钩子."""
        profiler = self

        for idx, block in enumerate(dit_model.transformer_blocks):
            is_cross = (idx % 2 == 0)
            block_type = "cross_attn" if is_cross else "self_attn"
            self.block_type[idx] = block_type

            def make_hooks(layer_idx):
                start_ts = [None]

                def pre_hook(mod, inp):
                    _sync_device()
                    start_ts[0] = time.perf_counter()

                def fwd_hook(mod, inp, out):
                    _sync_device()
                    if start_ts[0] is not None:
                        duration = (time.perf_counter() - start_ts[0]) * 1000
                        if profiler._run > profiler.warmup:
                            profiler.layer_times[layer_idx].append(duration)

                return pre_hook, fwd_hook

            pre, fwd = make_hooks(idx)
            pre_h = block.register_forward_pre_hook(pre)
            fwd_h = block.register_forward_hook(fwd)
            self._hooks.append((block, pre_h, fwd_h))

    def begin_run(self):
        self._run += 1

    def uninstall(self):
        for mod, pre_h, fwd_h in self._hooks:
            pre_h.remove()
            fwd_h.remove()
        self._hooks.clear()

    def print_stats(self):
        if not self.layer_times:
            print("[DiTLayerProfiler] 无数据")
            return

        print(f"\n{'='*80}")
        print(f"  DiT 32层 Transformer Block 逐层耗时分析 (每次去噪步骤)")
        print(f"  Per-Layer Latency Breakdown")
        print(f"{'='*80}")
        print(f"  {'Layer':<6s} {'Type':<14s} {'Avg(ms)':>9s} {'P95(ms)':>9s} "
              f"{'Min(ms)':>9s} {'Max(ms)':>9s} {'%Total':>7s}")
        print(f"  {'-'*6} {'-'*14} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")

        total_mean = sum(
            np.mean(v) for v in self.layer_times.values() if v
        )

        for idx in sorted(self.layer_times.keys()):
            arr = np.array(self.layer_times[idx])
            btype = self.block_type.get(idx, "?")
            pct = np.mean(arr) / total_mean * 100 if total_mean > 0 else 0
            bar = "█" * max(1, int(pct / 2))
            print(
                f"  L{idx:<5d} {btype:<14s} {np.mean(arr):8.3f}  "
                f"{np.percentile(arr, 95):8.3f}  {np.min(arr):8.3f}  "
                f"{np.max(arr):8.3f}  {pct:5.1f}%  {bar}"
            )

        # 汇总
        cross_attn_total = sum(
            np.mean(self.layer_times[i]) for i in self.layer_times
            if self.block_type.get(i) == "cross_attn"
        )
        self_attn_total = sum(
            np.mean(self.layer_times[i]) for i in self.layer_times
            if self.block_type.get(i) == "self_attn"
        )
        print(f"  {'-'*6} {'-'*14} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")
        print(
            f"  CROSS-ATTN (16层) sum: {cross_attn_total:.3f} ms "
            f"({cross_attn_total/total_mean*100:.1f}%)" if total_mean > 0 else ""
        )
        print(
            f"  SELF-ATTN  (16层) sum: {self_attn_total:.3f} ms "
            f"({self_attn_total/total_mean*100:.1f}%)" if total_mean > 0 else ""
        )
        print()


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GR00T-N1.6 NPU 推理延迟分析工具"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="/root/models/GR00T-N1.6-3B-FP16",
        help="模型路径"
    )
    parser.add_argument(
        "--num_warmup", type=int, default=5,
        help="预热运行次数"
    )
    parser.add_argument(
        "--num_runs", type=int, default=20,
        help="计时运行次数"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录 (自动持久化 JSON + TXT, 默认自动创建时间戳目录)"
    )
    parser.add_argument(
        "--mode", type=str, default="steps",
        choices=["steps", "hooks", "dit_layers", "all"],
        help="分析模式: steps=管线阶段, hooks=模块钩子, dit_layers=DiT层级, all=全部"
    )
    args = parser.parse_args()

    # 导入模型注册
    import gr00t.model  # noqa: F401
    from transformers import AutoModel

    print(f"设备: {_get_device_name()}")
    print(f"加载模型: {args.model_path}")

    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    model.to(dtype=torch.float16)

    device = next(model.parameters()).device
    print(f"模型所在设备: {device}")

    # 构造假输入（需要根据实际场景调整）
    # 这里使用最小有效输入，实际使用时替换为真实数据
    def build_dummy_inputs():
        batch_size = 1
        seq_len = 256
        backbone_dim = model.config.backbone_embedding_dim
        hidden_dim = model.config.hidden_size
        max_state_dim = model.config.max_state_dim
        max_action_dim = model.config.max_action_dim
        action_horizon = model.config.action_horizon

        backbone_output = {
            "backbone_features": torch.randn(
                batch_size, seq_len, backbone_dim,
                dtype=torch.float16, device=device
            ),
            "backbone_attention_mask": torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=device
            ),
            "image_mask": torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=device
            ),
        }
        action_input = {
            "state": torch.randn(batch_size, max_state_dim, dtype=torch.float16, device=device),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        }
        # 构造 get_action 的完整输入
        from transformers.feature_extraction_utils import BatchFeature
        backbone_batch = BatchFeature(backbone_output)
        action_batch = BatchFeature(action_input)
        return backbone_batch, action_batch

    print(f"\n预热 {args.num_warmup} 次 + 计时 {args.num_runs} 次...")

    if args.mode in ("steps", "all"):
        profiler = ManualStepProfiler(warmup=args.num_warmup)
        for _ in range(args.num_warmup + args.num_runs):
            bb, ai = build_dummy_inputs()
            # 简化版：直接调用 action head 的 get_action_with_features
            profiler._run += 1
            t_start = profiler._time()

            # encode
            features = model.action_head._encode_features(bb, ai)
            t_encode = profiler._time()

            # run denoising
            result = model.action_head.get_action_with_features(
                backbone_features=features.backbone_features,
                state_features=features.state_features,
                embodiment_id=ai.embodiment_id,
                backbone_output=bb,
            )
            t_end = profiler._time()

            if profiler._run > profiler.warmup:
                profiler.steps["encode"].append((t_encode - t_start) * 1000)
                profiler.steps["denoise_total"].append((t_end - t_encode) * 1000)
                profiler.steps["05__total_end_to_end"].append((t_end - t_start) * 1000)

        profiler.print_stats()

    if args.mode in ("dit_layers", "all"):
        dit_profiler = DiTLayerProfiler(warmup=args.num_warmup)
        dit_profiler.install_on(model.action_head.model)

        for _ in range(args.num_warmup + args.num_runs):
            dit_profiler.begin_run()
            bb, ai = build_dummy_inputs()
            features = model.action_head._encode_features(bb, ai)
            _ = model.action_head.get_action_with_features(
                backbone_features=features.backbone_features,
                state_features=features.state_features,
                embodiment_id=ai.embodiment_id,
                backbone_output=bb,
            )

        dit_profiler.print_stats()
        dit_profiler.uninstall()

    if args.mode in ("hooks", "all"):
        hook_profiler = Profiler(warmup=args.num_warmup)

        # 只为 action_head 安装钩子 (backbone 太大，钩子开销也大)
        hook_profiler.install_on(model.action_head, name_prefix="action_head")

        for _ in range(args.num_warmup + args.num_runs):
            with hook_profiler.run():
                bb, ai = build_dummy_inputs()
                _ = model.action_head.get_action(bb, ai)

        hook_profiler.print_stats()
        hook_profiler.uninstall()

    # ── 持久化保存 ──
    from scripts.profiling._persist import ResultCollector
    rc = ResultCollector("latency", args.output_dir)

    if args.mode in ("steps", "all"):
        data = {
            "device": _get_device_name(),
            "num_runs": args.num_runs,
            "mode": "steps",
            "stages": {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "p95": float(np.percentile(v, 95)),
                }
                for k, v in profiler.steps.items() if v
            },
        }
        rc.add_json("stage_breakdown", data)
        # 生成文本报告
        lines = ["GR00T 管线阶段耗时 (latency_profiler)", f"设备: {_get_device_name()}"]
        for k, v in data["stages"].items():
            lines.append(f"  {k:<30s} mean={v['mean']:7.1f}ms  p95={v['p95']:7.1f}ms")
        rc.add_text("stage_breakdown", "\n".join(lines))

    if args.mode in ("hooks", "all") and "hook_profiler" in dir():
        hook_data = {
            "device": _get_device_name(),
            "num_runs": args.num_runs,
            "mode": "hooks",
            "modules": {name: stats.summary()
                        for name, stats in hook_profiler.get_stats().items()},
        }
        rc.add_json("module_hooks", hook_data)

    if args.mode in ("dit_layers", "all") and "dit_profiler" in dir():
        dit_data = {
            "device": _get_device_name(),
            "mode": "dit_layers",
            "layers": {
                str(idx): {
                    "type": dit_profiler.block_type.get(idx, "unknown"),
                    "mean": float(np.mean(arr)),
                    "p95": float(np.percentile(arr, 95)),
                }
                for idx, arr in dit_profiler.layer_times.items() if arr
            },
        }
        rc.add_json("dit_layers", dit_data)

    rc.save_all()


if __name__ == "__main__":
    main()
