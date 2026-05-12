#!/usr/bin/env python3
"""
GR00T-N1.6 昇腾NPU 优化分析报告生成工具.

基于 profiling 数据生成:
- 时延瀑布图 (waterfall chart)
- 模块热力分布 (heatmap)
- 与 NVIDIA GPU 的对比分析
- 优化建议的自动生成

用法:
    python scripts/profiling/optimization_report.py \
        --profile_json profile_results.json \
        --output report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ─── 理论分析模型 ────────────────────────────────────────────────────

class LatencyModel:
    """
    GR00T-N1.6 在 Ascend 310P3 上的时延分析模型.

    基于模型结构和已知的 NPU 特性, 估算各阶段的理想时延,
    与实测对比后可定位异常热点.
    """

    # 310P3 关键规格
    NPU_FP16_TFLOPS = 32       # FP16 算力 (TOPS)
    NPU_MEM_BW = 56            # 显存带宽 (GB/s) - 310P3 实际约 56 GB/s
    NPU_HBM_SIZE = 43          # 每芯片 HBM (GB)

    # 参考: H100 FP16 = 989 TFLOPS, 310P3 = 32 TFLOPS
    # 算力比 ≈ 30x, 但实际延迟差 8-10x, 说明有优化空间

    def __init__(self):
        pass

    @staticmethod
    def estimate_dit_flops(
        batch: int = 1,
        seq_len: int = 66,     # 1 state + 16 action + 49 others
        ctx_len: int = 256,    # backbone context length
        hidden: int = 1536,    # inner_dim (32 heads × 48 dim)
        num_layers: int = 32,
        num_steps: int = 4,
    ) -> float:
        """估算 DiT 部分的 FLOPS."""
        # 每个 cross-attn block: 2 × QKV投影 + attention + output投影 + FFN
        # QKV: seq_len → hidden, K,V: ctx_len → hidden
        # 简化为: 2 * 4 * hidden^2 * (seq_len + ctx_len) per block
        inner_dim = hidden
        # Per cross-attn block
        qkv_flops = 4 * inner_dim * (seq_len * inner_dim)  # Q projection
        kv_flops = 2 * 2 * inner_dim * (ctx_len * inner_dim)  # K, V projections
        attn_flops = 2 * seq_len * ctx_len * inner_dim       # Q*K^T + attn*V
        ffn_flops = 2 * 4 * inner_dim * inner_dim * seq_len   # FFN (approx)
        per_block = qkv_flops + kv_flops + attn_flops + ffn_flops

        # self-attn blocks: simpler
        sa_qkv = 4 * inner_dim * (seq_len * inner_dim) * 1.5
        sa_attn = 2 * seq_len * seq_len * inner_dim
        sa_ffn = 2 * 4 * inner_dim * inner_dim * seq_len
        per_self_block = sa_qkv + sa_attn + sa_ffn

        total = num_layers // 2 * (per_block + per_self_block) * num_steps * batch
        return total

    @staticmethod
    def estimate_attention_memory(seq_len: int, ctx_len: int, hidden: int) -> float:
        """估算单次 attention 的内存访问量 (bytes)."""
        # Q, K, V 读写 + attention scores + output
        qkv_bytes = 2 * hidden * (seq_len + 2 * ctx_len) * 2  # FP16, read Q,K,V
        score_bytes = seq_len * ctx_len * 4                    # FP32 scores
        out_bytes = hidden * seq_len * 2                       # FP16 output
        return qkv_bytes + score_bytes + out_bytes

    @staticmethod
    def roofline_analysis(
        measured_latency_ms: float, flops: float, mem_bytes: float
    ) -> dict:
        """Roofline 分析: 计算实际算力利用率."""
        measured_s = measured_latency_ms / 1000
        actual_tflops = flops / measured_s / 1e12
        peak_tflops = LatencyModel.NPU_FP16_TFLOPS
        utilization = actual_tflops / peak_tflops * 100

        arith_intensity = flops / mem_bytes  # FLOP/byte
        ridge_point = peak_tflops * 1e12 / (LatencyModel.NPU_MEM_BW * 1e9)

        return {
            "actual_tflops": round(actual_tflops, 2),
            "peak_tflops": peak_tflops,
            "utilization_pct": round(utilization, 1),
            "arith_intensity": round(arith_intensity, 1),
            "ridge_point": round(ridge_point, 1),
            "bound": "memory" if arith_intensity < ridge_point else "compute",
        }


# ─── 优化建议引擎 ────────────────────────────────────────────────────

class OptimizationAdvisor:
    """基于 Profiling 数据自动生成优化建议."""

    ADVICE_TEMPLATES = [
        {
            "id": "atc_compile",
            "condition": lambda stats: True,
            "priority": 1,
            "title": "ATC 模型编译 (OM 格式)",
            "description": (
                "使用昇腾 ATC (Ascend Tensor Compiler) 将 DiT 模型编译为 OM 离线模型。"
                "310P3 的 ATC 可进行算子融合、常数折叠、内存规划，预期 2-3x 加速。"
            ),
            "effort": "中 (2-3天)",
            "expected_gain": "2-3x DiT 加速",
            "steps": [
                "export DiT to ONNX (torch.onnx.export)",
                "atc --model=dit.onnx --framework=5 --output=dit_310p3 --soc_version=Ascend310P3",
                "使用 ACL (AscendCL) C++ API 加载 OM 模型进行推理",
                "注意: 需要将 DiT 从 Eagle backbone 中解耦, 分别推理",
            ],
        },
        {
            "id": "ascend_c_attention",
            "condition": lambda stats: True,
            "priority": 2,
            "title": "Ascend C 自定义 Attention 算子",
            "description": (
                "当前使用手工 matmul+softmax+matmul 实现 attention, 三次独立 kernel launch "
                "且中间 attention scores 矩阵全部读写 HBM。用 Ascend C 编写 fused attention "
                "kernel, 将 Q*K^T + softmax + *V 融合为单 kernel, 在 L1 buffer 中完成在线计算。"
            ),
            "effort": "高 (1-2周)",
            "expected_gain": "DiT attention 部分 3-5x 加速",
            "steps": [
                "使用 Ascend C 编写 Flash-Attention-like kernel",
                "利用 Tiling 将 QKV 分块加载到 L1 buffer",
                "在 L1 中完成 online softmax + rescaling",
                "通过 torch_npu.npu_jit_compile 或 pybind11 集成到 PyTorch",
            ],
        },
        {
            "id": "npu_fusion",
            "condition": lambda stats: True,
            "priority": 3,
            "title": "NPU 算子融合优化",
            "description": (
                "torch_npu 提供了 npu_fused_attention 等融合算子。检查 torch_npu.npu_fused_attention "
                "是否可以替代当前的手工 attention。此外对 AdaLayerNorm + Linear 等模式可以用 "
                "torch.compile(backend='aot_ts_npu') 或 torch_npu.npu_group_norm 融合。"
            ),
            "effort": "低 (1-2天)",
            "expected_gain": "10-20% 延迟降低",
            "steps": [
                "尝试 torch_npu.npu_fused_attention(Q, K, V)",
                "对 DiT 内部的小算子使用 torch_npu.npu_fusion_attention",
                "测试 torch.compile 在 NPU 上的效果 (实验性)",
            ],
        },
        {
            "id": "reduce_denoise_steps",
            "condition": lambda stats: True,
            "priority": 4,
            "title": "减少去噪步数",
            "description": (
                "当前 num_inference_timesteps=4。尝试减少到 2 或 3 步, "
                "使用蒸馏或微调恢复动作质量。可将 DiT 延迟减半。"
            ),
            "effort": "中 (需要重新训练/蒸馏)",
            "expected_gain": "25-50% 总延迟降低",
            "steps": [
                "使用 1-2 步 teacher-forcing 蒸馏",
                "或调整 noise schedule 为更高斯 (更少步数)",
                "测试动作质量是否可接受",
            ],
        },
        {
            "id": "multi_stream",
            "condition": lambda stats: True,
            "priority": 5,
            "title": "多 Stream 并行 (Backbone + Action Head)",
            "description": (
                "当前 backbone 和 action head 串行执行。如果有连续多帧推理, "
                "可以使用双缓冲: backbone 处理下一帧时, action head 处理当前帧。"
            ),
            "effort": "中 (3-5天)",
            "expected_gain": "~30% 吞吐提升 (对单帧延迟无帮助)",
            "steps": [
                "使用 torch.npu.Stream 创建两个 stream",
                "在 stream1 运行 backbone(next_frame), stream2 运行 action_head(current_frame)",
                "通过 event 同步确保依赖关系",
            ],
        },
        {
            "id": "int8_quant",
            "condition": lambda stats: True,
            "priority": 6,
            "title": "INT8 量化推理",
            "description": (
                "310P3 的 INT8 算力为 32 TOPS, 与 FP16 相同。但 INT8 可以: "
                "1) 减少 50% 显存占用; 2) 减少显存带宽压力; 3) 允许更大 batch。"
                "对 memory-bound 的 attention 操作可能有帮助。"
            ),
            "effort": "中 (需要量化校准)",
            "expected_gain": "10-20% 延迟降低 (attention 部分)",
            "steps": [
                "使用 NPU 量化工具 (amct) 对权重和激活进行 INT8 量化",
                "校准数据集: 使用代表性的 backbone features",
                "测试精度损失是否可接受",
            ],
        },
        {
            "id": "reduce_llm_layers",
            "condition": lambda stats: True,
            "priority": 7,
            "title": "减少 Eagle LLM 层数",
            "description": (
                "当前 select_layer=16 (保留 16 层 LLM)。如果 backbone 耗时占比 > 40%, "
                "可尝试 select_layer=12 或 8, 用更少的 LLM 层提取特征。"
                "对动作预测精度的影响需要通过实验验证。"
            ),
            "effort": "低 (改配置即可)",
            "expected_gain": "若 backbone 占 40%, 减少到 8 层可节省 ~20% backbone 延迟",
            "steps": [
                "修改 config.json 中 select_layer 值",
                "重新加载模型测试精度",
                "权衡延迟 vs 精度",
            ],
        },
    ]

    @classmethod
    def generate(cls, stats: dict) -> List[dict]:
        """基于统计数据生成排序后的优化建议."""
        recommendations = []
        for template in cls.ADVICE_TEMPLATES:
            if template["condition"](stats):
                recommendations.append({
                    "priority": template["priority"],
                    "title": template["title"],
                    "description": template["description"],
                    "effort": template["effort"],
                    "expected_gain": template["expected_gain"],
                    "steps": template["steps"],
                })
        recommendations.sort(key=lambda x: x["priority"])
        return recommendations


# ─── 报告生成 ────────────────────────────────────────────────────────

def generate_html_report(
    profile_data: dict,
    output_path: str,
    device_name: str = "Ascend 310P3",
):
    """生成包含图表和优化建议的 HTML 报告."""

    stages = profile_data.get("stages", {})
    if not stages:
        print("错误: profile 数据中没有 stages 信息")
        return

    # 提取数据
    stage_names = []
    stage_avgs = []
    stage_colors = []

    color_map = {
        "prepare": "#6c757d",
        "backbone": "#0d6efd",
        "encode": "#6610f2",
        "denoise": "#dc3545",
        "total": "#198754",
    }

    for name, info in stages.items():
        stage_names.append(name)
        stage_avgs.append(info.get("mean", 0))

        if "prepare" in name:
            stage_colors.append(color_map["prepare"])
        elif "backbone" in name:
            stage_colors.append(color_map["backbone"])
        elif "encode" in name:
            stage_colors.append(color_map["encode"])
        elif "denoise" in name:
            stage_colors.append(color_map["denoise"])
        else:
            stage_colors.append(color_map["total"])

    # 生成优化建议
    advice = OptimizationAdvisor.generate(stages)
    advice_html = ""
    for a in advice:
        steps_html = "".join(f"<li>{s}</li>" for s in a["steps"])
        advice_html += f"""
        <div class="advice-card" style="border-left: 4px solid {'#dc3545' if a['priority'] <= 2 else '#fd7e14' if a['priority'] <= 4 else '#0d6efd'}">
            <h4>[P{a['priority']}] {a['title']}</h4>
            <p>{a['description']}</p>
            <div class="meta">
                <span>🔧 实施难度: {a['effort']}</span>
                <span>📈 预期收益: {a['expected_gain']}</span>
            </div>
            <details>
                <summary>实施步骤</summary>
                <ol>{steps_html}</ol>
            </details>
        </div>
        """

    # 瀑布图数据
    waterfall_items = ""
    for name, info in stages.items():
        mean_val = info.get("mean", 0)
        p95_val = info.get("p95", 0)
        max_bar = max(s["mean"] for s in stages.values()) if stages else 1
        bar_width_pct = mean_val / max_bar * 100 if max_bar > 0 else 0
        waterfall_items += f"""
        <div class="waterfall-item">
            <div class="wf-label">{name}</div>
            <div class="wf-bar-container">
                <div class="wf-bar" style="width:{bar_width_pct}%; background: linear-gradient(90deg, #0d6efd, #dc3545);"></div>
            </div>
            <div class="wf-value">{mean_val:.1f}ms</div>
            <div class="wf-p95">p95: {p95_val:.1f}ms</div>
        </div>
        """

    total_mean = sum(stage_avgs)
    stage_pct_items = ""
    for name, avg in zip(stage_names, stage_avgs):
        pct = avg / total_mean * 100 if total_mean > 0 else 0
        stage_pct_items += f"""
        <div class="pct-item">
            <div class="pct-label">{name}</div>
            <div class="pct-bar-bg">
                <div class="pct-bar" style="width:{pct}%"></div>
            </div>
            <div class="pct-value">{pct:.1f}%</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GR00T-N1.6 昇腾NPU 推理优化分析报告</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; color: #212529; line-height: 1.6; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}

    h1 {{ font-size: 1.8em; margin-bottom: 5px; }}
    h2 {{ font-size: 1.4em; margin: 30px 0 15px; border-bottom: 2px solid #0d6efd; padding-bottom: 8px; }}
    h3 {{ font-size: 1.1em; margin: 20px 0 10px; }}
    .subtitle {{ color: #6c757d; margin-bottom: 20px; }}

    .card {{ background: white; border-radius: 12px; padding: 24px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}

    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
    .metric {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .metric .value {{ font-size: 2em; font-weight: 700; color: #0d6efd; }}
    .metric .label {{ font-size: 0.85em; color: #6c757d; margin-top: 5px; }}

    /* Waterfall */
    .waterfall-item {{ display: flex; align-items: center; margin: 6px 0; gap: 12px; }}
    .wf-label {{ width: 180px; font-size: 0.85em; font-weight: 500; text-align: right; flex-shrink: 0; }}
    .wf-bar-container {{ flex: 1; background: #e9ecef; border-radius: 4px; height: 22px; overflow: hidden; }}
    .wf-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
    .wf-value {{ width: 70px; font-size: 0.9em; font-weight: 600; text-align: right; }}
    .wf-p95 {{ width: 80px; font-size: 0.8em; color: #6c757d; }}

    /* Stage percentage */
    .pct-item {{ display: flex; align-items: center; margin: 5px 0; gap: 10px; }}
    .pct-label {{ width: 160px; font-size: 0.85em; text-align: right; flex-shrink: 0; }}
    .pct-bar-bg {{ flex: 1; background: #e9ecef; border-radius: 4px; height: 18px; overflow: hidden; }}
    .pct-bar {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, #0d6efd, #dc3545); }}
    .pct-value {{ width: 50px; font-size: 0.9em; font-weight: 600; }}

    /* Advice cards */
    .advice-card {{ background: white; border-radius: 10px; padding: 20px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .advice-card h4 {{ margin-bottom: 8px; }}
    .advice-card p {{ font-size: 0.92em; color: #495057; margin-bottom: 10px; }}
    .advice-card .meta {{ display: flex; gap: 20px; font-size: 0.85em; color: #6c757d; margin-bottom: 10px; }}
    .advice-card details {{ font-size: 0.9em; }}
    .advice-card ol {{ margin-left: 20px; }}
    .advice-card li {{ margin: 4px 0; }}

    .comparison-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
    .comparison-table th, .comparison-table td {{ padding: 10px 14px; text-align: center; border-bottom: 1px solid #dee2e6; }}
    .comparison-table th {{ background: #f0f2f5; font-weight: 600; font-size: 0.9em; }}
    .comparison-table .highlight {{ background: #fff3cd; font-weight: 600; }}

    .footer {{ text-align: center; color: #adb5bd; font-size: 0.8em; margin-top: 40px; padding: 20px; }}
</style>
</head>
<body>
<div class="container">
    <h1>GR00T-N1.6-3B 推理优化分析报告</h1>
    <p class="subtitle">设备: {device_name} | 模型: Gr00tN1d6 | 精度: FP16 | 生成时间: 自动</p>

    <h2>1. 关键指标概览</h2>
    <div class="metrics-grid">
        <div class="metric">
            <div class="value">{total_mean:.1f}ms</div>
            <div class="label">端到端平均延迟</div>
        </div>
        <div class="metric">
            <div class="value">{1000/(total_mean if total_mean > 0 else 1):.1f} Hz</div>
            <div class="label">吞吐量</div>
        </div>
    </div>

    <h2>2. 管线阶段耗时分布</h2>
    <div class="card">
        <h3>时延瀑布图 (Waterfall)</h3>
        <p style="color:#6c757d;font-size:0.85em;margin-bottom:12px;">各阶段的平均耗时对比</p>
        {waterfall_items}
    </div>

    <div class="card">
        <h3>耗时占比</h3>
        {stage_pct_items}
    </div>

    <h2>3. 与 NVIDIA GPU 对比</h2>
    <div class="card">
        <table class="comparison-table">
            <tr><th>硬件</th><th>延迟</th><th>吞吐量</th><th>相对 Ascend</th></tr>
            <tr><td>H100 (torch.compile)</td><td>38 ms</td><td>26.3 Hz</td><td>8.0x 快</td></tr>
            <tr><td>RTX 5090 (torch.compile)</td><td>37 ms</td><td>27.3 Hz</td><td>8.2x 快</td></tr>
            <tr><td>RTX 4090 (torch.compile)</td><td>44 ms</td><td>22.8 Hz</td><td>6.9x 快</td></tr>
            <tr><td>Jetson Thor (torch.compile)</td><td>105 ms</td><td>9.5 Hz</td><td>2.9x 快</td></tr>
            <tr class="highlight"><td>Ascend 310P3 (本方案)</td><td>{total_mean:.0f} ms</td><td>{1000/(total_mean if total_mean > 0 else 1):.1f} Hz</td><td>基准</td></tr>
        </table>
    </div>

    <h2>4. 优化建议 (按优先级排序)</h2>
    {advice_html}

    <h2>5. 优化路线图 (推荐方案)</h2>
    <div class="card">
        <h3>阶段1: 快速见效 (1-2周)</h3>
        <ol>
            <li><strong>NPU 算子融合</strong>: 将 AdaLayerNorm+Linear 等模式用 torch_npu 融合算子替换 → 预计节省 10-20%</li>
            <li><strong>减少 LLM 层数验证</strong>: 测试 select_layer=12/8 对精度的影响 → 若可行可节省 backbone 延迟</li>
        </ol>

        <h3>阶段2: 中等投入 (2-4周)</h3>
        <ol>
            <li><strong>ATC 模型编译</strong>: DiT 导出 ONNX → ATC 编译为 OM 格式 → 预计 2-3x DiT 加速</li>
            <li><strong>去噪步数优化</strong>: 蒸馏或调参减少到 2-3 步 → 几乎线性减少 DiT 延迟</li>
        </ol>

        <h3>阶段3: 深度优化 (1-2月)</h3>
        <ol>
            <li><strong>Ascend C 自定义 Attention</strong>: 实现 Ascend C 版 Flash Attention → 预计 DiT attention 3-5x 加速</li>
            <li><strong>多 Stream 流水线</strong>: Backbone 和 Action Head 流水线并行 → 吞吐提升</li>
        </ol>
    </div>

    <div class="footer">
        Generated by GR00T Profiling Toolkit | Ascend 310P3 Optimization Report
    </div>
</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"报告已生成: {output_path}")


def generate_text_report(profile_data: dict, device_name: str = "Ascend 310P3"):
    """生成纯文本分析报告."""
    stages = profile_data.get("stages", {})

    print(f"\n{'='*70}")
    print(f"  GR00T-N1.6 昇腾NPU 推理优化分析报告")
    print(f"  设备: {device_name}")
    print(f"{'='*70}")

    print(f"\n  📊 管线阶段耗时分布:\n")
    total_mean = 0
    for name, info in stages.items():
        total_mean += info.get("mean", 0)

    for name, info in stages.items():
        mean_val = info.get("mean", 0)
        p95_val = info.get("p95", 0)
        pct = mean_val / total_mean * 100 if total_mean > 0 else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  {name:<30s} {mean_val:7.1f}ms (p95={p95_val:.1f}ms) {pct:5.1f}%  {bar}")

    print(f"  {'─'*30} {'───────'} {'─────────'} {'─────'}")
    print(f"  {'TOTAL':<30s} {total_mean:7.1f}ms")

    print(f"\n  📈 优化建议 (按优先级):\n")
    advice = OptimizationAdvisor.generate(stages)
    for i, a in enumerate(advice, 1):
        print(f"  [{a['priority']}] {a['title']}")
        print(f"      难度: {a['effort']} | 预期: {a['expected_gain']}")
        print(f"      {a['description'][:100]}...")
        print()


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GR00T-N1.6 NPU 优化分析报告生成"
    )
    parser.add_argument(
        "--profile_json", type=str, default=None,
        help="Profiling JSON 数据 (latency_profiler.py 的输出)"
    )
    parser.add_argument(
        "--output", type=str, default="optimization_report.html",
        help="HTML 报告输出路径"
    )
    parser.add_argument(
        "--format", type=str, default="html",
        choices=["html", "text"],
        help="报告格式"
    )
    args = parser.parse_args()

    device_name = "Ascend 310P3"

    if args.profile_json:
        with open(args.profile_json) as f:
            profile_data = json.load(f)
    else:
        # 使用部署文档中的已知数据生成示例报告
        profile_data = {
            "device": "Ascend 310P3",
            "stages": {
                "backbone": {"mean": 45.0, "std": 3.0, "min": 40.0, "max": 52.0, "p95": 50.0},
                "encode_features": {"mean": 2.5, "std": 0.3, "min": 2.0, "max": 3.5, "p95": 3.0},
                "denoise_step_0": {"mean": 63.0, "std": 3.0, "min": 58.0, "max": 70.0, "p95": 68.0},
                "denoise_step_1": {"mean": 62.5, "std": 2.8, "min": 57.0, "max": 69.0, "p95": 67.0},
                "denoise_step_2": {"mean": 62.8, "std": 3.1, "min": 58.0, "max": 71.0, "p95": 68.5},
                "denoise_step_3": {"mean": 63.2, "std": 3.2, "min": 57.5, "max": 72.0, "p95": 69.0},
                "total": {"mean": 304.6, "std": 4.0, "min": 295.2, "max": 314.1, "p95": 312.0},
            },
        }
        print("使用部署文档中的基准数据生成报告...")

    # Roofline 分析
    diT_flops = LatencyModel.estimate_dit_flops()
    diT_mem = LatencyModel.estimate_attention_memory(66, 256, 1536) * 32 * 4
    roofline = LatencyModel.roofline_analysis(304.0, diT_flops, diT_mem)
    print(f"\n  Roofline 分析: 实际 {roofline['actual_tflops']} TFLOPS / "
          f"峰值 {roofline['peak_tflops']} TFLOPS "
          f"(利用率 {roofline['utilization_pct']}%)")
    print(f"  算数强度: {roofline['arith_intensity']} FLOP/byte, "
          f"脊点: {roofline['ridge_point']} FLOP/byte")
    print(f"  瓶颈: {'带宽受限 (Memory-bound)' if roofline['bound'] == 'memory' else '算力受限 (Compute-bound)'}")

    if args.format == "html":
        generate_html_report(profile_data, args.output, device_name)
    else:
        generate_text_report(profile_data, device_name)


if __name__ == "__main__":
    main()
