"""
Profiling 工具持久化公共模块.

提供统一的结果保存接口，所有 profiling 工具通过此模块自动持久化输出.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _get_default_output_dir(tool_name: str) -> str:
    """获取默认输出目录: profiling_results/YYYY-MM-DD_HH-MM-SS_<tool>/"""
    repo_root = Path(__file__).resolve().parent.parent.parent
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return str(repo_root / "profiling_results" / f"{ts}_{tool_name}")


def ensure_output_dir(path: str = None, tool_name: str = "profiling") -> str:
    """确保输出目录存在并返回路径. 若未指定则创建时间戳目录."""
    if path is None:
        path = _get_default_output_dir(tool_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_text(text: str, filepath: str):
    with open(filepath, "w") as f:
        f.write(text)


class ResultCollector:
    """收集 profiling 结果并在 close() 时统一持久化.

    用法:
        rc = ResultCollector("my_profiler", output_dir="/path/to/dir")
        rc.add_json("stage_breakdown", {...})
        rc.add_text("stage_breakdown", "文本报告内容...")
        rc.save_all()
    """

    def __init__(self, tool_name: str, output_dir: str = None):
        self.tool_name = tool_name
        self.output_dir = ensure_output_dir(output_dir, tool_name)
        self._json_payloads: Dict[str, dict] = {}
        self._text_payloads: Dict[str, str] = {}
        self._saved = False

    def add_json(self, name: str, data: dict):
        self._json_payloads[name] = data

    def add_text(self, name: str, text: str):
        self._text_payloads[name] = text

    def save_all(self):
        if self._saved:
            return
        for name, data in self._json_payloads.items():
            path = os.path.join(self.output_dir, f"{name}.json")
            save_json(data, path)
        for name, text in self._text_payloads.items():
            path = os.path.join(self.output_dir, f"{name}.txt")
            save_text(text, path)
        self._saved = True
        # 打印摘要
        files = sorted(Path(self.output_dir).iterdir())
        print(f"\n  [持久化] 结果已保存至: {self.output_dir}/")
        for f in files:
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    def close(self):
        self.save_all()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def add_output_dir_arg(parser, default: str = None):
    """为 argparse 添加统一的 --output_dir 参数."""
    parser.add_argument(
        "--output_dir", type=str, default=default,
        help="输出目录 (默认: profiling_results/<timestamp>_<tool>/ 自动创建)"
    )
