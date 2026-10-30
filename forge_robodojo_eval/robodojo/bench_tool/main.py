"""bench_tool 工具面入口：py3.12 venv 里跑的 dora 节点。

用法（dataflow 里作为独立节点）：
    path: <BENCH_TOOL_VENV>/bin/python
    args: -u -m bench_tool.main

或直接：python -m bench_tool.main
"""

from __future__ import annotations

import os
import sys


def _log(msg: str) -> None:
    print(f"[robodojo_bench_tool] {msg}", flush=True)


def main() -> int:
    from .node import run_bench_tool

    try:
        return run_bench_tool(log=_log)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
