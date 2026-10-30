#!/usr/bin/env python3
"""bench_tool_node：薄壳启动器（PyInstaller onefile）。

exe 唯一的事：用 BENCH_TOOL_PYTHON（py3.12 venv）exec bench_tool 入口，
其余环境原样继承。Isaac Sim / forge_tool 的真实依赖都在各自 venv 里，
onefile 壳不打包它们。
"""

import os
import sys


def main() -> int:
    python = os.environ.get("BENCH_TOOL_PYTHON")
    if not python:
        print("bench_tool_node: BENCH_TOOL_PYTHON 未设置（py3.12 venv 的 python 路径）", file=sys.stderr)
        return 2
    src = os.environ.get("BENCH_TOOL_SRC", ".")
    # bench_tool 是包：必须 -m 方式跑（main.py 里的相对 import 才成立），
    # 包根（src）进 PYTHONPATH。
    argv = [python, "-u", "-m", "bench_tool.main", *sys.argv[1:]]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src, env.get("PYTHONPATH", "")) if p
    )
    os.execve(python, argv, env)  # 不返回
    return 1


if __name__ == "__main__":
    sys.exit(main())
