#!/usr/bin/env python3
"""rekep_policy_node：薄壳启动器（PyInstaller onefile）。

exe 用 BENCH_REKEP_PYTHON（ReKep 策略 venv）exec ReKep 策略节点的包入口
（entrypoint.py → rekep_policy.main）。dora 节点 env 块是替换语义，这里显式
构造：PYTHONPATH（含策略 src 与 forge_policy SDK）、默认机器人 profile、
CUDA 选择。ReKep 的算法参数（REKEP_IK / IK 目标 / 边界等）由 dataflow 的
节点 env 提供，本壳原样透传。

约定 env：
    BENCH_REKEP_PYTHON   策略 venv python（必填）
    BENCH_REKEP_SRC      策略节点源码目录（含 entrypoint.py / src/rekep_policy）
    REKEP_PROFILE        机器人 profile JSON（默认 <src>/rekep_profile.arx_x5_ee.json）
"""

import os
import sys


def main() -> int:
    python = os.environ.get("BENCH_REKEP_PYTHON")
    if not python:
        print(
            "rekep_policy_node: BENCH_REKEP_PYTHON 未设置（ReKep 策略 venv 的 python 路径）",
            file=sys.stderr,
        )
        return 2

    src = os.environ.get("BENCH_REKEP_SRC", ".")
    entry = os.path.join(src, "entrypoint.py")
    if not os.path.isfile(entry):
        print(f"rekep_policy_node: 找不到策略入口 {entry}", file=sys.stderr)
        return 2

    profile = os.environ.get(
        "REKEP_PROFILE", os.path.join(src, "rekep_profile.arx_x5_ee.json")
    )

    env = dict(os.environ)
    env.setdefault("REKEP_PROFILE", profile)
    # 策略包源码（rekep_policy.*）与同仓库 forge_policy SDK（三层上级的 policy/src）
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(src)))  # …/rekep_policy_dev
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            os.path.join(src, "src"),
            src,
            os.path.join(repo_root, "policy", "src"),
            env.get("PYTHONPATH", ""),
        )
        if p
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "6")

    argv = [python, "-u", entry, "--profile", profile, *sys.argv[1:]]
    os.execve(python, argv, env)
    return 1


if __name__ == "__main__":
    sys.exit(main())