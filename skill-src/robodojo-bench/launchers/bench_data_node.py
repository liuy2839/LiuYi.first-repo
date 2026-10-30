#!/usr/bin/env python3
"""bench_data_node：薄壳启动器（PyInstaller onefile）。

exe 用 BENCH_DATA_PYTHON（RoboDojo conda env）exec robodojo/main.py。
dora 的节点 env 块是**替换**语义，conda 环境的库路径不会自动带上；
这里显式构造 Isaac Sim 需要的环境（LD_LIBRARY_PATH / PYTHONPATH /
CUDA 设备可见性），与现役裸 dora 跑法的 env 对齐。
"""

import os
import sys


def main() -> int:
    python = os.environ.get("BENCH_DATA_PYTHON")
    if not python:
        print("bench_data_node: BENCH_DATA_PYTHON 未设置（RoboDojo env 的 python 路径）", file=sys.stderr)
        return 2
    src = os.environ.get("BENCH_DATA_SRC", ".")
    entry = os.path.join(src, "main.py")

    cfg = os.environ.get("BENCHMARK_CONFIG", "bench_config.yaml")
    if not os.path.isabs(cfg):
        here = os.path.dirname(os.path.abspath(sys.argv[0]))
        for base in (os.getcwd(), here, src):
            candidate = os.path.join(base, cfg)
            if os.path.isfile(candidate):
                cfg = candidate
                break

    dojo_root = os.environ.get("ROBODOJO_ROOT", "/data1/liuyi/robodojo_repos/RoboDojo")
    conda_env = os.path.dirname(os.path.dirname(python))  # .../envs/RoboDojo

    env = dict(os.environ)
    env["BENCHMARK_CONFIG"] = cfg
    env.setdefault("ROBODOJO_ROOT", dojo_root)
    # 与现役 dataflow.yaml 的 benchmark 节点 env 逐项对齐
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("PYTHONPATH", ""), f"{dojo_root}/XPolicyLab") if p
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p
        for p in (
            env.get("LD_LIBRARY_PATH", ""),
            f"{conda_env}/lib",
        )
        if p
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # isaaclab 的日志与资产缓存走 tempfile.gettempdir()（默认 /tmp）——
    # 共享机器上 /tmp/isaaclab 常被他人先建（属主不合 → PermissionError →
    # env 构造失败，两集全灭）。把 TMPDIR 指到自己的目录，一并避开。
    private_tmp = os.path.join(os.path.dirname(src), "tmp")
    os.makedirs(private_tmp, exist_ok=True)
    env["TMPDIR"] = private_tmp
    env["TEMP"] = private_tmp
    env["TMP"] = private_tmp

    argv = [python, "-u", entry, *sys.argv[1:]]
    os.execve(python, argv, env)
    return 1


if __name__ == "__main__":
    sys.exit(main())

