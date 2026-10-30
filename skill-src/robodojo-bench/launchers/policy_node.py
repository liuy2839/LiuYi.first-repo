#!/usr/bin/env python3
"""policy_node：薄壳启动器（PyInstaller onefile）。

exe 用 BENCH_POLICY_PYTHON（g05_eval env）exec robodojo/policy_node.py。
dora 节点 env 块是替换语义：G05 的全部路径（权重/处理器/tokenizer/
PYTHONPATH/CUDA）在这里显式构造，与现役裸 dora 跑法的 policy 节点 env
逐项对齐。
"""

import os
import sys


def main() -> int:
    python = os.environ.get("BENCH_POLICY_PYTHON")
    if not python:
        print("policy_node: BENCH_POLICY_PYTHON 未设置（策略 env 的 python 路径）", file=sys.stderr)
        return 2
    src = os.environ.get("BENCH_DATA_SRC", ".")
    entry = os.path.join(src, "policy_node.py")

    dojo_root = os.environ.get("ROBODOJO_ROOT", "/data0/liuyi/robodojo/RoboDojo")
    g05_root = f"{dojo_root}/XPolicyLab/policy/G05/G05"

    env = dict(os.environ)
    env.setdefault("XPOLICYLAB_ROOT", f"{dojo_root}/XPolicyLab")
    env.setdefault("POLICY_NAME", "G05")
    env.setdefault("ENV_CFG_TYPE", "arx_x5")
    env.setdefault("ACTION_TYPE", "joint")
    env.setdefault("CUDA_VISIBLE_DEVICES", "6")
    env.setdefault("G05_ROOT", g05_root)
    env.setdefault(
        "G05_CKPT_PATH",
        "/data0/liuyi/g05_models/sim/hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint",
    )
    env.setdefault(
        "G05_PROCESSOR_PATH",
        "/data0/liuyi/g05_models/base_assets/g05_robodojo_train_base_assets/pretrained/hf_processor",
    )
    env.setdefault(
        "G05_HF_PROCESSOR_PATH",
        env["G05_PROCESSOR_PATH"],
    )
    env.setdefault(
        "G05_ACTION_TOKENIZER_PATH",
        "/data0/liuyi/g05_models/sim/hf_g05_robodojo_fm_only_checkpoint/action_tokenizer.pt",
    )
    env.setdefault("ROBODOJO_G05_ACTION_SOURCE", "fm")
    env.setdefault(
        "ROBODOJO_LEROBOT_V30_ROOT",
        "/data0/liuyi/g05_models/sim/hf_g05_robodojo_fm_only_checkpoint",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            env.get("PYTHONPATH", ""),
            f"{g05_root}/src",
            g05_root,
        )
        if p
    )

    argv = [python, "-u", entry, *sys.argv[1:]]
    os.execve(python, argv, env)
    return 1


if __name__ == "__main__":
    sys.exit(main())
