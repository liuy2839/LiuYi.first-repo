#!/usr/bin/env python3
"""RoboDojo 的 forge policy 节点。

它在 dora 这一侧扮演 XPolicyLab 里 `model_client` 的角色，把观测喂给
`XPolicyLab.policy.<POLICY>.model.Model`，再把动作 chunk 逐个吐回 benchmark 节点。

**必须逐字复刻官方 rollout 的节奏**（XPolicyLab/policy/demo_policy/deploy.py）：

    model.reset()
    while not is_episode_end():
        model.update_obs(obs); actions = model.get_action()      # 一次一个 chunk
        for i, a in enumerate(actions):
            take_action(a)
            if is_episode_end() or i+1 == len(actions): break
            model.update_obs(next_obs)                            # chunk 内也每步喂

三条不能偷懒的地方：

1. **观测驱动，不是 tick 驱动。** benchmark 节点是严格 1:1 锁步，窗口外到达的
   动作会被直接丢弃。按 tick 无条件推理会发得远比消费快 —— RoboCasa 那轮实测
   丢掉 59% 的动作，把 chunk 抽稀成 41%，成绩腰斩。tick 在这里只当心跳。

2. **chunk 执行期间每一步都要 update_obs。** 上面内层那句不是冗余：有策略吃
   历史帧（Hy-Embodied 用 6 帧、每 20 步采一次）。只在需要动作时喂会静默劣化。

3. **chunk 最后一个动作要打 chunk_end 标记。** RoboDojo 的 `is_episode_end()`
   有副作用（会跑 `_final_check()`），官方每个 chunk 查 len+1 次，我们 1:1 天然
   只查 len 次。标记让 benchmark 侧补上那一次。详见 adapters/robodojo_adapter.py。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from dora import Node
from adapters.messages import Image, JointCommand, JointState, PolicyCommand

# chunk 边界标记塞在 effort 里 —— mode 是 Literal 标量塞不下自定义值，
# name/position 是契约本身不能动。详见 _core/adapter.py。
CHUNK_END_SENTINEL = 1.0

# 状态/动作的分量规格。**维度绝对不能写死** —— ARX X5 是 6 DOF 手臂
# (arm_dim=[6,6], ee_dim=[1,1]，总 14)，不是想当然的 7。一律走官方的
# get_robot_action_dim_info(env_cfg_type)，和 benchmark 侧同源。
# 两边算出来必须完全一致：不一致时 benchmark 的 action_component_names 校验
# 会直接把这一集判失败（这正是我们要的：宁可炸掉，也不要静默跑歪）。
def _build_obs_spec(env_cfg_type: str) -> tuple[tuple[str, int], ...]:
    """观测 state 的完整规格（含 ee_pose，28 维），按键名字典序 —— 必须与
    benchmark 侧 build_obs_spec() 完全一致。"""
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)))
        spec.append((f"{p}ee_joint_state", int(ed)))
        spec.append((f"{p}ee_pose", 7))
    return tuple(sorted(spec, key=lambda kv: kv[0]))


def _build_state_spec(env_cfg_type: str, action_type: str) -> tuple[tuple[str, int], ...]:
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)) if action_type == "joint" else (f"{p}ee_pose", 7))
        spec.append((f"{p}ee_joint_state", int(ed)))
    return tuple(spec)

CHANNEL_CAM = {
    "image/head": "cam_head",
    "image/left_wrist": "cam_left_wrist",
    "image/right_wrist": "cam_right_wrist",
}
OBS_PARTS = {"proprio_state", *CHANNEL_CAM.keys()}


def log(msg: str) -> None:
    print(f"[robodojo_policy] {msg}", flush=True)


def _build_model() -> Any:
    """按 XPolicyLab 的约定加载策略适配器。

    注意 CWD：官方 setup_eval_* 脚本要求 CWD 是 policy 目录（策略里有相对路径）。
    """
    xpl_root = os.environ.get("XPOLICYLAB_ROOT")
    if not xpl_root or not os.path.isdir(xpl_root):
        raise RuntimeError("必须设置 XPOLICYLAB_ROOT 指向 XPolicyLab 检出目录")
    parent = os.path.dirname(os.path.abspath(xpl_root))
    for p in (parent, xpl_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    policy = os.environ.get("POLICY_NAME", "G05")
    import importlib

    from utils.load_file import load_yaml  # type: ignore

    deploy = load_yaml(os.path.join(xpl_root, "policy", policy, "deploy.yml")) or {}
    # **不能用 setdefault**：deploy.yml 里这些键是存在的、值为 null（demo_policy
    # 的模板就长这样），setdefault 不覆盖 None，结果 Model 拿到 env_cfg_type=None
    # 去找 `env_cfg/None.yml`。必须显式判空。
    for _k, _v in (
        ("env_cfg_type", os.environ.get("ENV_CFG_TYPE", "arx_x5")),
        ("action_type", os.environ.get("ACTION_TYPE", "joint")),
        ("bench_name", "RoboDojo"),
        ("task_name", os.environ.get("TASK_NAME")),
        ("seed", 0),
    ):
        if deploy.get(_k) is None and _v is not None:
            deploy[_k] = _v
    ckpt = os.environ.get("CKPT_NAME")
    if ckpt:
        deploy["ckpt_name"] = ckpt

    # G05 resolves weights from G05_CKPT_PATH inside its model adapter.  The
    # processor path is likewise deployment-specific; when supplied, replace
    # any stale path from an upstream deploy.yml with the portable override.
    if policy == "G05":
        processor_path = os.environ.get("G05_PROCESSOR_PATH")
        if not processor_path:
            raise RuntimeError(
                "G05 requires G05_PROCESSOR_PATH; do not rely on a machine-specific "
                "processor path from the upstream deploy.yml"
            )
        overrides = [
            str(item)
            for item in (deploy.get("hydra_overrides") or [])
            if not str(item).startswith(
                ("model.model_arch.hf_processor_path=", "model.processor.hf_processor_path=")
            )
        ]
        overrides.append(f"model.model_arch.hf_processor_path={processor_path}")
        deploy["hydra_overrides"] = overrides

    os.chdir(os.path.join(xpl_root, "policy", policy))
    mod = importlib.import_module(f"XPolicyLab.policy.{policy}.model")
    log(f"加载策略 {policy}（action_type={deploy['action_type']}, env_cfg={deploy['env_cfg_type']}）")
    return mod.Model(deploy)


def _flatten_action(action: dict[str, Any], spec) -> np.ndarray:
    parts = []
    for key, dim in spec:
        v = action.get(key)
        if v is None:
            raise KeyError(f"策略动作缺少 {key}；实得键={sorted(action)}")
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"{key} 维度应为 {dim}，收到 {arr.size}")
        parts.append(arr)
    return np.concatenate(parts).astype(np.float32)


def main() -> int:
    node = Node()
    model = _build_model()
    env_cfg_type = os.environ.get("ENV_CFG_TYPE", "arx_x5")
    action_type = os.environ.get("ACTION_TYPE", "joint")
    state_spec = _build_state_spec(env_cfg_type, action_type)   # 动作 14 维
    obs_spec = _build_obs_spec(env_cfg_type)                    # 观测 28 维
    component_names = [f"{k}.{i}" for k, d in state_spec for i in range(d)]
    log(f"动作 {len(component_names)} 维 / 观测 {sum(d for _, d in obs_spec)} 维")

    state: np.ndarray | None = None
    images: dict[str, np.ndarray] = {}
    instruction: str = ""
    fresh: set[str] = set()
    chunk: list[np.ndarray] = []
    chunk_pos = 0
    phase = "idle"
    sent = 0

    def send_action(vec: np.ndarray, chunk_end: bool) -> None:
        nonlocal sent
        cmd = JointCommand(
            name=list(component_names),
            position=[float(x) for x in vec],
            mode="position",
            # chunk 最后一个动作打标记，让 benchmark 侧补那次 is_episode_end()
            # forge_msgs 要求 effort 要么为空、要么与 name 等长，所以填满。
            effort=[CHUNK_END_SENTINEL] * len(component_names) if chunk_end else [],
        )
        node.send_output("action", cmd.to_arrow())
        sent += 1

    def build_obs() -> dict[str, Any]:
        """拼回 XPolicyLab 的标准观测格式。图像已是 RGB，**不做任何通道转换**。"""
        off = 0
        st: dict[str, Any] = {}
        for key, dim in obs_spec:
            seg = state[off : off + dim]
            # 还原 env.get_obs() 的原生类型：夹爪是 Python list，其余是 float32
            # ndarray。传输走 float64 不丢精度，但**类型必须还原**——
            # _encode_observation 把关节转回 float32，ee_pose 却按 float64 用，
            # 多出来的精度会直接走进位姿还原的数学里。
            if key.endswith("ee_joint_state"):
                st[key] = [float(x) for x in seg]
            else:
                st[key] = np.asarray(seg, dtype=np.float32)
            off += dim
        return {
            "instruction": instruction,
            "vision": {cam: {"color": img} for cam, img in images.items()},
            "state": st,
        }

    def on_observation() -> bool:
        """收齐一份观测后：先无条件 update_obs，再决定要不要新推一个 chunk。"""
        nonlocal chunk, chunk_pos
        obs = build_obs()
        model.update_obs(obs)                  # 坑 2：每步都喂，不管要不要动作
        if chunk_pos >= len(chunk):
            actions = model.get_action()
            if not actions:
                log("策略返回空 chunk，跳过这一帧")
                return False
            chunk = [_flatten_action(a, state_spec) for a in actions]
            chunk_pos = 0
        send_action(chunk[chunk_pos], chunk_end=(chunk_pos + 1 == len(chunk)))
        chunk_pos += 1
        return True

    log("就绪，等 benchmark 节点的观测")
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        eid = event["id"]

        # **`set_instruction` / `reset_scene` 不是独立的 dora 通道。**
        # `_core/runner.py:155-156` 是用 `policy_command(...)` 发的，也就是
        # `policy_command` 通道上的 `PolicyCommand` 消息，`command` 字段区分类型、
        # payload 在 `inputs_json` 里。按独立通道去收**永远收不到，而且不报错**：
        # 指令一直是空串、`model.reset()` 一次都不会调。逐行 diff 才把它揪出来
        # （指令 1050 帧全不一致 + reset 记录 0 次 vs 官方 2 次）。
        if eid == "policy_command":
            cmd = PolicyCommand.from_arrow(event["value"])
            if cmd.command == "start":
                phase = "running"
                log("收到 start")
            elif cmd.command in ("stop", "cancel"):
                log(f"收到 {cmd.command}，退出")
                break
            elif cmd.command == "set_instruction":
                import json as _json

                payload = _json.loads(cmd.inputs_json or "{}")
                instruction = str(payload.get("instruction", ""))
                log(f"指令：{instruction}")
            elif cmd.command == "reset_scene":
                # 新 episode：清模型内部状态 + 清本地缓存。
                # 坑：reset_scene 与观测走**不同的 dora 通道，跨通道到达顺序没有
                # 保证**，所以这里清掉的可能是已经先到的观测。benchmark 侧的看门狗
                # 会重发，重发是幂等的（见 _core/runner.republish_if_stalled）。
                # 官方每集调两次 reset（env.reset() 里一次 + eval_one_episode 开头
                # 一次），对有内部状态的策略这个次数是有意义的，所以照抄。
                model.reset()
                model.reset()
                state, images, fresh = None, {}, set()
                chunk, chunk_pos = [], 0
            continue

        if eid == "proprio_state":
            state = np.asarray(JointState.from_arrow(event["value"]).position, dtype=np.float64)
            fresh.add(eid)
        elif eid in CHANNEL_CAM:
            # 必须用官方的 to_numpy()：`Image.data` 是 **bytes**，
            # `np.asarray(bytes, dtype=np.uint8)` 会把整个 bytes 当成一个标量去
            # int() → `ValueError: invalid literal for int() with base 10: b'\x84{h}...'`。
            # to_numpy() 还顺带处理 encoding/step，不用我们自己拼形状。
            images[CHANNEL_CAM[eid]] = np.ascontiguousarray(
                Image.from_arrow(event["value"]).to_numpy(), dtype=np.uint8
            )
            fresh.add(eid)
        else:
            continue        # tick 之类：只当心跳，绝不在这里推理（坑 1）

        if phase == "running" and state is not None and fresh >= OBS_PARTS:
            if on_observation():
                fresh.clear()

    log(f"退出，共发出 {sent} 个动作")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
