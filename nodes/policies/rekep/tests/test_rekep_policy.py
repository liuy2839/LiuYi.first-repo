from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "policy" / "src"))

_ = __import__("types").ModuleType("dora")
_.Node = object
sys.modules.setdefault("dora", _)

from rekep_policy import ReKepPolicy, RobotProfile  # noqa: E402
from forge_policy import PolicyError  # noqa: E402


def make_profile(**kw) -> RobotProfile:
    defaults = dict(
        action_names=("j1", "j2", "j3", "grip", "other.0"),
        arm_action_names=("j1", "j2", "j3"),
        gripper_action_name="grip",
        gripper_open_value=0.0,
        gripper_close_value=1.0,
        ik=_MapIK(),
    )
    defaults.update(kw)
    return RobotProfile(**defaults)


class _MapIK:
    def solve(self, target_pos, target_quat_xyzw, seed_q):  # noqa: ANN001
        return np.asarray(target_pos) / 10.0  # 3 目标映射进 3 臂关节


class _FixedPerception:
    def keypoints(self, obs):  # noqa: ANN001
        return {"observation_id": "obs-1", "keypoints": []}


class _ScriptPlanner:
    def plan(self, instruction, snapshot):  # noqa: ANN001
        assert isinstance(instruction, str) and instruction
        return {"plan_id": "p1", "instruction": instruction}


class _ScriptExecutor:
    def __init__(self, intents):  # noqa: ANN001
        self.intents = list(intents)
        self.started = None

    def start_episode(self, plan, snapshot):  # noqa: ANN001
        self.started = (plan, snapshot)

    def step(self, obs):  # noqa: ANN001
        if not self.intents:
            return {"type": "done"}
        return self.intents.pop(0)


def _obs(state=(0.1, -0.2, 0.3, 0.4, 0.5, 1.0)):
    return {
        "state": np.asarray(state, dtype=np.float64),
        "state_names": ["j1", "j2", "j3", "grip", "other.0"],
        "images": {},
        "depths": {},
        "segs": {},
    }


def test_profile_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        make_profile(action_names=("j1", "j2", "j3", "grip"), arm_action_names=("j1", "jX"))


def test_compose_action_holds_uncontrolled() -> None:
    profile = make_profile()
    out = profile.compose_joint_action(
        np.array([0.05, -0.02, 0.07]), "close", np.array([9.0, 9.0, 9.0, 9.0, 0.25]),
        ["j1", "j2", "j3", "grip", "other.0"],
    )
    assert out[0] == pytest.approx(0.05)
    assert out[3] == pytest.approx(1.0)  # gripper close
    assert out[4] == pytest.approx(0.25)  # hold 未控制分量
    # 关 gripper 时
    out2 = profile.compose_joint_action(np.zeros(3), "open", np.zeros(5),
                                  ["j1", "j2", "j3", "grip", "other.0"])
    assert out2[3] == pytest.approx(0.0)


def test_replan_on_new_instruction() -> None:
    intents = [{"type": "cartesian", "pose7": [0.3, 0, 0.4, 0, 0, 0, 1], "gripper": "open"}]
    intents += [dict(intents[0])]  # 第二次规划后也要有一个 cartesian 意图可执行
    executor = _ScriptExecutor(intents)
    policy = ReKepPolicy(make_profile(), _FixedPerception(), _ScriptPlanner(), executor)
    policy.set_instruction("pick cube")
    a1 = policy.step(_obs())
    assert a1 is not None and a1.size == 5
    policy.set_instruction("pick the blue one")  # 新指令 → 重规划
    a2 = policy.step(_obs())
    assert a2 is not None
    assert executor.started[0]["instruction"] == "pick the blue one"


def test_done_then_hold() -> None:
    executor = _ScriptExecutor([])
    policy = ReKepPolicy(make_profile(), _FixedPerception(), _ScriptPlanner(), executor)
    policy.set_instruction("x")
    first = policy.step(_obs())  # executor 立即 done → 转保持位姿并继续供动作
    assert first is not None and first.size == 5
    again = policy.step(_obs())
    assert again is not None and again.size == 5  # 结束后保持 1:1，不再规划


def test_reset_replans() -> None:
    executor = _ScriptExecutor([])
    policy = ReKepPolicy(make_profile(), _FixedPerception(), _ScriptPlanner(), executor)
    policy.set_instruction("x")
    policy.step(_obs())
    policy.reset_episode()
    assert policy.step(_obs()) is not None  # reset 后重新感知规划（随后保持位姿）


def _hold_exec():
    from types import SimpleNamespace

    return SimpleNamespace(
        start_episode=lambda plan, snapshot: None,
        step=lambda world: {"type": "failed", "error": "solver boom"},
    )


def test_failed_intent_keeps_1to1_by_holding() -> None:
    policy = ReKepPolicy(make_profile(), _FixedPerception(), _ScriptPlanner(), _hold_exec())
    policy.set_instruction("x")
    action = policy.step(_obs())
    assert action is not None and action.size == 5  # 失败后转为保持位姿，仍每帧给动作
    assert policy.step(_obs()) is not None


def test_no_instruction_raises() -> None:
    policy = ReKepPolicy(make_profile(), _FixedPerception(), _ScriptPlanner(), _ScriptExecutor([]))
    with pytest.raises(PolicyError) as ei:
        policy.step(_obs())
    assert ei.value.code == "NO_INSTRUCTION"


def test_arm_limits_clip() -> None:
    profile = make_profile(
        arm_lower=(-0.001, -0.001, -0.001), arm_upper=(0.001, 0.001, 0.001)
    )
    out = profile.compose_joint_action(np.array([5.0, -5.0, 5.0]), "open", np.zeros(5),
                                 ["j1", "j2", "j3", "grip", "other.0"])
    assert out[0] == pytest.approx(0.001)
    assert out[1] == pytest.approx(-0.001)


def test_invalid_pose_is_dropped_to_hold() -> None:
    from types import SimpleNamespace

    bad = np.array([np.nan, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
    ex = SimpleNamespace(
        start_episode=lambda plan, snapshot: None,
        step=lambda world: {"type": "cartesian", "pose7": bad.tolist(), "gripper": "hold"},
    )
    policy = ReKepPolicy(make_profile(), _FixedPerception(), _ScriptPlanner(), ex)
    policy.set_instruction("x")
    action = policy.step(_obs())
    assert action is not None and action.size == 5        # 击落非法位姿，改发保持动作
    assert np.all(np.isfinite(action))
