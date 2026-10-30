from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "policy" / "src"))

_ = __import__("types").ModuleType("dora")
_.Node = object
sys.modules.setdefault("dora", _)

from rekep_alg.planner import (  # noqa: E402
    ConstraintSecurityError,
    digest,
    load_functions,
    make_template_program,
    plan,
)


def _snapshot():
    return {
        "observation_id": "obs-1",
        "keypoints": [
            {"index": 0, "position_m": [0.4, 0.0, 0.05], "object": "red_cube", "kind": "center"},
            {"index": 1, "position_m": [0.3, 0.2, 0.02], "object": "green_zone", "kind": "center"},
        ],
        "objects": [
            {"name": "red_cube", "movable": True, "region": False, "position_m": [0.4, 0.0, 0.05]},
            {"name": "green_zone", "movable": False, "region": True, "position_m": [0.3, 0.2, 0.02]},
        ],
    }


def test_template_plan_structure_and_digest() -> None:
    snap = _snapshot()
    p = plan("put the red cube into the green zone", snap, source="template")
    assert p["num_stages"] == 3
    assert p["source"] == "template"
    assert p["grasp_keypoints"] == [0, -1, -1]
    assert p["release_keypoints"] == [-1, -1, 0]
    assert p["success"]["object"] == "red_cube"
    # digest 可复算
    body = {k: v for k, v in p.items() if k != "plan_digest"}
    assert digest(body) == p["plan_digest"]
    # 约束文本可被沙箱加载
    for s in p["stages"]:
        if s["subgoal_constraints"]:
            fns = load_functions(s["subgoal_constraints"])
            assert fns
    fns = load_functions(p["stages"][1]["path_constraints"], grasped_keypoints=[0])
    ee = np.array([0.4, 0.0, 0.05, 0, 0, 0, 1.0])
    kps = np.asarray([k["position_m"] for k in snap["keypoints"]])
    assert fns[0](ee, kps) == 0.0  # 已抓(keypoint 0) → grasping cost 0
    fns2 = load_functions(p["stages"][1]["path_constraints"], grasped_keypoints=[])
    assert fns2[0](ee, kps) == 1.0  # 未抓 → 1


def test_sandbox_rejects_escape() -> None:
    with pytest.raises(ConstraintSecurityError):
        load_functions("def f(ee, kps):\n    return __import__('os').system('id')\n")
    with pytest.raises(ConstraintSecurityError):
        load_functions("def f(ee, kps):\n    x = [i for i in range(3)]\n    return 0.0\n")
    with pytest.raises(ConstraintSecurityError):
        load_functions("def f(ee, kps):\n    return open('/etc/passwd').read()\n")


def test_executor_state_machine_without_solvers() -> None:
    """executor 的密度/夹爪辅助函数与终态语义（不触发真求解器）。"""
    from rekep_alg.executor import ReKepExecutor

    class NoIK:
        def solve(self, *a, **k):
            raise RuntimeError("not used in this test")

    ex = ReKepExecutor(NoIK())
    # 未 start_episode → done
    assert ex.step({"ee_pose": np.zeros(7), "keypoints": np.zeros((2, 3)),
                    "arm_q": np.zeros(7)}) == {"type": "done"}
    # gripper 语义映射
    assert ReKepExecutor._gripper_of(1.0) == "close"
    assert ReKepExecutor._gripper_of(-1.0) == "open"
    assert ReKepExecutor._gripper_of(0.0) == "hold"
    # grasp 失败 → 终止
    ex._plan = {"num_stages": 3}
    ex._stage = 1
    assert ex.on_grasp_result(ok=False) == {"type": "done"}


def test_executor_runs_stages_with_reference_input_shapes() -> None:
    from types import SimpleNamespace
    from rekep_alg.executor import ReKepExecutor

    calls = []

    class Subgoal:
        def solve(self, ee, keypoints, mask, goals, paths, sdf, collision, grasp, joints, **kwargs):
            assert keypoints.shape == (3, 3)
            assert mask.shape == (3,)
            np.testing.assert_array_equal(keypoints[0], ee[:3])
            assert collision.shape[0] > 0
            for fn in goals + paths:
                assert np.isfinite(fn(ee[:3], keypoints[1:]))
            calls.append("subgoal")
            return ee.copy(), {}

    class Path:
        def solve(self, ee, goal, keypoints, mask, constraints, *args, **kwargs):
            assert keypoints.shape[0] == len(mask)
            calls.append("path")
            return goal.reshape(1, 7), {}

    ex = ReKepExecutor(
        object(), solver_factory=lambda *_: (Subgoal(), Path()),
        transform_utils=SimpleNamespace(convert_pose_quat2mat=lambda pose: np.eye(4)),
    )
    ex._densify = lambda ee, path: [np.concatenate([pose, [0.0]]) for pose in path]
    snapshot = _snapshot()
    ex.start_episode(make_template_program(snapshot, "pick and place"), snapshot)
    world = {
        "ee_pose": np.array([0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        "arm_q": np.zeros(6),
        "keypoints": np.asarray([k["position_m"] for k in snapshot["keypoints"]]),
    }

    assert ex.step(world)["type"] == "cartesian"
    assert ex.step(world)["type"] == "close_gripper"
    ex.on_grasp_result(True)
    assert ex.step(world)["type"] == "cartesian"
    assert ex.step(world)["type"] == "cartesian"
    assert ex.step(world)["type"] == "open_gripper"
    ex.on_release_done()
    assert ex.step(world)["type"] == "done"
    assert calls == ["subgoal", "path"] * 3


def test_lift_template_when_no_region() -> None:
    snap = {
        "observation_id": "obs-lift",
        "keypoints": [
            {"index": 0, "position_m": [0.1, -0.1, 0.80], "object": "bottle_0_1", "kind": "center"},
            {"index": 1, "position_m": [0.2, 0.0, 0.80], "object": "cup_0_2", "kind": "center"},
        ],
        "objects": [
            {"name": "bottle_0_1", "movable": True, "region": False, "position_m": [0.1, -0.1, 0.80]},
            {"name": "cup_0_2", "movable": True, "region": False, "position_m": [0.2, 0.0, 0.80]},
        ],
    }
    p = plan("Pick up the bottle by 10 cm.", snap, source="template")
    assert p["num_stages"] == 2
    assert p["success"]["mode"] == "lift"
    assert p["success"]["object"] == "bottle_0_1"      # 指令词匹配
    assert p["grasp_keypoints"] == [0, -1]
    assert p["release_keypoints"] == [-1, -1]
    # 抬升约束：目标 z0+0.1 之前代价 > 0，之后为 0
    fns = load_functions(p["stages"][1]["subgoal_constraints"])
    low = np.array([[0.1, -0.1, 0.80]])
    high = np.array([[0.1, -0.1, 0.90]])
    ee = np.zeros(3)
    lift_fn = fns[0]  # 抬升约束（第二个约束是 ee 靠近目标，另行验证）
    assert lift_fn(ee, low) > 0
    assert lift_fn(ee, high) <= 0
