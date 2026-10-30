"""真实算法装配：把 rekep_alg 三个模块接进 ReKepPolicy 的协议接口。

装配点（与 core.py 的 Protocols 对应）：
- PerceptionModule : SceneObjectPerception（场景物体真值 → center 关键点）
- PlannerModule    : rekep_alg.planner.plan（模板或 VLM，AST 沙箱 + digest）
- ExecutorModule   : rekep_alg.executor.ReKepExecutor（步进状态机 + reference 求解器）
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .perception_adapters import SceneObjectPerception
from .profile import RobotProfile


def _env_json(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    return json.loads(raw)


def build_ik(profile: RobotProfile):
    """按 REKEP_IK 选择求解器实现（仅求解器内部用；ee 动作空间不依赖它出动作）。"""
    kind = os.environ.get("REKEP_IK", "stub").lower()
    if kind == "x5":
        from .ik_adapters import X5IK

        urdf = os.environ.get("REKEP_X5_URDF")
        if not urdf:
            raise RuntimeError("REKEP_IK=x5 需要 REKEP_X5_URDF")
        return X5IK(
            urdf,
            joint_names=tuple(profile.arm_action_names) if profile.action_space == "joint"
            else ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
            ee_body=os.environ.get("REKEP_X5_EE_BODY", "link6"),
            tool_offset=tuple(_env_json("REKEP_X5_TOOL_OFFSET", [0.0, 0.0, 0.0])),
            base_pos=tuple(_env_json("REKEP_X5_BASE_POS", [0.0, 0.0, 0.0])),
            base_quat_wxyz=tuple(_env_json("REKEP_X5_BASE_QUAT_WXYZ", [1, 0, 0, 0])),
        )
    from .ik_adapters import StubIK

    return StubIK(dof=len(profile.arm_action_names) or 6)


class PlannerAdapter:
    def __init__(self, source: str | None = None) -> None:
        self.source = source or os.environ.get("REKEP_PLAN_SOURCE", "template")

    def plan(self, instruction: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        from rekep_alg.planner import plan as alg_plan

        return alg_plan(instruction, snapshot, source=self.source)


class ExecutorAdapter:
    """包装 ReKepExecutor，注入求解器用的 IK 与工作空间边界。"""

    def __init__(self, profile: RobotProfile, log: Any = None) -> None:
        from rekep_alg.executor import ExecutorConfig, IKAdapter, ReKepExecutor

        self._log = log or (lambda _m: None)
        ik_impl = build_ik(profile)
        self._ik = ik_impl
        reset_q = _env_json("REKEP_RESET_JOINT_POS", None)
        bounds_min = _env_json("REKEP_BOUNDS_MIN", [-0.8, -1.0, 0.2])
        bounds_max = _env_json("REKEP_BOUNDS_MAX", [0.8, 0.6, 1.4])
        config = ExecutorConfig(
            subgoal_sampling_maxfun=int(os.environ.get("REKEP_SUBGOAL_MAXFUN", "200")),
            path_sampling_maxfun=int(os.environ.get("REKEP_PATH_MAXFUN", "200")),
            minimizer_maxiter=int(os.environ.get("REKEP_MINIMIZER_MAXITER", "50")),
            obstacle_radius=float(os.environ.get("REKEP_OBSTACLE_RADIUS", "0.06")),
        )
        self._impl = ReKepExecutor(
            IKAdapter(ik_impl.solve_ik_result),
            config=config,
            reset_joint_pos=np.asarray(reset_q, dtype=float) if reset_q else None,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )

    def start_episode(self, plan: dict[str, Any], snapshot: dict[str, Any]) -> None:
        self._impl.start_episode(plan, snapshot)

    def step(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return self._impl.step(world_state)

    def on_grasp_result(self, ok: bool) -> dict[str, Any]:
        return self._impl.on_grasp_result(ok)

    def on_release_done(self) -> dict[str, Any]:
        return self._impl.on_release_done()


def build_alg_modules(profile: RobotProfile, log: Any = None) -> dict[str, Any]:
    # 动作位置裁剪边界（与求解器工作空间保持一致）
    profile.bounds_min = tuple(_env_json("REKEP_BOUNDS_MIN", [-0.8, -1.0, 0.2]))
    profile.bounds_max = tuple(_env_json("REKEP_BOUNDS_MAX", [0.8, 0.6, 1.4]))
    kind = os.environ.get("REKEP_PERCEPTION", "scene_objects").lower()
    if kind == "dino":
        from .perception_adapters import DepthDinoPerception

        perception = DepthDinoPerception(
            image_topic=os.environ.get("REKEP_IMAGE_TOPIC", "image/head"),
            depth_topic=os.environ.get("REKEP_DEPTH_TOPIC", "depth/head"),
            ee_pose_prefix=os.environ.get("REKEP_EE_POSE_PREFIX", "left_ee_pose"),
            arm_prefix=os.environ.get("REKEP_ARM_PREFIX", "left_arm_joint_state"),
        )
    else:
        perception = SceneObjectPerception(
            ee_pose_prefix=os.environ.get("REKEP_EE_POSE_PREFIX", "left_ee_pose"),
            arm_prefix=os.environ.get("REKEP_ARM_PREFIX", "left_arm_joint_state"),
        )
    return {
        "perception": perception,
        "planner": PlannerAdapter(),
        "executor": ExecutorAdapter(profile, log=log),
    }
