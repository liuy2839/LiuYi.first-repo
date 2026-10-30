"""ReKep 策略核心：把感知→规划→执行的算法接到统一策略协议上。

一个 episode 内的时序（同一进程内顺序调用，无并发）：
    首帧：perception.keypoints(obs) → planner.plan(...) → executor.start_episode(...)
    每帧：perception.world_state(obs) → executor.step(world) → 动作意图
          → joint 模式走 IK，ee 模式直接发位姿 → compose 成评测端命名动作

算法实现由 main.py 装配注入（rekep_alg 或 fake），本模块只依赖 Protocols。
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

from forge_policy import Policy, PolicyError

from .profile import RobotProfile


class PerceptionModule(Protocol):
    def keypoints(self, obs: dict[str, Any]) -> dict[str, Any]:
        """一帧观测 → perception snapshot（keypoints/objects/observation_id）。"""
        ...


class PlannerModule(Protocol):
    def plan(self, instruction: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        """指令 + snapshot → digest-bound ReKep 计划（stages/constraints）。"""
        ...


class ExecutorModule(Protocol):
    def start_episode(self, plan: dict[str, Any], snapshot: dict[str, Any]) -> None: ...

    def step(self, world_state: dict[str, Any]) -> dict[str, Any]:
        """推进执行，返回动作意图：
        {"type": "cartesian", "pose7": [x,y,z,qx,qy,qz,qw], "gripper": "open"|"close"|"hold"}
        {"type": "close_gripper"} / {"type": "open_gripper"}
        {"type": "done"} / {"type": "waiting"}
        """
        ...


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q[:4])
    return np.array([w, x, y, z], dtype=np.float64)


class ReKepPolicy(Policy):
    """感知/规划在 episode 首帧触发一次；其后每帧执行推进一步。"""

    def __init__(
        self,
        profile: RobotProfile,
        perception: PerceptionModule,
        planner: PlannerModule,
        executor: ExecutorModule,
        *,
        log: Any = None,
    ) -> None:
        self.profile = profile
        self.perception = perception
        self.planner = planner
        self.executor = executor
        self._log = log or (lambda _m: None)
        self._instruction = ""
        self._snapshot: dict[str, Any] | None = None
        self._plan: dict[str, Any] | None = None
        self._last_world: dict[str, Any] | None = None
        self._episode_frames = 0
        self._finished = False
        self._hold = False
        self._bad_pose_logged = False

    def reset_episode(self) -> None:
        self._snapshot = None
        self._plan = None
        self._last_world = None
        self._episode_frames = 0
        self._finished = False
        self._hold = False
        self._bad_pose_logged = False

    def set_instruction(self, text: str) -> None:
        self._instruction = text
        # 新指令使旧计划失效；下一次 step 重新感知+规划
        self._plan = None
        self._snapshot = None

    # ----------------------------------------------------------------- step
    def step(self, obs: dict[str, Any]) -> np.ndarray | None:
        if self._finished and not self._hold:
            return None
        state = np.asarray(obs.get("state", []), dtype=np.float64)
        names = list(obs.get("state_names", []))
        if state.size == 0 or not names:
            return None
        self._episode_frames += 1

        if self._plan is None:
            self._snapshot = self.perception.keypoints(obs)
            if not self._instruction:
                raise PolicyError("NO_INSTRUCTION", "评测端未下发任务指令")
            self._plan = self.planner.plan(self._instruction, self._snapshot)
            self.executor.start_episode(self._plan, self._snapshot)
            self._log(
                f"episode 规划完成：plan={self._plan.get('plan_id')} "
                f"stages={self._plan.get('num_stages')} source={self._plan.get('source')}"
            )
            objs = self._snapshot.get("objects") or []
            self._log(
                "snapshot objects: "
                + ", ".join(f"{o['name']}({'M' if o.get('movable') else 'R'})" for o in objs)
            )
            if os.environ.get("REKEP_SAFE_HOLD") == "1":
                # 诊断开关：只规划不动作，用于区分"策略运动导致崩溃"与"场景本身崩溃"
                self._hold = True
                self._log("REKEP_SAFE_HOLD=1：本集只保持位姿，不执行计划")

        world_fn = getattr(self.perception, "world_state", None)
        if callable(world_fn):
            world = world_fn(obs, self._snapshot)
        else:
            world = self._default_world(obs)
        self._last_world = world
        if self._hold:
            return self._hold_action(world, state, names)
        intent = self.executor.step(world)
        return self._to_action(intent, world, state, names)

    def _sanitize_pose(self, pose7: np.ndarray) -> np.ndarray | None:
        """位姿合法性：有限值、四元数归一化、工作空间裁剪。非法则返回 None。"""
        p = np.asarray(pose7, dtype=float).reshape(-1)
        if p.size != 7 or not np.all(np.isfinite(p)):
            return None
        quat = p[3:]
        n = float(np.linalg.norm(quat))
        if n < 1e-6:
            return None
        p = p.copy()
        p[3:] = quat / n
        if self.profile.bounds_min is not None and self.profile.bounds_max is not None:
            p[:3] = np.clip(p[:3], np.asarray(self.profile.bounds_min),
                            np.asarray(self.profile.bounds_max))
        return p

    def _hold_action(self, world, state, names) -> np.ndarray | None:
        """episode 已结束/失败后保持位姿：继续 1:1 供动作，让评测端正常走到步数上限。"""
        pose7 = np.asarray(world["ee_pose"], dtype=float)
        if self.profile.action_space == "ee":
            pose_wxyz = np.concatenate([pose7[:3], _quat_xyzw_to_wxyz(pose7[3:])])
            return self.profile.compose_ee_action(pose_wxyz, "hold", state, names)
        seed = np.array(
            [dict(zip(names, state, strict=False)).get(n, 0.0)
             for n in self.profile.arm_action_names], dtype=np.float64)
        return self.profile.compose_joint_action(seed, "hold", state, names)

    def _default_world(self, obs: dict[str, Any]) -> dict[str, Any]:
        """感知模块未提供 world_state 时的兜底（主要供测试/假组件使用）。"""
        by_name = dict(
            zip(obs.get("state_names", []), obs.get("state", []), strict=False)
        )
        ee = np.array(
            [float(by_name.get(f"left_ee_pose.{i}", 0.0)) for i in range(7)]
        )
        arm = np.array(
            [float(by_name.get(n, 0.0)) for n in self.profile.arm_action_names],
            dtype=float,
        )
        kps = np.asarray(
            [k["position_m"] for k in (self._snapshot or {}).get("keypoints", [])],
            dtype=float,
        )
        if kps.size == 0:
            kps = np.zeros((1, 3), dtype=float)
        return {"ee_pose": ee, "arm_q": arm, "keypoints": kps}

    # ------------------------------------------------------- intent → action
    def _to_action(
        self,
        intent: dict[str, Any],
        world: dict[str, Any],
        state: np.ndarray,
        names: list[str],
    ) -> np.ndarray | None:
        kind = intent.get("type")
        if kind in ("done", "failed"):
            self._finished = True
            self._hold = True
            if kind == "failed":
                self._log(f"episode 执行失败: {intent.get('error')}")
            else:
                self._log("episode 执行结束")
            return self._hold_action(world, state, names)
        if kind == "waiting":
            return None

        if kind == "cartesian":
            pose7 = np.asarray(intent["pose7"], dtype=float)
            gripper = str(intent.get("gripper", "hold"))
            clean = self._sanitize_pose(pose7)
            if clean is None:
                # 非法位姿（NaN/退化四元数）绝不能下发给环境：PhysX 会整体崩 CUDA
                if not self._bad_pose_logged:
                    self._bad_pose_logged = True
                    self._log(f"丢弃非法位姿动作，改为保持位姿: {pose7}")
                return self._hold_action(world, state, names)
            pose7 = clean
        elif kind in ("close_gripper", "open_gripper"):
            # 夹爪动作：保持当前末端位姿，只改夹爪
            pose7 = np.asarray(world["ee_pose"], dtype=float)
            gripper = "close" if kind == "close_gripper" else "open"
        else:
            raise PolicyError("BAD_INTENT", f"未知动作意图: {kind!r}")

        if self.profile.action_space == "ee":
            pose_wxyz = np.concatenate([pose7[:3], _quat_xyzw_to_wxyz(pose7[3:])])
            action = self.profile.compose_ee_action(pose_wxyz, gripper, state, names)
        else:
            seed = np.array(
                [dict(zip(names, state, strict=False)).get(n, 0.0)
                 for n in self.profile.arm_action_names],
                dtype=np.float64,
            )
            arm_q = self.profile.solve_ik(pose7, seed)
            action = self.profile.compose_joint_action(arm_q, gripper, state, names)

        # 夹爪命令发出后按"执行成功"推进状态机（评测观测不提供抓取反馈）
        if kind == "close_gripper":
            self.executor.on_grasp_result(True)  # type: ignore[attr-defined]
        elif kind == "open_gripper":
            self.executor.on_release_done()  # type: ignore[attr-defined]
        return action