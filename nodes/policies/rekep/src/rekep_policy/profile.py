"""ReKep 策略的机器人 profile：动作命名映射 + 动作空间适配。

两种动作空间（评测端配置决定，必须与 benchmark 的 action_type 一致）：
- "joint"：策略输出关节目标（内部 IK，Cartesian→joint）。
- "ee"   ：策略输出末端位姿 [x,y,z,qw,qx,qy,qz]（RoboDojo 的 ee 动作模式，
           环境内部用 cuRobo IK 解算），无需自己 IK。

评测端期望的动作分量名（action_names）必须与 benchmark 的
action_component_names() 逐字一致；未控制的分量 hold 当前状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from forge_policy import PolicyError


class IKSolver(Protocol):
    def solve(self, target_pos: np.ndarray, target_quat_xyzw: np.ndarray,
              seed_q: np.ndarray) -> np.ndarray: ...


@dataclass
class RobotProfile:
    # 评测端期望的完整动作分量名（= PolicyContract.action_names）
    action_names: tuple[str, ...]
    # 本策略控制的臂关节名（joint 模式，按 IK 输出顺序；ee 模式下用于取当前关节 seed）
    arm_action_names: tuple[str, ...]
    # 夹爪分量名（1 个）
    gripper_action_name: str
    # 动作空间："joint" | "ee"
    action_space: str = "joint"
    # ee 模式下 7 个位姿分量名（顺序 [x,y,z,qw,qx,qy,qz]）
    pose_action_names: tuple[str, ...] = ()
    # 夹爪开/合对应的指令值
    gripper_open_value: float = 0.0
    gripper_close_value: float = 1.0
    # 臂关节限位（joint 模式；None 则不限幅）
    arm_lower: tuple[float, ...] | None = None
    arm_upper: tuple[float, ...] | None = None
    # 未控制分量的 hold 策略：True=跟随当前状态，False=固定 0
    hold_uncontrolled: bool = True
    # 任务工作空间（动作位置裁剪；None 不裁剪）
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None
    ik: IKSolver | None = None

    def __post_init__(self) -> None:
        if self.action_space not in ("joint", "ee"):
            raise ValueError(f"action_space 必须是 joint/ee，收到 {self.action_space!r}")
        needed = set(self.arm_action_names) | {self.gripper_action_name}
        if self.action_space == "ee":
            if len(self.pose_action_names) != 7:
                raise ValueError("ee 模式必须提供 7 个 pose_action_names")
            needed |= set(self.pose_action_names)
        missing = needed - set(self.action_names)
        if missing:
            raise ValueError(f"robot profile 引用了评测动作集之外的名字: {sorted(missing)}")
        if self.arm_lower is not None and len(self.arm_lower) != len(self.arm_action_names):
            raise ValueError("arm_lower 与 arm_action_names 长度不一致")
        if self.arm_upper is not None and len(self.arm_upper) != len(self.arm_action_names):
            raise ValueError("arm_upper 与 arm_action_names 长度不一致")

    # ------------------------------------------------------------ helpers
    def _by_name(self, current_state: np.ndarray, state_names: list[str]) -> dict[str, float]:
        return dict(zip(state_names, (float(x) for x in current_state), strict=False))

    def _gripper_value(self, gripper: str, by_name: dict[str, float]) -> float:
        if gripper == "close":
            return self.gripper_close_value
        if gripper == "open":
            return self.gripper_open_value
        return by_name.get(self.gripper_action_name, self.gripper_open_value)

    # ------------------------------------------------------------ compose
    def compose_joint_action(
        self,
        arm_q: np.ndarray,
        gripper: str,
        current_state: np.ndarray,
        state_names: list[str],
    ) -> np.ndarray:
        """joint 模式：关节目标 + 夹爪语义（close/open/hold）。"""
        by_name = self._by_name(current_state, state_names)
        arm = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if arm.size != len(self.arm_action_names):
            raise PolicyError(
                "IK_DIM_MISMATCH",
                f"IK 输出 {arm.size} 维 != 臂动作 {len(self.arm_action_names)} 维",
            )
        if self.arm_lower is not None:
            arm = np.clip(arm, np.asarray(self.arm_lower), np.asarray(self.arm_upper))
        out = np.zeros(len(self.action_names), dtype=np.float64)
        for i, name in enumerate(self.action_names):
            if name in self.arm_action_names:
                out[i] = arm[self.arm_action_names.index(name)]
            elif name == self.gripper_action_name:
                out[i] = self._gripper_value(gripper, by_name)
            else:
                out[i] = by_name.get(name, 0.0) if self.hold_uncontrolled else 0.0
        return out

    def compose_ee_action(
        self,
        pose7_wxyz: np.ndarray,
        gripper: str,
        current_state: np.ndarray,
        state_names: list[str],
    ) -> np.ndarray:
        """ee 模式：末端位姿 [x,y,z,qw,qx,qy,qz] + 夹爪语义。"""
        by_name = self._by_name(current_state, state_names)
        pose = np.asarray(pose7_wxyz, dtype=np.float64).reshape(-1)
        if pose.size != 7 or not np.all(np.isfinite(pose)):
            raise PolicyError("INVALID_CARTESIAN", f"非法位姿动作: {pose}")
        out = np.zeros(len(self.action_names), dtype=np.float64)
        for i, name in enumerate(self.action_names):
            if name in self.pose_action_names:
                out[i] = pose[self.pose_action_names.index(name)]
            elif name == self.gripper_action_name:
                out[i] = self._gripper_value(gripper, by_name)
            else:
                # 未控制臂：保持当前位姿/关节，避免另一条臂乱动
                out[i] = by_name.get(name, 0.0) if self.hold_uncontrolled else 0.0
        return out

    # ---------------------------------------------------------------- IK
    def solve_ik(self, cart: np.ndarray, seed_q: np.ndarray) -> np.ndarray:
        """joint 模式用：cart = [x,y,z,qx,qy,qz,qw]（xyzw）。"""
        if self.ik is None:
            raise PolicyError("IK_UNAVAILABLE", "robot profile 未配置 IK 求解器")
        cart = np.asarray(cart, dtype=np.float64).reshape(-1)
        if cart.size != 7 or not np.all(np.isfinite(cart)):
            raise PolicyError("INVALID_CARTESIAN", f"非法 Cartesian 目标: {cart}")
        q = self.ik.solve(cart[:3], cart[3:7], np.asarray(seed_q, dtype=np.float64))
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != len(self.arm_action_names):
            raise PolicyError("IK_DIM_MISMATCH", f"IK 返回 {q.size} 维")
        return q
