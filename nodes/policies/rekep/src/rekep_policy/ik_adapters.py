"""MuJoCo 数值 IK（用于 ReKep 求解器的可达性/路径代价评估）。

两种实现：
- StubIK：恒返回当前关节并"成功"。用于环境自带 IK（RoboDojo ee 动作模式由
  cuRobo 解算）时的 MVP：求解器不做可达性约束，真实 IK 由评测环境负责。
- X5IK：从 RoboDojo 自带的 ARX X5 URDF 建 MuJoCo 模型做真实数值 IK，
  按机身位姿把世界系目标变换进 base 系，用 body 雅可比求解。

约定：ReKep 内部 Cartesian 用 [x,y,z,qx,qy,qz,qw]（米/xyzw）；评测动作
用 [x,y,z,qw,qx,qy,qz]（wxyz），转换在 profile/core 边界完成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float)[:4]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float)[:4]
    return _quat_xyzw_to_mat(np.array([x, y, z, w]))


class StubIK:
    """不做可达性约束：返回 seed 关节并声明成功。

    说明：RoboDojo 的 ee 动作模式在环境内部用 cuRobo 求解，不可达目标会被
    环境拒绝；求解器只用于生成规划目标，MVP 阶段不需要第二套运动学。
    """

    def __init__(self, dof: int = 6) -> None:
        self.dof = dof

    def solve(self, target_pos, target_quat_xyzw, seed_q):
        return np.asarray(seed_q, dtype=float).reshape(-1)

    def solve_ik_result(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        class _Result:
            success = True
            num_descents = 1
            position_error = 0.0
            orientation_error = 0.0

        result = _Result()
        result.cspace_position = np.asarray(
            initial_joint_pos if initial_joint_pos is not None else np.zeros(self.dof), dtype=float
        )
        return result


class X5IK:
    """ARX X5 URDF 上的阻尼最小二乘 IK。"""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        joint_names: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        ee_body: str = "link6",
        tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(urdf_path).resolve()))
        self.data = mujoco.MjData(self.model)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in jids):
            bad = [n for n, j in zip(joint_names, jids, strict=True) if j < 0]
            raise ValueError(f"URDF 缺少关节: {bad}")
        self.joint_ids = np.asarray(jids, dtype=int)
        self.qposadr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.dofadr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.body_id < 0:
            raise ValueError(f"URDF 缺少 body: {ee_body}")
        self.tool_offset = np.asarray(tool_offset, dtype=float)
        # 世界→base 的逆变换
        self._base_p = np.asarray(base_pos, dtype=float)
        self._base_R = _quat_wxyz_to_mat(np.asarray(base_quat_wxyz, dtype=float))
        mujoco.mj_forward(self.model, self.data)
        self._reset_q = self.data.qpos[self.qposadr].copy()
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------- FK/IK
    def _tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.body_id].copy()
        rot = self.data.xmat[self.body_id].reshape(3, 3).copy()
        tool = pos + rot @ self.tool_offset
        return tool, rot

    def _ik_once(
        self, target_pos: np.ndarray, target_rot: np.ndarray, seed: np.ndarray,
        iterations: int = 400, pos_tol: float = 5e-4, rot_tol: float = 5e-3,
    ) -> np.ndarray:
        m = self._mujoco
        q = np.clip(np.asarray(seed, dtype=float).copy(), self.lower + 1e-5, self.upper - 1e-5)

        def _err(qq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.data.qpos[self.qposadr] = qq
            m.mj_forward(self.model, self.data)
            p, r = self._tool_pose()
            pe = target_pos - p
            re = 0.5 * np.sum(np.cross(r, target_rot), axis=0)
            return pe, re

        pe, re = _err(q)
        err = np.concatenate([pe, re])
        step = 0.25
        for _ in range(iterations):
            if np.linalg.norm(pe) < pos_tol and np.linalg.norm(re) < rot_tol:
                return q
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            m.mj_jacBody(self.model, self.data, jacp, jacr, self.body_id)
            jac = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            accepted = False
            for _try in range(10):
                q_new = np.clip(q + step * dq, self.lower + 1e-5, self.upper - 1e-5)
                pe2, re2 = _err(q_new)
                err2 = np.concatenate([pe2, re2])
                if np.linalg.norm(err2) < np.linalg.norm(err):
                    q, pe, re, err = q_new, pe2, re2, err2
                    step = min(step * 1.4, 0.6)
                    accepted = True
                    break
                step *= 0.5
                if step < 1e-4:
                    break
            if not accepted and step < 1e-4:
                break
        raise RuntimeError(f"IK 未收敛: pos_err={np.linalg.norm(pe):.4f}")

    def solve(self, target_pos, target_quat_xyzw, seed_q, *, iterations: int = 120,
              seed_tries: int = 3):
        """世界系目标（xyzw）→ base 系 → 多 seed IK；失败抛 RuntimeError。"""
        pos_w = np.asarray(target_pos, dtype=float)[:3]
        rot_w = _quat_xyzw_to_mat(np.asarray(target_quat_xyzw, dtype=float))
        pos_b = self._base_R.T @ (pos_w - self._base_p)
        rot_b = self._base_R.T @ rot_w
        seeds = [np.asarray(seed_q, dtype=float), self._reset_q]
        for _ in range(max(0, seed_tries - 2)):
            seeds.append(self._rng.uniform(self.lower, self.upper))
        last: Exception | None = None
        for seed in seeds[: max(1, seed_tries)]:
            try:
                return self._ik_once(pos_b, rot_b, seed, iterations=iterations)
            except RuntimeError as exc:
                last = exc
        raise RuntimeError(f"IK 无解: {last}")

    def solve_ik_result(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        target = np.asarray(target_pose_homo, dtype=float).reshape(4, 4)
        pos = target[:3, 3]
        rot = target[:3, :3]
        # rot → xyzw
        w = np.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
        x = np.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) / 2.0
        y = np.sqrt(max(0.0, 1.0 - rot[0, 0] + rot[1, 1] - rot[2, 2])) / 2.0
        z = np.sqrt(max(0.0, 1.0 - rot[0, 0] - rot[1, 1] + rot[2, 2])) / 2.0
        x = np.copysign(x, rot[2, 1] - rot[1, 2])
        y = np.copysign(y, rot[0, 2] - rot[2, 0])
        z = np.copysign(z, rot[1, 0] - rot[0, 1])
        quat = np.array([x, y, z, w])

        class _Result:
            pass

        result = _Result()
        seed = (
            np.asarray(initial_joint_pos, dtype=float)
            if initial_joint_pos is not None
            else self.data.qpos[self.qposadr].copy()
        )
        try:
            q = self.solve(pos, quat, seed, iterations=max(10, int(max_iterations)), seed_tries=2)
            result.success = True
            result.num_descents = 1
            result.position_error = 0.0
            result.orientation_error = 0.0
            result.cspace_position = q
        except RuntimeError:
            result.success = False
            result.num_descents = 1
            result.position_error = float("inf")
            result.orientation_error = float("inf")
            result.cspace_position = seed
        return result
