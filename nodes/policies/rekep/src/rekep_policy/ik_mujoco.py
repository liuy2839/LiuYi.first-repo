"""MuJoCo 数值 IK 求解器（移植自 rekep-skill-paos-main/src/rekep_sim/env/mujoco_env.py
的 ik/_robust_ik，去掉仿真步进，只保留求解），用于 Cartesian→Joint 转换。

机器人模型：MuJoCo MJCF（如 Franka Panda menagerie scene.xml）。site 名默认
"tool"（Panda hand 上的工具点；不存在时由调用方在 MJCF 里声明）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forge_policy import PolicyError


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class MujocoIK:
    """阻尼最小二乘 + 自适应步长的数值 IK；限位裁剪；多 seed 重试。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        joint_names: tuple[str, ...],
        site_name: str = "tool",
        base_frame: str | None = None,
        world_to_base: np.ndarray | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names
        ]
        if any(j < 0 for j in jids):
            bad = [n for n, j in zip(joint_names, jids, strict=True) if j < 0]
            raise ValueError(f"MJCF 缺少关节: {bad}")
        self.joint_ids = np.asarray(jids, dtype=int)
        self.qposadr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.dofadr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"MJCF 缺少 site: {site_name}")
        # 目标位姿给到 base 坐标系；评测世界系→base 系的固定变换（无则恒等）
        self.world_to_base = (
            np.eye(4) if world_to_base is None else np.asarray(world_to_base, dtype=float)
        )
        mujoco.mj_forward(self.model, self.data)
        self._rng = np.random.default_rng(0)
        self._reset_q = self.data.qpos[self.qposadr].copy()

    def _fk_site(self) -> tuple[np.ndarray, np.ndarray]:
        m = self._mujoco
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def _ik_from_seed(
        self, target_pos: np.ndarray, target_rot: np.ndarray, seed: np.ndarray,
        iterations: int = 500, pos_tol: float = 2e-4, rot_tol: float = 2e-3,
    ) -> np.ndarray:
        m = self._mujoco
        q = np.clip(np.asarray(seed, dtype=float).copy(), self.lower + 1e-5, self.upper - 1e-5)

        def _err(qq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.data.qpos[self.qposadr] = qq
            m.mj_forward(self.model, self.data)
            p, r = self._fk_site()
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
            m.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            accepted = False
            for _attempt in range(10):
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
        raise PolicyError("IK_SEED_FAILED", f"IK 未收敛: pos_err={np.linalg.norm(pe):.4f}")

    def solve(self, target_pos: np.ndarray, target_quat_xyzw: np.ndarray,
              seed_q: np.ndarray) -> np.ndarray:
        """评测世界系目标 → base 系 → 多 seed 数值 IK。失败抛 PolicyError。"""
        # world→base
        tb = self.world_to_base
        pos_w = np.asarray(target_pos, dtype=float)
        pos_b = (tb[:3, :3] @ pos_w + tb[:3, 3])
        rot_b = tb[:3, :3] @ _quat_xyzw_to_mat(np.asarray(target_quat_xyzw, dtype=float))
        seeds = [np.asarray(seed_q, dtype=float), self._reset_q]
        for _ in range(12):
            seeds.append(self._rng.uniform(self.lower, self.upper))
        last: Exception | None = None
        for i, seed in enumerate(seeds):
            try:
                q = self._ik_from_seed(pos_b, rot_b, seed)
                if i > 0:  # 随机 seed 命中时优先取连续解：仍返回它（可达性优先）
                    return q
                return q
            except PolicyError as exc:
                last = exc
                continue
        raise PolicyError("IK_UNREACHABLE", f"全部 seed 不可达: {last}")
