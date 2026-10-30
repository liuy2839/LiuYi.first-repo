"""rekep_alg.executor：分阶段求解执行状态机（步进式，不推进仿真）。

移植自 rekep_sim/runtime.py 的 execute() 主循环：原来内部 while 循环连解带
执行（env.execute_action 步进自有 MuJoCo）；本模块改为增量状态机——
world_state 由协议层每帧喂入（末端位姿/关节/关键点），每次 step() 至多产出
一个动作意图（Cartesian 目标 + 夹爪），由评测端 1:1 执行。

求解器（subgoal/path）来自 reference runtime，IK 由外部注入
（rekep_policy 的 MujocoIK 适配 solve_ik_result 接口）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .planner import load_functions


class ExecutorConfig:
    def __init__(self, **kw: Any) -> None:
        self.constraint_tolerance: float = kw.get("constraint_tolerance", 0.10)
        self.grasp_depth: float = kw.get("grasp_depth", 0.03)
        self.interpolate_pos_step_size: float = kw.get("interpolate_pos_step_size", 0.05)
        self.interpolate_rot_step_size: float = kw.get("interpolate_rot_step_size", 0.34)
        self.subgoal_sampling_maxfun: int = kw.get("subgoal_sampling_maxfun", 2000)
        self.path_sampling_maxfun: int = kw.get("path_sampling_maxfun", 2000)
        self.minimizer_maxiter: int = kw.get("minimizer_maxiter", 100)
        self.seed: int = kw.get("seed", 0)
        # 安全：路径点距任意障碍物中心小于该半径时跳过（环境无碰撞模型）
        self.obstacle_radius: float = kw.get("obstacle_radius", 0.06)
        # 安全：末端最低高度（相对场景最低关键点）
        self.min_clearance: float = kw.get("min_clearance", 0.01)


class IKAdapter:
    """把外部 IK 适配成 reference solver 期望的 .solve(target_homo, ...)。"""

    def __init__(self, solve_result_fn) -> None:
        self._fn = solve_result_fn

    def solve(self, target_pose_homo, position_tolerance=0.01, orientation_tolerance=0.05,
              position_weight=1.0, orientation_weight=0.05, max_iterations=150,
              initial_joint_pos=None):
        return self._fn(target_pose_homo, position_tolerance, orientation_tolerance,
                        position_weight, orientation_weight, max_iterations,
                        initial_joint_pos)


class _StepResult(dict):
    """动作意图：见 rekep_policy.core.ExecutorModule 协议。"""


class ReKepExecutor:
    """episode 级状态机：start_episode(plan) 后每帧 step(world_state)。"""

    def __init__(self, ik, *, config: ExecutorConfig | None = None,
                 solver_factory=None, transform_utils=None,
                 reset_joint_pos: np.ndarray | None = None,
                 bounds_min=(-0.7, -0.9, 0.0), bounds_max=(0.7, 0.6, 1.3)):
        """ik: IKAdapter；solver_factory(subgoal_cfg, path_cfg, ik, reset_q) →
        (subgoal_solver, path_solver)；transform_utils: 参考 T 模块（ quat 转换）。
        均可由部署层注入；默认走 reference runtime 的实现。"""
        self.ik = ik
        self.config = config or ExecutorConfig()
        self._solver_factory = solver_factory
        self._T = transform_utils
        self._reset_q = (
            np.asarray(reset_joint_pos, dtype=float)
            if reset_joint_pos is not None
            else np.zeros(6, dtype=float)
        )
        self._bounds_min = np.asarray(bounds_min, dtype=float)
        self._bounds_max = np.asarray(bounds_max, dtype=float)
        self._plan: dict | None = None
        self._stage = 1
        self._action_queue: list[np.ndarray] = []
        self._first_iter = True
        self._stage_planned = False
        self._phases: list[dict] = []
        self._finished = False
        self._subgoal_solver = None
        self._path_solver = None

    # ---------------------------------------------------------------- state
    def start_episode(self, plan: dict, snapshot: dict) -> None:
        self._plan = plan
        self._stage = 1
        self._action_queue = []
        self._first_iter = True
        self._stage_planned = False
        self._phases = []
        self._finished = False
        self._build_solvers(self._reset_q, self._bounds_min, self._bounds_max)

    def _build_solvers(self, reset_q: np.ndarray, bmin, bmax) -> None:
        if self._solver_factory is None:
            from .perception import _ensure_refimpl

            _ensure_refimpl()
            import path_solver as path_mod
            import subgoal_solver as subgoal_mod

            # reference 的构造预热用随机数据触发 numba 编译；偶发命中内部断言
            # （numba 下断言异常本身还会 PicklingError）。预热只是加速，失败不应致命。
            for mod in (subgoal_mod, path_mod):
                cls = mod.SubgoalSolver if mod is subgoal_mod else mod.PathSolver
                orig = cls._warmup
                if not getattr(orig, "_rekep_safe", False):
                    def _safe_warmup(self, _orig=orig):
                        try:
                            _orig(self)
                        except Exception:
                            pass

                    _safe_warmup._rekep_safe = True  # type: ignore[attr-defined]
                    cls._warmup = _safe_warmup

            cfg = self.config
            subgoal_cfg = {
                "bounds_min": list(bmin), "bounds_max": list(bmax),
                "sampling_maxfun": cfg.subgoal_sampling_maxfun,
                "max_collision_points": 60,
                "constraint_tolerance": 1e-4,
                "minimizer_options": {"maxiter": cfg.minimizer_maxiter},
                "grasp_axis_local": [1.0, 0.0, 0.0],
                "grasp_preferred_world_dir": [0.0, 0.0, -1.0],
            }
            path_cfg = {
                "bounds_min": list(bmin), "bounds_max": list(bmax),
                "opt_pos_step_size": 0.20, "opt_rot_step_size": 0.78,
                "opt_interpolate_pos_step_size": 0.03,
                "opt_interpolate_rot_step_size": 0.20,
                "max_collision_points": 60,
                "sampling_maxfun": cfg.path_sampling_maxfun,
                "constraint_tolerance": 1e-4,
                "minimizer_options": {"maxiter": cfg.minimizer_maxiter},
            }
            self._subgoal_solver = subgoal_mod.SubgoalSolver(subgoal_cfg, self.ik, reset_q)
            self._path_solver = path_mod.PathSolver(path_cfg, self.ik, reset_q)
        else:
            self._subgoal_solver, self._path_solver = self._solver_factory(
                reset_q, bmin, bmax
            )
        if self._T is None:
            from .perception import _ensure_refimpl

            _ensure_refimpl()
            import transform_utils as T

            self._T = T

    # ----------------------------------------------------------------- step
    def step(self, world_state: dict[str, Any]) -> dict[str, Any]:
        """推进状态机，至多产出一个动作意图。

        world_state: {
          ee_pose: (7,) xyzw 米; arm_q: (n,); keypoints: (K,3) 米;
          sdf_voxels: ndarray|None; collision_points: ndarray|None;
          is_grasping: bool|None
        }
        """
        if self._finished or self._plan is None:
            return {"type": "done"}

        obstacles = self._obstacles(world_state)
        kps = np.asarray(world_state.get("keypoints", np.zeros((0, 3))), dtype=float)
        floor_z = float(kps[:, 2].min()) - 0.02 if kps.size else None

        # 1) 队列里还有插值动作：逐个下发（gripper null）
        while self._action_queue:
            act = self._action_queue.pop(0)
            safe = self._guard(act, obstacles, floor_z)
            if safe is None:
                continue
            return {"type": "cartesian", "pose7": safe[:7].tolist(),
                    "gripper": self._gripper_of(safe[7])}

        plan = self._plan
        stage = self._stage
        num_stages = int(plan["num_stages"])
        grasp_kps = list(plan["grasp_keypoints"])
        release_kps = list(plan["release_keypoints"])
        cfg = self.config
        ee_pose = np.asarray(world_state["ee_pose"], dtype=float)
        keypoints = np.asarray(world_state["keypoints"], dtype=float)
        solver_keypoints = np.vstack([ee_pose[:3], keypoints])
        arm_q = np.asarray(world_state["arm_q"], dtype=float)
        sdf = world_state.get("sdf_voxels")
        collision = world_state.get("collision_points")
        if sdf is None:
            # Diagnostic profile without collision geometry; negative values are free space in the reference solver.
            sdf = np.full((8, 8, 8), -1.0, dtype=float)
        if collision is None or len(collision) == 0:
            collision = ee_pose[:3].reshape(1, 3)

        # 2) 回退检查：上一 stage 的 path 约束被破坏 → 退一 stage
        if stage > 1:
            grasped = {int(grasp_kps[j - 1]) for j in range(1, stage)
                       if grasp_kps[j - 1] != -1}
            text = plan["stages"][stage - 1]["path_constraints"]
            pat_fns = load_functions(text, grasped_keypoints=grasped) if text.strip() else []
            violated = any(
                float(fn(ee_pose[:3], keypoints)) > cfg.constraint_tolerance for fn in pat_fns
            ) if pat_fns else False
            if violated:
                self._stage = max(1, stage - 1)
                self._first_iter = True
                self._stage_planned = False
                self._action_queue = []
                self._phases.append({"phase": f"backtrack_to_stage_{self._stage}"})
                return {"type": "waiting"}

        if self._stage_planned:
            if grasp_kps[stage - 1] != -1:
                return {"type": "close_gripper"}
            if release_kps[stage - 1] != -1:
                return {"type": "open_gripper"}
            if stage >= num_stages:
                self._finished = True
                return {"type": "done"}
            self._stage += 1
            self._first_iter = True
            self._stage_planned = False
            return self.step(world_state)

        # 3) 求解当前 stage 的 subgoal + path
        is_grasp = grasp_kps[stage - 1] != -1
        is_release = release_kps[stage - 1] != -1
        grasped = {int(grasp_kps[j - 1]) for j in range(1, stage)
                   if grasp_kps[j - 1] != -1}
        sub_text = plan["stages"][stage - 1]["subgoal_constraints"]
        path_text = plan["stages"][stage - 1]["path_constraints"]
        sub_fns = load_functions(sub_text, grasped_keypoints=grasped) if sub_text.strip() else []
        pat_fns = load_functions(path_text, grasped_keypoints=grasped) if path_text.strip() else []
        movable_mask = np.zeros(len(keypoints) + 1, dtype=bool)
        movable_mask[0] = True
        # 抓着的物体的关键点在优化中视为可动（参考实现按 env.is_grasping 更新；
        # 协议层通过 world_state.movable_keypoints 提供）
        mk = world_state.get("movable_keypoints")
        if mk is not None:
            for i in np.asarray(mk, dtype=int):
                if 0 < i < len(movable_mask):
                    movable_mask[i] = True

        try:
            subgoal, _dbg1 = self._subgoal_solver.solve(
                ee_pose, solver_keypoints, movable_mask, sub_fns, pat_fns,
                sdf, collision, is_grasp, arm_q, from_scratch=self._first_iter,
            )
            subgoal_pose = np.asarray(subgoal, dtype=float)
            if is_grasp:
                homo = self._T.convert_pose_quat2mat(subgoal_pose)
                subgoal_pose = subgoal_pose.copy()
                subgoal_pose[:3] += homo[:3, :3] @ np.array([-cfg.grasp_depth / 2.0, 0.0, 0.0])
            path, _dbg2 = self._path_solver.solve(
                ee_pose, subgoal_pose, solver_keypoints, movable_mask, pat_fns,
                sdf, collision, arm_q, from_scratch=self._first_iter,
            )
            self._action_queue = self._densify(ee_pose, path)
        except Exception as exc:  # noqa: BLE001 求解失败不应打死节点
            self._finished = True
            self._phases.append({"phase": f"stage{stage}_solve_failed", "error": repr(exc)})
            return {"type": "failed", "error": repr(exc)}
        self._first_iter = False
        self._stage_planned = True
        while self._action_queue:
            act = self._action_queue.pop(0)
            safe = self._guard(act, obstacles, floor_z)
            if safe is None:
                continue
            return {"type": "cartesian", "pose7": safe[:7].tolist(),
                    "gripper": self._gripper_of(safe[7])}

        # 4) stage 收尾：抓 / 放 / 推进
        if is_grasp:
            return {"type": "close_gripper"}
        if is_release:
            return {"type": "open_gripper"}
        if stage >= num_stages:
            self._finished = True
            return {"type": "done"}
        self._stage = stage + 1
        self._first_iter = True
        return {"type": "waiting"}

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _gripper_of(v: float) -> str:
        if v > 0:
            return "close"
        if v < 0:
            return "open"
        return "hold"

    # ------------------------------------------------------- safety guards
    def _obstacles(self, world_state: dict[str, Any]) -> np.ndarray:
        raw = world_state.get("obstacles")
        if raw is None:
            return np.zeros((0, 3), dtype=float)
        arr = np.asarray(raw, dtype=float).reshape(-1, 3)
        plan = self._plan or {}
        grasp = list(plan.get("grasp_keypoints") or [])
        stage = self._stage
        if 0 < stage <= len(grasp) and int(grasp[stage - 1]) != -1:
            kps = np.asarray(world_state.get("keypoints", np.zeros((0, 3))), dtype=float)
            idx = int(grasp[stage - 1])
            if 0 <= idx < len(kps):
                # 当前要抓的目标不算障碍，否则永远无法靠近
                arr = arr[np.linalg.norm(arr - kps[idx], axis=1) > 0.05]
        return arr

    def _guard(self, act: np.ndarray, obstacles: np.ndarray,
               floor_z: float | None) -> np.ndarray | None:
        """动作安全护栏：有限值、离地高度、避障半径。不合格返回 None 表示跳过。"""
        a = np.asarray(act, dtype=float).copy()
        if not np.all(np.isfinite(a)):
            return None
        if floor_z is not None:
            a[2] = max(float(a[2]), float(floor_z))
        if obstacles is not None and len(obstacles):
            if float(np.linalg.norm(obstacles - a[:3], axis=1).min()) < self.config.obstacle_radius:
                return None
        return a

    def _densify(self, ee_pose: np.ndarray, path_quat: np.ndarray) -> list[np.ndarray]:
        """路径样条加密（参考 _process_path；gripper 位为 null=0）。"""
        cfg = self.config
        from .perception import _ensure_refimpl

        _ensure_refimpl()
        import utils as ref_utils

        full = np.concatenate([ee_pose.reshape(1, -1), path_quat], axis=0)
        num_steps = ref_utils.get_linear_interpolation_steps(
            full[0], full[-1],
            cfg.interpolate_pos_step_size, cfg.interpolate_rot_step_size,
        )
        dense = ref_utils.spline_interpolate_poses(full, max(num_steps, 2))
        actions = np.zeros((dense.shape[0], 8))
        actions[:, :7] = dense
        actions[:, 7] = 0.0  # null gripper
        return [a for a in actions]

    def on_grasp_result(self, ok: bool) -> dict[str, Any]:
        """协议层执行完 close_gripper 后回告结果。"""
        plan = self._plan
        self._phases.append({"phase": f"stage{self._stage}_grasp", "ok": bool(ok)})
        if not ok:
            self._finished = True
            return {"type": "done"}
        if self._stage >= int(plan["num_stages"]):
            self._finished = True
            return {"type": "done"}
        self._stage += 1
        self._first_iter = True
        self._stage_planned = False
        return {"type": "waiting"}

    def on_release_done(self) -> dict[str, Any]:
        plan = self._plan
        self._phases.append({"phase": f"stage{self._stage}_release", "ok": True})
        if self._stage >= int(plan["num_stages"]):
            self._finished = True
            return {"type": "done"}
        self._stage += 1
        self._first_iter = True
        self._stage_planned = False
        return {"type": "waiting"}
