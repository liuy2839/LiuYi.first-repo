"""感知适配器：把评测观测转成 ReKep 感知快照与逐帧世界状态。

SceneObjectPerception（当前评测路径）：环境不提供 depth/seg（RoboDojo 实测
只有 RGB），改用评测端发布的 scene_objects（物体名/类型/世界系位姿）构建
center 关键点。优势是跟踪免费——每帧都是仿真真值；代价是没有 DINOv2 的
表面细粒度关键点，任务表达以物体中心为单位（对抓取-放置足够）。

ReKep 内部坐标/四元数约定：position 米、world 系；四元数 **xyzw**
（评测端观测/动作是 wxyz，转换在本模块边界完成）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _quat_wxyz_to_xyzw(q: list[float] | np.ndarray) -> list[float]:
    w, x, y, z = (float(v) for v in q[:4])
    return [x, y, z, w]


def _quat_xyzw_to_wxyz(q: np.ndarray) -> list[float]:
    x, y, z, w = (float(v) for v in q[:4])
    return [w, x, y, z]


class SceneObjectPerception:
    """基于 scene_objects 的感知适配（方案 B）。"""

    def __init__(
        self,
        *,
        movable_types: tuple[str, ...] = ("rigid",),
        region_types: tuple[str, ...] = ("geometry", "articulation"),
        ee_pose_prefix: str = "left_ee_pose",
        arm_prefix: str = "left_arm_joint_state",
    ) -> None:
        self.movable_types = movable_types
        self.region_types = region_types
        self.ee_pose_prefix = ee_pose_prefix
        self.arm_prefix = arm_prefix

    # 容器/目标区语义（layout_manager 的 geometry 里也混有桌子等静态体）
    REGION_KEYWORDS = ("dustbin", "bin", "zone", "region", "container", "box", "holder", "vase", "cup")

    def _is_region(self, o: dict[str, Any]) -> bool:
        if o.get("type") not in self.region_types:
            return False
        name = str(o.get("name", "")).lower()
        return any(k in name for k in self.REGION_KEYWORDS)

    # ------------------------------------------------------------ snapshot
    def keypoints(self, obs: dict[str, Any]) -> dict[str, Any]:
        objects = obs.get("scene_objects") or []
        if not objects:
            raise PerceptionUnavailable("scene_objects 为空：评测端未发布场景物体")
        movables = [o for o in objects if o.get("type") in self.movable_types]
        regions = [o for o in objects if self._is_region(o)]
        keypoints: list[dict[str, Any]] = []
        for o in movables + regions:
            keypoints.append(
                {
                    "index": len(keypoints),
                    "position_m": [float(v) for v in o["position"][:3]],
                    "pixel_uv": [-1, -1],
                    "object": o["name"],
                    "kind": "center",
                }
            )
        objects_meta = [
            {
                "name": o["name"],
                "movable": o.get("type") in self.movable_types,
                "region": self._is_region(o),
                "position_m": [float(v) for v in o["position"][:3]],
                "quat_xyzw": _quat_wxyz_to_xyzw(o.get("quat_wxyz", [1, 0, 0, 0])),
            }
            for o in movables + regions
        ]
        payload = [
            {"name": o["name"], "p": [round(float(v), 4) for v in o["position"][:3]]}
            for o in movables + regions
        ]
        observation_id = "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "schema_version": "rekep_policy.scene_objects.v1",
            "observation_id": observation_id,
            "keypoints": keypoints,
            "objects": objects_meta,
        }

    # --------------------------------------------------------- world state
    def world_state(self, obs: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        live = {
            o["name"]: np.asarray(o["position"][:3], dtype=float)
            for o in (obs.get("scene_objects") or [])
        }
        keypoints = np.asarray(
            [live.get(k["object"], np.asarray(k["position_m"], dtype=float))
             for k in snapshot["keypoints"]],
            dtype=float,
        )
        ee_pose = self._read_ee_pose(obs)
        arm_q = self._read_arm_q(obs)
        return {"ee_pose": ee_pose, "arm_q": arm_q, "keypoints": keypoints}

    def _read_ee_pose(self, obs: dict[str, Any]) -> np.ndarray:
        by_name = dict(zip(obs.get("state_names", []), obs.get("state", []), strict=False))
        vals = [by_name.get(f"{self.ee_pose_prefix}.{i}") for i in range(7)]
        if any(v is None for v in vals):
            raise PerceptionUnavailable(
                f"观测缺少 {self.ee_pose_prefix}.0..6（末端位姿）"
            )
        pose_wxyz = np.asarray([float(v) for v in vals], dtype=float)
        out = pose_wxyz.copy()
        out[3:] = _quat_wxyz_to_xyzw(pose_wxyz[3:])
        return out

    def _read_arm_q(self, obs: dict[str, Any]) -> np.ndarray:
        by_name = dict(zip(obs.get("state_names", []), obs.get("state", []), strict=False))
        vals: list[float] = []
        i = 0
        while True:
            v = by_name.get(f"{self.arm_prefix}.{i}")
            if v is None:
                break
            vals.append(float(v))
            i += 1
        if not vals:
            raise PerceptionUnavailable(f"观测缺少 {self.arm_prefix}.0.. （臂关节）")
        return np.asarray(vals, dtype=float)


class PerceptionUnavailable(RuntimeError):
    """本帧观测不足以做感知（runtime 视为可重试）。"""


class DepthDinoPerception:
    """真实视觉感知：深度反投影 → 前景连通域伪分割 → DINOv2 关键点。

    RoboDojo 观测没有实例分割，这里用深度在相机系反投影成点云，再按
    工作空间高度阈值取前景、连通域标号作为伪分割标签；DINOv2 在每块前景上
    聚类出关键点。区域（如垃圾桶）用 scene_objects 的命名真值合并进来，
    保证规划器能按名字引用。
    逐帧跟踪不重跑 DINOv2：按伪分割标号把快照关键点吸附到最新连通域质心。
    """

    def __init__(
        self,
        *,
        image_topic: str = "image/head",
        depth_topic: str = "depth/head",
        depth_min: float = 0.15,
        depth_max: float = 2.5,
        table_z_margin: float = 0.03,
        max_objects: int = 6,
        min_area_px: int = 400,
        ee_pose_prefix: str = "left_ee_pose",
        arm_prefix: str = "left_arm_joint_state",
        proposer_config: dict | None = None,
        seed: int = 0,
        log: Any = None,
    ) -> None:
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.table_z_margin = table_z_margin
        self.max_objects = max_objects
        self.min_area_px = min_area_px
        self.ee_pose_prefix = ee_pose_prefix
        self.arm_prefix = arm_prefix
        self.proposer_config = dict(proposer_config or {})
        self.seed = seed
        self._labels: dict[int, str] = {}
        self._convention: str | None = None
        self._movable_anchors: list[np.ndarray] = []
        self._log = log or print

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W]
        z = depth
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _to_world(pts: np.ndarray, E: np.ndarray) -> np.ndarray:
        R = E[:3, :3]
        t = E[:3, 3]
        return pts @ R.T + t

    def _points_world(self, depth: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
        """反投影到世界系。相机光学系带有 y/z 翻转的不确定性：用真值物体
        位置自动挑选（8 个候选：E 或 E⁻¹ × 4 种轴翻转）。"""
        pts_cam = self._backproject(depth, K)
        cands = self._candidate_points(depth, pts_cam, E)
        anchor = self._movable_anchors
        if not self._convention and anchor:
            best = None
            for key, pts in cands.items():
                score = self._match_score(pts, depth, anchor)
                if best is None or score < best[1]:
                    best = (key, score)
            if best is not None:
                self._convention = best[0]
                self._log(f"投影约定自动标定: {best[0]} (score={best[1]:.3f})")
        key = self._convention or "E+"
        return cands.get(key, next(iter(cands.values())))

    def _candidate_points(self, depth, pts_cam, E) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for ename, mat in (("E", E), ("Ei", np.linalg.inv(E))):
            for tag, sy, sz in (
                ("", 1.0, 1.0),
                ("y", -1.0, 1.0),
                ("z", 1.0, -1.0),
                ("yz", -1.0, -1.0),
            ):
                flipped = pts_cam * np.array([1.0, sy, sz])
                out[f"{ename}+{tag}".replace("++", "+")] = self._to_world(flipped, mat)
        return out

    def _match_score(self, pts: np.ndarray, depth: np.ndarray, anchors: list[np.ndarray]) -> float:
        try:
            seg, _ = self._foreground_seg(pts, depth)
        except PerceptionUnavailable:
            return float("inf")
        centroids = []
        for lab in np.unique(seg):
            if lab == 0:
                continue
            mask = seg == lab
            c = np.nanmean(np.where(mask[..., None], pts, np.nan), axis=(0, 1))
            if np.isfinite(c).all():
                centroids.append(c)
        if not centroids:
            return float("inf")
        centroids_arr = np.asarray(centroids)
        return float(np.mean([
            np.min(np.linalg.norm(centroids_arr - a, axis=1)) for a in anchors
        ]))

    def _foreground_seg(self, pts: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy import ndimage

        valid = (
            np.isfinite(depth) & (depth > self.depth_min) & (depth < self.depth_max)
            & np.isfinite(pts).all(axis=-1)
        )
        z = pts[..., 2]
        z_valid = np.where(valid, z, np.nan)
        table = np.nanpercentile(z_valid, 15) if np.isfinite(z_valid).any() else 0.0
        fg = valid & (z > table + self.table_z_margin)
        labels, num = ndimage.label(fg)
        if num == 0:
            raise PerceptionUnavailable("深度前景为空：检查深度范围/工作空间阈值")
        sizes = ndimage.sum(fg, labels, range(1, num + 1))
        order = np.argsort(sizes)[::-1][: self.max_objects]
        keep = {int(i + 1) for i in order if sizes[i] >= self.min_area_px}
        out = np.zeros_like(labels)
        remap = {lab: idx + 1 for idx, lab in enumerate(sorted(keep))}
        for lab, new in remap.items():
            out[labels == lab] = new
        if not remap:
            raise PerceptionUnavailable("前景连通域都小于 min_area_px")
        return out, remap

    def _region_objects(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        """只把"容器/目标区"类实例当 region。

        RoboDojo 的 layout_manager 把桌子等静态体也标成 geometry；若不加过滤，
        模板规划会误判为"放到区域"。这里按名字关键词识别容器语义。
        """
        keywords = ("dustbin", "bin", "zone", "region", "container", "box", "holder", "vase", "cup")
        out = []
        for o in obs.get("scene_objects") or []:
            if o.get("type") not in ("geometry", "articulation"):
                continue
            name = str(o.get("name", "")).lower()
            if not any(k in name for k in keywords):
                continue
            out.append({
                "name": o["name"],
                "movable": False,
                "region": True,
                "position_m": [float(v) for v in o["position"][:3]],
                "quat_xyzw": _quat_wxyz_to_xyzw(o.get("quat_wxyz", [1, 0, 0, 0])),
            })
        return out

    # ------------------------------------------------------------ snapshot
    def keypoints(self, obs: dict[str, Any]) -> dict[str, Any]:
        from rekep_alg.perception import keypoints_from_observation

        rgb = obs["images"][self.image_topic]
        depth = np.asarray(obs["depths"][self.depth_topic], dtype=np.float32)
        info = (obs.get("camera_info") or {})[self.depth_topic]
        K = np.asarray(info["intrinsics"], dtype=float)
        E = np.asarray(info["extrinsics"], dtype=float)
        self._movable_anchors = [
            np.asarray(o["position"][:3], dtype=float)
            for o in (obs.get("scene_objects") or [])
            if o.get("type") == "rigid"
        ]
        pts = self._points_world(depth, K, E)
        seg, remap = self._foreground_seg(pts, depth)

        # 物体簇命名：object_<n>；区域来自 scene_objects 的命名真值
        labels = sorted(remap.values())
        cluster_objects = []
        for lab in labels:
            mask = seg == lab
            pos = np.nanmean(np.where(mask[..., None], pts, np.nan), axis=(0, 1))
            cluster_objects.append({
                "name": f"object_{lab}",
                "movable": True,
                "region": False,
                "position_m": [float(v) for v in pos],
            })
        regions = self._region_objects(obs)
        objects = [self._merge_quat(o) for o in cluster_objects]
        objects += [self._merge_quat(o) for o in regions]

        region_pts = {tuple(np.round(o["position_m"], 3)) for o in regions}
        bounds_min = [float(pts[..., i][np.isfinite(pts[..., i])].min()) for i in range(3)]
        bounds_max = [float(pts[..., i][np.isfinite(pts[..., i])].max()) for i in range(3)]
        snapshot = keypoints_from_observation(
            rgb, pts, seg,
            bounds_min=bounds_min, bounds_max=bounds_max,
            objects=objects, proposer_config=self.proposer_config or None,
            seed=self.seed,
        )
        snapshot["objects"] = objects
        snapshot["_seg_labels"] = {k["index"]: None for k in snapshot["keypoints"]}
        self._labels = {lab: f"object_{lab}" for lab in labels}
        # 记录每个关键点所属簇（供逐帧吸附）
        lab_by_name = {f"object_{lab}": lab for lab in labels}
        for k in snapshot["keypoints"]:
            k["_cluster"] = lab_by_name.get(k.get("object"))
        _ = region_pts
        return snapshot

    @staticmethod
    def _merge_quat(o: dict[str, Any]) -> dict[str, Any]:
        o.setdefault("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        return o

    # --------------------------------------------------------- world state
    def world_state(self, obs: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        by_name = dict(zip(obs.get("state_names", []), obs.get("state", []), strict=False))
        vals = [by_name.get(f"{self.ee_pose_prefix}.{i}") for i in range(7)]
        if any(v is None for v in vals):
            raise PerceptionUnavailable(f"观测缺少 {self.ee_pose_prefix}.0..6")
        pose_wxyz = np.asarray([float(v) for v in vals], dtype=float)
        ee = pose_wxyz.copy()
        ee[3:] = _quat_wxyz_to_xyzw(pose_wxyz[3:])
        arm = []
        i = 0
        while True:
            v = by_name.get(f"{self.arm_prefix}.{i}")
            if v is None:
                break
            arm.append(float(v))
            i += 1
        if not arm:
            raise PerceptionUnavailable(f"观测缺少 {self.arm_prefix}.0..")

        keypoints = np.asarray(
            [k["position_m"] for k in snapshot["keypoints"]], dtype=float
        )
        # 逐帧吸附：按簇质心更新（不重跑 DINOv2）
        live_regions = {
            o["name"]: np.asarray(o["position"][:3], dtype=float)
            for o in (obs.get("scene_objects") or [])
        }
        try:
            info = (obs.get("camera_info") or {})[self.depth_topic]
            depth = np.asarray(obs["depths"][self.depth_topic], dtype=np.float32)
            pts = self._points_world(depth, np.asarray(info["intrinsics"], float),
                                     np.asarray(info["extrinsics"], float))
            seg, _ = self._foreground_seg(pts, depth)
            centroids = {}
            for lab in np.unique(seg):
                if lab == 0:
                    continue
                mask = seg == lab
                centroids[int(lab)] = np.nanmean(
                    np.where(mask[..., None], pts, np.nan), axis=(0, 1)
                )
            for idx, k in enumerate(snapshot["keypoints"]):
                lab = k.get("_cluster")
                if lab is not None and int(lab) in centroids and np.isfinite(centroids[int(lab)]).all():
                    keypoints[idx] = centroids[int(lab)]
        except Exception:  # noqa: BLE001 跟踪失败退回快照位置
            pass
        for idx, k in enumerate(snapshot["keypoints"]):
            name = k.get("object")
            if name in live_regions:
                keypoints[idx] = live_regions[name]
        obstacles = np.asarray(list(live_regions.values()) or [], dtype=float).reshape(-1, 3)
        return {"ee_pose": ee, "arm_q": np.asarray(arm, dtype=float),
                "keypoints": keypoints, "obstacles": obstacles}
