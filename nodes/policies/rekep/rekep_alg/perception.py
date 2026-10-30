"""rekep_alg.perception：DINOv2 关键点提议（纯函数，无仿真环境）。

移植自 rekep_sim/perception.py 的 perceive()：原来依赖 MujocoReKepEnv 提供
rgb/points/seg/objects_info；本模块改为显式参数输入，由协议层（rekep_policy
的感知适配）从评测观测构造。DINOv2 加载胶水复用 rekep_sim/refimpl 的方式
（本地 repo+权重，不打网）。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any

import numpy as np


def _ensure_refimpl():
    """把 reference runtime 加进 sys.path（纯 import 副作用，幂等）。

    PyInstaller 打包后源码目录不存在，必须用 REKEP_REF_RUNTIME 指向随包资产。
    """
    import sys

    explicit = os.environ.get("REKEP_REF_RUNTIME", "").strip()
    if explicit and os.path.isdir(explicit):
        explicit = os.path.abspath(explicit)
        if explicit not in sys.path:
            sys.path.insert(0, explicit)
        return explicit

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # 部署布局：rekep_alg/../third_party/rekep-real-plugin/runtime
        os.path.join(here, "..", "..", "third_party", "rekep-real-plugin", "runtime"),
        # 服务器部署布局：rekep_policy_dev/third_party/...
        os.path.join(here, "..", "..", "..", "..", "third_party", "rekep-real-plugin", "runtime"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            c = os.path.abspath(c)
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    # 已在 sys.path（如同进程其它模块已加）
    return None


# —— 与 rekep_sim/perception.py 相同的常量与缓存策略 ——
DEFAULT_PROPOSER_CONFIG = {
    "num_candidates_per_mask": 5,
    "min_dist_bt_keypoints": 0.06,
    "max_mask_ratio": 0.5,
    "device": "cpu",
    "seed": 0,
}

_PROPOSER_CACHE: dict[str, Any] = {}
_PROPOSER_LOCK = threading.Lock()


def _patch_kmeans(keypoint_proposal_module: Any) -> None:
    """有界 kmeans（照 rekep_sim/refimpl._patch_kmeans，防 NaN 死循环）。"""
    import kmeans_pytorch

    if getattr(kmeans_pytorch.kmeans, "_rekep_bounded", False):
        return

    def bounded_kmeans(X, num_clusters, distance="euclidean", tol=1e-4,
                       device=None, max_iter=100, **_ignored):
        import numpy as np
        import torch

        device = device or torch.device("cpu")
        pairwise = (kmeans_pytorch.pairwise_distance if distance == "euclidean"
                    else kmeans_pytorch.pairwise_cosine)
        X = X.float().to(device)
        if not torch.isfinite(X).all():
            raise RuntimeError("kmeans input contains non-finite values")
        num_samples = int(X.shape[0])
        k = int(min(num_clusters, num_samples))
        idx = np.random.choice(num_samples, k, replace=False)
        state = X[idx].clone()
        choice = torch.zeros(num_samples, dtype=torch.long, device=device)
        for _ in range(max(1, int(max_iter))):
            dis = pairwise(X, state)
            choice = torch.argmin(dis, dim=1)
            prev = state.clone()
            for cluster in range(k):
                selected = torch.nonzero(choice == cluster).squeeze(1).to(device)
                if selected.numel() == 0:
                    continue
                state[cluster] = X.index_select(0, selected).mean(dim=0)
            center_shift = torch.sum(torch.sqrt(torch.sum((state - prev) ** 2, dim=1)))
            if bool(torch.isfinite(center_shift)) and float(center_shift) ** 2 < tol:
                break
        return choice.cpu(), state.cpu()

    bounded_kmeans._rekep_bounded = True
    keypoint_proposal_module.kmeans = bounded_kmeans


def _local_dinov2(*_a: Any, **_k: Any):
    """torch.hub.load 替身：本地 repo + 权重（REKEP_DINOV2_* 环境变量）。"""
    import torch

    repo = os.environ.get("REKEP_DINOV2_REPO", "").strip()
    weights = os.environ.get("REKEP_DINOV2_WEIGHTS", "").strip()
    model = os.environ.get("REKEP_DINOV2_MODEL", "dinov2_vits14").strip()
    if not repo or not weights:
        raise RuntimeError("REKEP_DINOV2_REPO/WEIGHTS 必须指向本地 DINOv2 源码与权重")
    net = torch.hub.load(repo, model, source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    cleaned = {
        str(k).removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()
    }
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if len(missing) > 32 or len(unexpected) > 32:
        raise RuntimeError(f"checkpoint incompatible: missing={len(missing)}, unexpected={len(unexpected)}")
    return net


def _patch_torch_hub() -> None:
    import torch

    current = torch.hub.load
    if getattr(current, "_rekep_local_patch", False):
        return

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir) in ("facebookresearch/dinov2", "dinov2"):
            return _local_dinov2(repo_or_dir, model, *args, **kwargs)
        return current(repo_or_dir, model, *args, **kwargs)

    load._rekep_local_patch = True  # type: ignore[attr-defined]
    torch.hub.load = load  # type: ignore[assignment]


def _proposer(bounds_min, bounds_max, proposer_config: dict | None, seed: int):
    cfg = dict(DEFAULT_PROPOSER_CONFIG)
    if proposer_config:
        cfg.update(proposer_config)
    # reference _get_features 带 @torch.amp.autocast('cuda')；CPU 输入在 cuda
    # autocast 下会得到 NaN 特征。无 GPU 或未要求 GPU 时强制 cpu + 关闭
    # autocast 的 monkey patch（对 cuda 部署无影响：autocast 本来就是 cuda 的）。
    if cfg.get("device", "cpu") == "cpu":
        import torch

        _orig_autocast = torch.amp.autocast

        def _cpu_safe_autocast(device_type, *a, **k):
            if str(device_type) == "cuda" and not torch.cuda.is_available():
                return _orig_autocast("cpu", enabled=False)
            return _orig_autocast(device_type, *a, **k)

        if not getattr(torch.amp.autocast, "_rekep_cpu_safe", False):
            _cpu_safe_autocast._rekep_cpu_safe = True  # type: ignore[attr-defined]
            torch.amp.autocast = _cpu_safe_autocast  # type: ignore[assignment]
    cfg["bounds_min"] = list(bounds_min)
    cfg["bounds_max"] = list(bounds_max)
    cfg["seed"] = seed
    key = repr(sorted(cfg.items()))
    with _PROPOSER_LOCK:
        cached = _PROPOSER_CACHE.get(key)
        if cached is None:
            _ensure_refimpl()
            _patch_torch_hub()
            import keypoint_proposal  # reference runtime

            _patch_kmeans(keypoint_proposal)
            try:
                import torch

                torch.set_num_threads(int(os.environ.get("REKEP_TORCH_THREADS", "4")))
            except Exception:
                pass
            cached = keypoint_proposal.KeypointProposer(cfg)
            try:
                cached.mean_shift.n_jobs = 1
            except Exception:
                pass
            _PROPOSER_CACHE[key] = cached
    return cached


def keypoints_from_observation(
    rgb: np.ndarray,
    points: np.ndarray,
    seg: np.ndarray,
    *,
    bounds_min,
    bounds_max,
    objects: list[dict[str, Any]] | None = None,
    proposer_config: dict | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """一帧观测 → perception snapshot（与 rekep_sim.perception 输出同构）。

    rgb: HxWx3 uint8；points: HxWx3 float（米，世界系）；seg: HxW int 实例标签。
    objects: 物体元数据 [{name, movable, region, position_m, geom_kind?}]，
    用于 center/axis 关键点；缺省时跳过补充关键点。
    """
    proposer = _proposer(bounds_min, bounds_max, proposer_config, seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    kp, projected, meta = proposer.get_keypoints(rgb, points, seg, return_metadata=True)
    keypoints = []
    pixels = meta["candidate_pixels"]
    for i, p in enumerate(kp):
        keypoints.append({
            "index": i,
            "position_m": [float(x) for x in p],
            "pixel_uv": [int(pixels[i][1]), int(pixels[i][0])],
            "kind": "proposal",
        })
    if objects:
        for o in objects:
            keypoints.append({
                "index": len(keypoints),
                "position_m": [float(x) for x in o["position_m"]],
                "pixel_uv": [-1, -1],
                "object": o["name"],
                "kind": "center",
            })
    rgb_sha = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    observation_id = "sha256:" + hashlib.sha256(
        json.dumps({"rgb": rgb_sha, "kps": [k["position_m"] for k in keypoints]},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "rekep_alg.perception.v1",
        "observation_id": observation_id,
        "rgb_sha256": rgb_sha,
        "bounds_min": list(bounds_min),
        "bounds_max": list(bounds_max),
        "keypoints": keypoints,
        "objects": objects or [],
        "projected": projected,
    }
