"""rekep_alg.planner：约束生成（模板/VLM）+ digest 绑定计划（纯模块）。

移植自 rekep_sim/runtime.py 的 make_template_program/_program_from_text/
plan/_plan_vlm。VLM 路径依赖 reference runtime 的 constraint_generation
（同进程 sys.path 注入，见 perception._ensure_refimpl 的候选布局）。
沙箱原样复用 rekep_sim/sandbox.py 的语义（此处内置同实现，避免包间依赖）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
import uuid
from typing import Any, Callable, Iterable

import numpy as np


# ------------------------------------------------------------------ sandbox
# 与 rekep_sim/sandbox.py 相同的 AST 白名单（安全红线，保持逐条一致）
_FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.Import, ast.ImportFrom, ast.If, ast.While, ast.For, ast.AsyncFor,
    ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Lambda, ast.Global,
    ast.Nonlocal, ast.ClassDef, ast.Try, ast.Raise, ast.Assert, ast.Delete,
    ast.Yield, ast.YieldFrom, ast.Await, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.NamedExpr, ast.Match,
)
_DISALLOWED_CALL_NAMES = {
    "eval", "exec", "open", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
    "breakpoint", "help", "exit", "quit", "print",
}
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "float": float, "int": int, "round": round, "bool": bool, "pow": pow,
    "sorted": sorted, "math": math, "True": True, "False": False, "None": None,
}
_GRASP_FN = "get_grasping_cost_by_keypoint_idx"
_ALLOWED_BARE = set(_SAFE_BUILTINS) | {_GRASP_FN, "np"}
_FORBIDDEN_ATTRS = {
    "f_locals", "f_globals", "f_builtins", "func_globals", "gi_frame",
    "cr_frame", "__globals__", "__builtins__",
}


class ConstraintSecurityError(ValueError):
    pass


class ConstraintSyntaxError(ValueError):
    pass


def _reject(node: ast.AST, reason: str) -> None:
    raise ConstraintSecurityError(
        f"forbidden {type(node).__name__}: {reason} (line {getattr(node, 'lineno', '?')})"
    )


def _validate(tree: ast.AST) -> list[str]:
    defined: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            _reject(node, "syntax not allowed in constraints")
        if isinstance(node, ast.FunctionDef):
            defined.append(node.name)
            if node.decorator_list:
                _reject(node, "decorators are not allowed")
            args = node.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults:
                _reject(node, "only plain positional parameters are allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRS:
                _reject(node, f"attribute {node.attr!r} is forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            _reject(node, f"name {node.id!r} is forbidden")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in _DISALLOWED_CALL_NAMES or func.id not in _ALLOWED_BARE:
                    _reject(node, f"call to {func.id!r} is not allowed")
            elif isinstance(func, ast.Attribute):
                root = func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (isinstance(root, ast.Name) and root.id in {"np", "math"}):
                    _reject(node, "only np.* / math.* calls are allowed")
            else:
                _reject(node, "unsupported call target")
    if not defined:
        raise ConstraintSecurityError("constraint program defines no functions")
    return defined


def _grasp_fn(grasped: Iterable[int] | None):
    g = {int(x) for x in (grasped or [])}

    def get_grasping_cost_by_keypoint_idx(keypoint_idx):  # noqa: N802
        return 0 if int(keypoint_idx) in g else 1

    return get_grasping_cost_by_keypoint_idx


def load_functions(source: str, *, grasped_keypoints: Iterable[int] | None = None):
    if not isinstance(source, str) or not source.strip():
        raise ConstraintSyntaxError("constraint source is empty")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ConstraintSyntaxError(f"invalid Python: {exc}") from exc
    _validate(tree)
    ns: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS), "np": np, "math": math,
        _GRASP_FN: _grasp_fn(grasped_keypoints),
    }
    exec(compile(tree, "<rekep_constraints>", "exec"), ns, ns)  # noqa: S102 白名单后执行
    # 只返回程序定义的 stage 约束函数；沙箱注入的 grasp-cost 助手不算
    fns = [v for k, v in ns.items()
           if not k.startswith("__") and callable(v) and k != _GRASP_FN and k not in ("np", "math")]
    if not fns:
        raise ConstraintSecurityError("constraint program produced no callable")
    return fns


# ------------------------------------------------------------------- plan
def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parse_list(text: str, name: str, num_stages: int) -> list[int]:
    m = re.search(rf"{name}\s*=\s*\[([^\]]*)\]", text)
    if not m:
        return [-1] * num_stages
    items = [x.strip() for x in m.group(1).split(",") if x.strip()]
    vals = [int(x) for x in items]
    if len(vals) != num_stages:
        raise ValueError(f"{name} length {len(vals)} != num_stages {num_stages}")
    return vals


def _append_group(stage_funcs: list[dict[str, str]], name: str, lines: list[str]) -> None:
    m = re.match(r"stage(\d+)_(subgoal|path)", name)
    if not m:
        return
    idx, kind = int(m.group(1)) - 1, m.group(2)
    if 0 <= idx < len(stage_funcs):
        stage_funcs[idx][kind] = (stage_funcs[idx][kind] + "\n" + "\n".join(lines)).strip()


def _program_from_text(text: str, snapshot: dict, instruction: str, *, source: str,
                       success: dict | None = None, vlm_trace: str | None = None) -> dict:
    num_stages = int(re.search(r"num_stages\s*=\s*(\d+)", text).group(1))
    grasp = _parse_list(text, "grasp_keypoints", num_stages)
    release = _parse_list(text, "release_keypoints", num_stages)
    stage_funcs: list[dict[str, str]] = [{"subgoal": "", "path": ""} for _ in range(num_stages)]
    name, current = None, []
    for line in text.splitlines():
        if line.startswith("def "):
            if name is not None:
                _append_group(stage_funcs, name, current)
            name = line.split("(")[0].replace("def ", "").strip()
            current = [line]
        elif name is not None:
            current.append(line)
            if line.strip().startswith("return "):
                _append_group(stage_funcs, name, current)
                name, current = None, []
    core = {
        "instruction": instruction,
        "observation_id": snapshot.get("observation_id"),
        "num_stages": num_stages,
        "grasp_keypoints": grasp,
        "release_keypoints": release,
        "keypoints": snapshot["keypoints"],
        "init_keypoint_positions": [k["position_m"] for k in snapshot["keypoints"]],
        "stages": [
            {
                "stage": i + 1,
                "subgoal_constraints": stage_funcs[i]["subgoal"],
                "path_constraints": stage_funcs[i]["path"],
                "grasp_keypoint": grasp[i],
                "release_keypoint": release[i],
            }
            for i in range(num_stages)
        ],
        "source": source,
    }
    if success is not None:
        core["success"] = success
    if vlm_trace:
        core["vlm_trace"] = vlm_trace
    core["plan_id"] = f"plan_{uuid.uuid4().hex}"
    core["created_timestamp_ns"] = str(time.time_ns())
    core["plan_digest"] = digest({k: v for k, v in core.items() if k != "plan_digest"})
    return core


def _role_index(keypoints: list[dict], object_name: str) -> int:
    for k in keypoints:
        if k.get("object") == object_name:
            return int(k["index"])
    return 0 if object_name == "place_zone" else len(keypoints) - 1


def _select_target(objects: list[dict], instruction: str) -> str | None:
    """从 movable 里挑任务目标：优先名字含 target，其次与指令词匹配，最后第一个。"""
    movables = [o for o in objects if o.get("movable")]
    if not movables:
        return None
    for o in movables:
        if "target" in o["name"].lower():
            return o["name"]
    text = (instruction or "").lower()
    for o in movables:
        # "bottle_41_1" → 取类别前缀 "bottle"
        base = o["name"].split("_")[0].lower()
        if base and base in text:
            return o["name"]
    return movables[0]["name"]


def make_template_program(snapshot: dict, instruction: str) -> dict:
    """确定性模板。

    有 region：抓 movable → 移到 region → 释放（三段）。
    无 region（如 general_pickup）：抓目标 → 抬升 10cm（两段）。
    """
    kps = snapshot["keypoints"]
    objects = snapshot.get("objects") or []
    regions = [o for o in objects if o.get("region")]
    target = _select_target(objects, instruction) or "pick_cube"
    ci = _role_index(kps, target)

    if not regions:
        z0 = float(kps[ci]["position_m"][2])
        lift = z0 + 0.10
        prog = (
            "num_stages = 2\n"
            "def stage1_subgoal_constraint1(end_effector, keypoints):\n"
            f"    return np.linalg.norm(end_effector - keypoints[{ci}])\n"
            "def stage2_subgoal_constraint1(end_effector, keypoints):\n"
            f"    return np.maximum(0.0, {lift!r} - keypoints[{ci}][2])\n"
            "def stage2_subgoal_constraint2(end_effector, keypoints):\n"
            f"    return np.linalg.norm(end_effector - keypoints[{ci}])\n"
            "def stage2_path_constraint1(end_effector, keypoints):\n"
            f"    return get_grasping_cost_by_keypoint_idx({ci})\n"
            f"grasp_keypoints = [{ci}, -1]\n"
            "release_keypoints = [-1, -1]\n"
        )
        success = {"object": target, "mode": "lift", "min_delta_m": 0.10}
        return _program_from_text(prog, snapshot, instruction, source="template", success=success)

    place = regions[0]["name"]
    pi = _role_index(kps, place)
    prog = (
        "num_stages = 3\n"
        "def stage1_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(end_effector - keypoints[{ci}])\n"
        "def stage2_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(keypoints[{ci}] - keypoints[{pi}])\n"
        "def stage2_path_constraint1(end_effector, keypoints):\n"
        f"    return get_grasping_cost_by_keypoint_idx({ci})\n"
        "def stage3_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(keypoints[{ci}] - keypoints[{pi}])\n"
        f"grasp_keypoints = [{ci}, -1, -1]\n"
        f"release_keypoints = [-1, -1, {ci}]\n"
    )
    success = {"object": target, "target_region": place, "mode": "in_region", "tol": 0.06}
    return _program_from_text(prog, snapshot, instruction, source="template", success=success)


def plan(instruction: str, snapshot: dict, *, source: str = "template") -> dict:
    if source == "vlm":
        return _plan_vlm(instruction, snapshot)
    return make_template_program(snapshot, instruction)


def _plan_vlm(instruction: str, snapshot: dict) -> dict:
    """VLM 约束生成：走 reference runtime 的 ConstraintGenerator。"""
    import cv2

    from .perception import _ensure_refimpl

    _ensure_refimpl()
    import constraint_generation  # reference runtime

    gen = constraint_generation.ConstraintGenerator(
        {"model": "gpt-5.4", "temperature": 0.0, "max_tokens": 16384}
    )
    keypoints = snapshot["keypoints"]
    objects = snapshot.get("objects") or []
    guard = ["Use exact keypoint IDs from this table. Never remap ID meanings:"]
    for k in keypoints:
        guard.append(
            f"- id {k['index']}: object={k.get('object', 'unknown')}, "
            f"position_m={[round(float(x), 3) for x in k['position_m']]}"
        )
    for o in objects:
        kind = "movable object" if o.get("movable") else "region"
        guard.append(f"- {o['name']}: {kind}")
    guard_text = "\n".join(guard)
    orig = gen._build_prompt

    def guarded(image_path, instr):
        messages, text = orig(image_path, instr)
        suffix = "\n\n## Additional Hard Constraints\n" + guard_text + "\n"
        for m in messages:
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "text":
                        part["text"] += suffix
        return messages, text + suffix

    gen._build_prompt = guarded
    metadata = {
        "init_keypoint_positions": [k["position_m"] for k in keypoints],
        "num_keypoints": len(keypoints),
        "task_schema": {"keypoints": keypoints},
    }
    rgb = cv2.imread(snapshot["overlay_path"]) if snapshot.get("overlay_path") else None
    if rgb is None:
        raise RuntimeError("plan(source=vlm) requires an overlay image in the snapshot")
    task_dir = gen.generate(rgb, instruction, metadata)
    with open(f"{task_dir}/metadata.json") as f:
        meta = json.load(f)
    parts = []
    num_stages = int(meta["num_stages"])
    for i in range(1, num_stages + 1):
        for kind in ("subgoal", "path"):
            p = f"{task_dir}/stage{i}_{kind}_constraints.txt"
            try:
                with open(p) as f:
                    parts.append(f.read())
            except FileNotFoundError:
                pass
    text = "\n".join(parts)
    text += f"\nnum_stages = {num_stages}\n"
    text += f"grasp_keypoints = {meta['grasp_keypoints']}\n"
    text += f"release_keypoints = {meta['release_keypoints']}\n"
    return _program_from_text(
        text, snapshot, instruction, source="vlm",
        vlm_trace=f"{task_dir}/vlm_trace.json",
    )
