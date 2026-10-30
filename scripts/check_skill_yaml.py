# -*- coding: utf-8 -*-
"""校验 robodojo-bench Skill 树的自洽性（支持多 profile：g05 / rekep）。

每个 profile：
- skill.yaml：required_binaries ⊆ artifacts.nodes；dataflow 引用的
  ${FORGE_RUNTIME_BIN}/<name> 都在 artifacts 里
- gateway.yaml：specs 的 (endpoint_id, operation) 唯一、providers 覆盖所有 endpoint、
  semantics 为 describe=query / run=session
- dataflow：桥接通道（bench_command / benchmark_status / benchmark_result /
  benchmark_capabilities）两边对齐；gateway 的 tool 路由通道按约定命名
- bench_config：serve 打开、suite 非空
- environment：覆盖该 profile 各节点需要的入口变量
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skill-src" / "robodojo-bench"

REQUIRED_ENV = {
    "g05": (
        "BENCH_DATA_PYTHON",
        "BENCH_DATA_SRC",
        "BENCH_TOOL_PYTHON",
        "BENCH_TOOL_SRC",
        "BENCH_POLICY_PYTHON",
    ),
    "rekep": (
        "BENCH_DATA_PYTHON",
        "BENCH_DATA_SRC",
        "BENCH_TOOL_PYTHON",
        "BENCH_TOOL_SRC",
        "BENCH_REKEP_PYTHON",
        "BENCH_REKEP_SRC",
    ),
}

rc = 0


def fail(msg):
    global rc
    rc = 1
    print("FAIL", msg)


skill = yaml.safe_load((SKILL / "skill.yaml").read_text(encoding="utf-8"))

assert skill["manifest_version"] == 2
for key in ("name", "version", "skill_document", "gateway_url", "required_tools", "profiles", "artifacts"):
    if key not in skill:
        fail(f"skill.yaml 缺 {key}")

artifacts = set(skill["artifacts"]["nodes"])

for name, profile in skill["profiles"].items():
    pdir = SKILL / Path(profile["dataflow"]).parent

    # --- gateway.yaml ---
    gw = yaml.safe_load((pdir / "gateway.yaml").read_text(encoding="utf-8"))
    spec_ids = {s["tool_id"] for s in gw["tools"]["specs"]}
    missing = set(skill["required_tools"]) - spec_ids
    if missing:
        fail(f"[{name}] required_tools 不在 gateway specs 里：{missing}")
    pairs = [(s["endpoint_id"], s["operation"]) for s in gw["tools"]["specs"]]
    if len(pairs) != len(set(pairs)):
        fail(f"[{name}] gateway specs 存在重复 (endpoint, operation)")
    prov = {p["endpoint_id"] for p in gw["tools"]["providers"]}
    if prov != {s["endpoint_id"] for s in gw["tools"]["specs"]}:
        fail(f"[{name}] providers 与 specs 的 endpoint 不一致：{prov}")
    sem = {s["operation"]: s["semantics"] for s in gw["tools"]["specs"]}
    if sem.get("describe") != "query" or sem.get("run") != "session":
        fail(f"[{name}] semantics 不符：{sem}")
    p = gw["tools"]["providers"][0]
    if p["input_id"] != "bench_tool_in" or p["output_id"] != "bench_tool_out":
        fail(f"[{name}] provider 通道名与约定不符：{p}")

    # --- dataflow ---
    df = yaml.safe_load((pdir / "dataflow.yaml").read_text(encoding="utf-8"))
    binaries = set(profile["required_binaries"])
    if not binaries <= artifacts:
        fail(f"[{name}] required_binaries 未在 artifacts.nodes 登记：{binaries - artifacts}")
    for node in df["nodes"]:
        path = node["path"]
        if path.startswith("${FORGE_RUNTIME_BIN}/"):
            bname = path.split("/", 1)[1]
            if bname not in artifacts:
                fail(f"[{name}] dataflow 节点 {node['id']} 引用未登记的 binary {bname}")
            if bname not in binaries:
                fail(f"[{name}] dataflow 引用 {bname} 但 required_binaries 未列出")

    outs = {n["id"]: set(n.get("outputs", [])) for n in df["nodes"]}
    bt = next(n for n in df["nodes"] if n["id"] == "bench_tool")
    bd = next(n for n in df["nodes"] if n["id"] == "bench_data")
    for ch in ("benchmark_status", "benchmark_result", "benchmark_capabilities"):
        if ch not in bt["inputs"] or ch not in outs.get("bench_data", set()):
            fail(f"[{name}] 桥接通道 {ch} 两边未对齐")
    if "bench_command" not in outs.get("bench_tool", set()):
        fail(f"[{name}] bench_tool 缺 bench_command 输出")
    gw_node = next(n for n in df["nodes"] if n["id"] == "gateway")
    if "bench_tool_in" not in gw_node["inputs"]:
        fail(f"[{name}] gateway 缺 bench_tool_in 输入")

    # --- bench_config ---
    cfg = yaml.safe_load((pdir / "bench_config.yaml").read_text(encoding="utf-8"))
    if not cfg.get("serve"):
        fail(f"[{name}] bench_config.yaml 未开 serve")
    if not cfg.get("suite"):
        fail(f"[{name}] bench_config.yaml 缺 suite")

    # --- environment ---
    env = profile.get("environment", {})
    for var in REQUIRED_ENV.get(name, ()):
        if var not in env:
            fail(f"[{name}] environment 缺 {var}")

    # --- assets 与 profile 文件一致 ---
    for asset in profile.get("required_assets", []):
        if not (SKILL / asset).exists():
            fail(f"[{name}] required_assets 缺文件：{asset}")

print("RC =", rc)
sys.exit(rc)