#!/usr/bin/env bash
# zy 上构建 robodojo-bench 的三个 launcher 节点并打包 Skill（gateway 由
# build_gateway_robodojo.sh 先行产出）。产物：
#   $EVAL/dist/nodes/{bench_tool_node,bench_data_node,policy_node}-0.1.0-linux-x86_64.tar.gz
#   $EVAL/dist/skills/robodojo-bench-0.1.0.tar.gz
#
# 用法（在 zy 上）：PAOS_CORE=/data0/liuyi/PhyAgentOS-core bash scripts/build_nodes_skill_robodojo.sh
set -euo pipefail

EVAL="${EVAL:-/data0/liuyi/forge_robodojo_eval}"
PAOS_CORE="${PAOS_CORE:-/data0/liuyi/PhyAgentOS-core}"
BUILD_PY="${BUILD_PY:-$EVAL/bench_tool_venv/bin/python}"
SKILL_SRC="$EVAL/skill-src/robodojo-bench"
DIST="$EVAL/dist"
NODES_DIR="$DIST/nodes"
SKILL_OUT="$DIST/skills"
VERSION="0.1.0"
PLATFORM_TAG="linux-x86_64"

mkdir -p "$NODES_DIR" "$SKILL_OUT"
export TMPDIR=$EVAL/tmp
mkdir -p "$TMPDIR"

build_one() {  # build_one <launcher.py> <node_name>
    local launcher="$1" name="$2"
    local work
    work="$(mktemp -d)"
    "$BUILD_PY" -m PyInstaller --onefile --clean -y \
        --distpath "${work}" --workpath "${work}/build" --specpath "${work}" \
        --name "${name}" "${launcher}" >/dev/null
    local tar="${NODES_DIR}/${name}-${VERSION}-${PLATFORM_TAG}.tar.gz"
    rm -f "$tar"
    tar -C "${work}" -czf "$tar" "${name}"
    sha256sum "$tar" | cut -d' ' -f1 > "${NODES_DIR}/${name}.sha256"
    rm -rf "${work}"
    echo "${name}: $(cat "${NODES_DIR}/${name}.sha256")"
}

build_one "$SKILL_SRC/launchers/bench_tool_node.py" bench_tool_node
build_one "$SKILL_SRC/launchers/bench_data_node.py" bench_data_node
build_one "$SKILL_SRC/launchers/policy_node.py" policy_node
build_one "$SKILL_SRC/launchers/rekep_policy_node.py" rekep_policy_node

if [ ! -f "$NODES_DIR/gateway.sha256" ]; then
    echo "gateway.sha256 不存在：先跑 build_gateway_robodojo.sh" >&2
    exit 1
fi

# 用真实 sha256 重写 skill.yaml 的 artifacts（PLACEHOLDER_* 替换）
python3 - "$SKILL_SRC/skill.yaml" "$NODES_DIR" <<'PY'
import re
import sys
from pathlib import Path

yaml_path, nodes_dir = Path(sys.argv[1]), Path(sys.argv[2])
text = yaml_path.read_text(encoding="utf-8")
mapping = {
    "gateway": ("gateway-1.1.1-linux-x86_64", "gateway.sha256"),
    "bench_tool_node": ("bench_tool_node-0.1.0-linux-x86_64", "bench_tool_node.sha256"),
    "bench_data_node": ("bench_data_node-0.1.0-linux-x86_64", "bench_data_node.sha256"),
    "policy_node": ("policy_node-0.1.0-linux-x86_64", "policy_node.sha256"),
    "rekep_policy_node": (
        "rekep_policy_node-0.1.0-linux-x86_64",
        "rekep_policy_node.sha256",
    ),
}
for node, (prefix, sha_file) in mapping.items():
    sha_file = nodes_dir / sha_file
    if not sha_file.exists():
        raise SystemExit(f"缺 {sha_file}")
    sha = sha_file.read_text().strip()
    pattern = re.compile(
        rf"(    {node}:\n      artifact_id: ){re.escape(prefix)}(\n(?:.*\n)*?      sha256: )\S+"
    )
    text, n = pattern.subn(rf"\g<1>{prefix}\g<2>{sha}", text)
    if n != 1:
        raise SystemExit(f"skill.yaml 重写失败：{node} 匹配 {n} 处（应为 1）")
yaml_path.write_text(text, encoding="utf-8")
print("skill.yaml artifacts 已写回真实 sha256")
PY

# PAOS 官方打包器出确定性 bundle
"$BUILD_PY" "$PAOS_CORE/scripts/package_skill.py" "$SKILL_SRC" --output-dir "$SKILL_OUT"
echo "bundle: $(ls "$SKILL_OUT")"
