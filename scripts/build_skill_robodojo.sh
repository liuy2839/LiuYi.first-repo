#!/usr/bin/env bash
# robodojo-bench Skill 构建脚本（在 zy 服务器上跑；参照 rekep-skill-paos/scripts/build_skill.sh）。
#
# 前置：
#   - bench_tool venv（py3.12，装了 forge-tool[dora]==1.0.0 forge-msgs==1.3.0 dora-rs==0.4.0+）
#   - PhyAgentOS-core 检出（scripts/package_skill.py）
#   - gateway onefile 制品（forge-gateway 1.1.x，PyInstaller；或复用已登记的 gateway-1.1.0 tar）
#
# 产物：
#   dist/nodes/*.tar.gz          四个 node 制品（gateway 复用已有 tar）
#   dist/skills/robodojo-bench-0.1.0.tar.gz   可 paos skill install --local 的 bundle
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="${HERE}/../skill-src/robodojo-bench"
DIST="${HERE}/../dist"
NODES_DIR="${DIST}/nodes"
SKILL_OUT="${DIST}/skills"
PAOS_CORE="${PAOS_CORE:?请设置 PAOS_CORE 指向 PhyAgentOS-core 检出}"
# 已有 gateway 制品（tar.gz）路径；没有则给 GATEWAY_SRC 指源码现场 PyInstaller
GATEWAY_TAR="${GATEWAY_TAR:-}"
# bench_tool 的 py3.12 venv python（构建机也要能跑 PyInstaller）
BUILD_PY="${BUILD_PY:-python3.12}"

VERSION="0.1.0"
PLATFORM_TAG="linux-x86_64"

mkdir -p "${NODES_DIR}" "${SKILL_OUT}"

build_one() {  # build_one <launcher.py> <node_name>
    local launcher="$1" name="$2"
    local work
    work="$(mktemp -d)"
    "${BUILD_PY}" -m PyInstaller \
        --onefile --clean -y \
        --distpath "${work}" \
        --workpath "${work}/build" \
        --specpath "${work}" \
        --name "${name}" \
        "${launcher}" >/dev/null
    local tar="${NODES_DIR}/${name}-${VERSION}-${PLATFORM_TAG}.tar.gz"
    tar -C "${work}" -czf "${tar}" "${name}"
    local sha
    sha="$(sha256sum "${tar}" | cut -d' ' -f1)"
    rm -rf "${work}"
    echo "${name}: ${tar} sha256=${sha}"
    printf '%s %s\n' "${name}" "${sha}" > "${NODES_DIR}/${name}.sha256"
}

build_one "${SKILL_SRC}/launchers/bench_tool_node.py" bench_tool_node
build_one "${SKILL_SRC}/launchers/bench_data_node.py" bench_data_node
build_one "${SKILL_SRC}/launchers/policy_node.py" policy_node

# gateway：复用现成制品（版本与 sha 以制品为准，写回 skill.yaml）
if [[ -z "${GATEWAY_TAR}" ]]; then
    echo "GATEWAY_TAR 未设置：跳过 gateway 构建，skill.yaml 里保持占位" >&2
else
    cp -f "${GATEWAY_TAR}" "${NODES_DIR}/gateway-1.1.1-${PLATFORM_TAG}.tar.gz"
    sha="$(sha256sum "${NODES_DIR}/gateway-1.1.1-${PLATFORM_TAG}.tar.gz" | cut -d' ' -f1)"
    printf '%s %s\n' gateway "${sha}" > "${NODES_DIR}/gateway.sha256"
fi

# 用真实 sha256 重写 skill.yaml 的 artifacts（占位符替换）
SKILL_YAML="${SKILL_SRC}/skill.yaml"
python3 - "${SKILL_YAML}" "${NODES_DIR}" <<'PY'
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
}
for node, (prefix, sha_file) in mapping.items():
    sha_file = nodes_dir / sha_file
    if not sha_file.exists():
        print(f"skip {node}（无 sha 文件）")
        continue
    sha = sha_file.read_text().split()[1]
    # 该 node 段落里的 artifact_id 与 sha256 行
    pattern = re.compile(
        rf"(    {node}:\n      artifact_id: ){re.escape(prefix)}(\n(?:.*\n)*?      sha256: )\S+"
    )
    text, n = pattern.subn(rf"\g<1>{prefix}\g<2>{sha}", text)
    if n != 1:
        raise SystemExit(f"skill.yaml 重写失败：{node} 匹配 {n} 处（应为 1）")
yaml_path.write_text(text, encoding="utf-8")
print("skill.yaml artifacts 已写回真实 sha256")
PY

# 用 PAOS 官方打包器出确定性 bundle
"${BUILD_PY}" "${PAOS_CORE}/scripts/package_skill.py" "${SKILL_SRC}" --output-dir "${SKILL_OUT}"
echo "bundle: ${SKILL_OUT}/robodojo-bench-${VERSION}.tar.gz"
