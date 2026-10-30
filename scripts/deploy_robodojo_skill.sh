#!/usr/bin/env bash
# robodojo-bench 部署到 zy 并安装（本地 Windows 在 Git Bash/WSL 里跑，或服务器上跑）。
#
# 做 4 件事：
#   1. rsync 改动后的评测代码到 zy:/data0/liuyi/forge_robodojo_eval/robodojo
#      （_core + robodojo 两棵子树；autorun 用法零破坏）
#   2. 建bench_tool venv（py3.12，仅首次；forge-tool/forge-msgs 从 forge stable/1.x 装）
#   3. 服务器上跑 build_skill_robodojo.sh（PyInstaller onefile + package_skill）
#   4. paos skill install --local + install_nodes_local（放 node 制品 + 回执）
set -euo pipefail

ZY="${ZY:-zy}"
EVAL_DIR="/data0/liuyi/forge_robodojo_eval"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${SRC_DIR}/forge_robodojo_eval/robodojo"

# 1) 同步代码（_core 与 robodojo）
rsync -a --delete \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
    "${BENCH_DIR}/_core/" "${ZY}:${EVAL_DIR}/robodojo/_core/"
rsync -a --delete \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
    --exclude '/bench_tool_venv' --exclude '/_core' \
    "${BENCH_DIR}/" "${ZY}:${EVAL_DIR}/robodojo/"
rsync -a "${SRC_DIR}/scripts/build_skill_robodojo.sh" "${ZY}:${EVAL_DIR}/"
rsync -a "${SRC_DIR}/skill-src/" "${ZY}:${EVAL_DIR}/skill-src/"

# 1b) 策略侧（forge_policy SDK + ReKep 节点 + venv + reference 运行时）
bash "${SRC_DIR}/scripts/deploy_rekep_policy.sh"

# 2) bench_tool venv（py3.12；幂等）
ssh "${ZY}" bash -s <<'REMOTE'
set -euo pipefail
EVAL_DIR=/data0/liuyi/forge_robodojo_eval
VENV="${EVAL_DIR}/bench_tool_venv"
if [ ! -x "${VENV}/bin/python" ]; then
    python3.12 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q "dora-rs==0.4.1" "pyarrow" "numpy" "pyyaml" "PyInstaller"
# forge-tool / forge-msgs：pip 没有 1.x 线时从 forge stable/1.x 源装
"${VENV}/bin/pip" install -q "forge-tool[dora]==1.0.0" "forge-msgs==1.3.0" || {
    [ -d "${EVAL_DIR}/forge-1x" ] || git clone --branch stable/1.x --depth 1 \
        https://github.com/PhyAgentOS/forge.git "${EVAL_DIR}/forge-1x"
    "${VENV}/bin/pip" install -q "${EVAL_DIR}/forge-1x/packages/tool" "${EVAL_DIR}/forge-1x/packages/msgs"
}
REMOTE

# 3) 服务器构建（PAOS_CORE 指向 core 检出；GATEWAY_TAR 可复用）
ssh "${ZY}" "cd ${EVAL_DIR} && PAOS_CORE=/data0/liuyi/PhyAgentOS-core BUILD_PY=${EVAL_DIR}/bench_tool_venv/bin/python bash build_skill_robodojo.sh"

# 4) 安装（paos 在 conda env paos 里）
ssh "${ZY}" bash -s <<'REMOTE'
set -euo pipefail
source /data0/liuyi/miniconda3/etc/profile.d/conda.sh
conda activate paos
EVAL_DIR=/data0/liuyi/forge_robodojo_eval
paos skill install "${EVAL_DIR}/dist/skills/robodojo-bench-0.1.0.tar.gz" --local --yes
# 本地 resolver：把 node 制品放进 PAOS 期望的位置并写回执（仿 rekep install_nodes_local.sh）
REMOTE
echo "部署脚本完成；install_nodes_local 与 skill start 见 M4 验收清单"
