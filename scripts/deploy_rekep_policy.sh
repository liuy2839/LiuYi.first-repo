#!/usr/bin/env bash
# 部署 ReKep 策略侧代码（forge_policy SDK + rekep_policy 节点）到评测机，并准备运行环境。
#
# 做 4 件事（幂等）：
#   1. rsync 本仓库 policy/ 与 nodes/policies/rekep/ 到 <REMOTE_REKEP_ROOT>
#   2. 建策略 venv（默认用 RoboDojo conda 的 python3.11）+ 安装依赖
#   3. 准备 reference 算法运行时（subgoal/path solver），pin 到 c532a28
#   4. 跑策略侧单测做冒烟
#
# 用法：
#   bash scripts/deploy_rekep_policy.sh
# 可用环境变量覆盖：
#   ZY=zy  REMOTE_REKEP_ROOT=/data0/liuyi/rekep_policy_dev
#   ROBODOJO_PYTHON=/data0/liuyi/miniconda3/envs/RoboDojo/bin/python3
set -euo pipefail

ZY="${ZY:-zy}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${REMOTE_REKEP_ROOT:-/data0/liuyi/rekep_policy_dev}"
REF_PIN="${REKEP_REF_PIN:-c532a28}"

echo "== 1/4 同步策略代码 → ${ZY}:${REMOTE_ROOT}"
rsync -a --delete \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
    --exclude 'dist' --exclude 'build' \
    "${SRC_DIR}/policy/" "${ZY}:${REMOTE_ROOT}/policy/"
rsync -a --delete \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
    --exclude 'dist' --exclude 'build' \
    "${SRC_DIR}/nodes/policies/rekep/" "${ZY}:${REMOTE_ROOT}/nodes/policies/rekep/"

echo "== 2/4 准备策略 venv 与依赖"
ssh "${ZY}" bash -s <<REMOTE
set -euo pipefail
ROOT="${REMOTE_ROOT}"
PY_BASE="\${ROBODOJO_PYTHON:-/data0/liuyi/miniconda3/envs/RoboDojo/bin/python3}"
VENV="\${ROOT}/.venv-policy"
if [ ! -x "\${VENV}/bin/python" ]; then
    "\${PY_BASE}" -m venv "\${VENV}"
fi
"\${VENV}/bin/pip" install -q --upgrade pip
# 基础依赖（每次幂等安装，缺什么补什么）
"\${VENV}/bin/pip" install -q "dora-rs==0.4.1" numpy scipy pyarrow pyyaml pillow pydantic \
    scikit-learn kmeans-pytorch numba opencv-python-headless tqdm mujoco
# torch 只在缺失时装（CPU 版体积大）
"\${VENV}/bin/python" -c "import torch" 2>/dev/null || \
    "\${VENV}/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"\${VENV}/bin/python" - <<'PY'
import dora, numpy, scipy, pyarrow, yaml, sklearn, kmeans_pytorch, numba, cv2, mujoco, torch
print("deps ok: torch", torch.__version__, "| mujoco", mujoco.__version__)
PY
REMOTE

echo "== 3/4 准备 reference 算法运行时（pin ${REF_PIN}）"
ssh "${ZY}" bash -s <<REMOTE
set -euo pipefail
ROOT="${REMOTE_ROOT}"
REF="\${ROOT}/third_party/rekep-real-plugin"
if [ ! -d "\${REF}/runtime" ]; then
    mkdir -p "\${ROOT}/third_party"
    git clone https://github.com/baiyu858/PhyAgentOS-rekep-real-plugin.git "\${REF}" 2>/dev/null || true
fi
if [ -d "\${REF}/.git" ]; then
    ( cd "\${REF}" && git checkout -q "${REF_PIN}" && git log --oneline -1 )
else
    echo "警告：reference runtime 缺失（${REF}）。请在可访问 GitHub 的机器上放置后重试。" >&2
fi
test -f "\${REF}/runtime/subgoal_solver.py" && echo "reference runtime ok"
REMOTE

echo "== 4/4 策略侧单测冒烟"
ssh "${ZY}" bash -s <<REMOTE
set -euo pipefail
ROOT="${REMOTE_ROOT}"
export PYTHONPATH="\${ROOT}/policy/src:\${ROOT}/nodes/policies/rekep/src:\${ROOT}/nodes/policies/rekep"
"\${ROOT}/.venv-policy/bin/python" -m pytest "\${ROOT}/policy/tests" "\${ROOT}/nodes/policies/rekep/tests" \
    -q -p no:cacheprovider
REMOTE

echo "完成：策略侧已同步并自检通过（${REMOTE_ROOT}）"