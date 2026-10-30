#!/usr/bin/env bash
# 构建 rekep_policy 单文件节点制品 + Skill bundle（PAOS manifest v2）。
# 运行环境：Linux x86_64，Python >=3.11 虚拟环境（已装 torch CPU/mujoco/numba 等）。
# 用法： bash build_node.sh [--skip-tests]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"            # Paos 仓库根
POLICY_SRC="${POLICY_SRC:-$REPO/policy/src}"
REKEP_SRC="${REKEP_SRC:-$ROOT/src}"
VERSION="${REKEP_POLICY_VERSION:-0.1.0}"
PLATFORM=linux
ARCH=x86_64
DIST="$ROOT/dist"
SKILL_DIR="$ROOT/skill"
PY="${PYTHON:-python}"

cd "$ROOT"
if [ "${1:-}" != "--skip-tests" ]; then
    PYTHONPATH="$POLICY_SRC:$REKEP_SRC:$ROOT" "$PY" -m pytest tests -q -p no:cacheprovider
fi

rm -rf "$DIST" build
mkdir -p "$DIST"

PYTHONPATH="$POLICY_SRC:$REKEP_SRC:$ROOT" "$PY" -m PyInstaller \
    --noconfirm --clean --onefile \
    --name rekep_policy \
    --paths "$POLICY_SRC" --paths "$REKEP_SRC" --paths "$ROOT" \
    --hidden-import rekep_alg.perception \
    --hidden-import rekep_alg.planner \
    --hidden-import rekep_alg.executor \
    --hidden-import rekep_policy.perception_adapters \
    --hidden-import rekep_policy.assembly \
    --hidden-import rekep_policy.ik_adapters \
    --hidden-import dora \
    --hidden-import forge_msgs \
    --hidden-import yaml \
    --hidden-import scipy.optimize \
    --hidden-import scipy.interpolate \
    --hidden-import scipy.ndimage \
    --hidden-import sklearn.cluster \
    --hidden-import kmeans_pytorch \
    --hidden-import numba \
    --hidden-import cv2 \
    --hidden-import mujoco \
    --collect-submodules forge_msgs \
    --collect-submodules rekep_alg \
    --collect-submodules rekep_policy \
    entrypoint.py

BIN="$DIST/rekep_policy"
[ -x "$BIN" ] || { echo "构建失败：$BIN 不存在" >&2; exit 1; }

ARCHIVE="$DIST/rekep_policy-${VERSION}-${PLATFORM}-${ARCH}.tar.gz"
tar -czf "$ARCHIVE" -C "$DIST" rekep_policy
SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
SIZE=$(wc -c < "$ARCHIVE")

echo "=== 制品 ==="
echo "archive: $ARCHIVE"
echo "sha256:  $SHA"
echo "bytes:   $SIZE"

# 生成 skill.yaml（manifest v2）
cat > "$SKILL_DIR/skill.yaml" <<EOF
manifest_version: 2
name: rekep-policy
version: "$VERSION"
description: ReKep policy node (unified Forge policy contract) evaluated on RoboDojo.
skill_document: SKILL.md
required_tools: []
profiles:
  robodojo-rekep:
    dataflow: profiles/robodojo-rekep/dataflow.yaml
    required_binaries:
      - rekep_policy
    required_assets:
      - profiles/robodojo-rekep/config.rekep.yaml
    required_environment:
      - ROBODOJO_ENV
      - ROBODOJO_ROOT
      - ROBODOJO_BENCHMARK_DIR
      - ROBODOJO_ASSETS_DIR
    environment:
      PYTHONPATH: ""
artifacts:
  resolver: local
  nodes:
    rekep_policy:
      artifact_id: rekep_policy-${VERSION}-${PLATFORM}-${ARCH}
      version: "$VERSION"
      platform: $PLATFORM
      arch: $ARCH
      artifact_type: executable_tar_gz
      entrypoint: rekep_policy
      sha256: $SHA
EOF
echo "=== skill.yaml 已生成：$SKILL_DIR/skill.yaml ==="
sed -n '1,45p' "$SKILL_DIR/skill.yaml"