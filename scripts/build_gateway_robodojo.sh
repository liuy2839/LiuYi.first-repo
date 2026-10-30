#!/usr/bin/env bash
# zy 上构建 gateway onefile（官方 adapter-forge-gateway 1.1.1 → PyInstaller）。
# 产物：$EVAL/dist/nodes/gateway-1.1.1-linux-x86_64.tar.gz + gateway.sha256
#
# 用法（在 zy 上）：bash scripts/build_gateway_robodojo.sh
set -euo pipefail

EVAL="${EVAL:-/data0/liuyi/forge_robodojo_eval}"
GW_SRC=$EVAL/gateway_src
VENV=$EVAL/gateway_venv
DIST=$EVAL/dist/nodes
mkdir -p "$DIST"

if [ ! -d "$GW_SRC/adapter-forge-gateway" ]; then
  mkdir -p "$GW_SRC"
  git clone --branch v1.1.1 --depth 1 \
    https://github.com/PhyAgentOS/adapter-forge-gateway.git \
    "$GW_SRC/adapter-forge-gateway" 2>/dev/null ||
  git clone --depth 1 https://github.com/PhyAgentOS/adapter-forge-gateway.git \
    "$GW_SRC/adapter-forge-gateway"
fi
cd "$GW_SRC/adapter-forge-gateway"

if [ ! -x "$VENV/bin/python" ]; then
  /data0/liuyi/miniconda3/envs/paos/bin/python -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
# 仓库是 [tool.uv] package = false（非安装包，无 build-system）：只装依赖与
# PyInstaller，运行时用 --paths src 让 main.py 找到 src/forge_gateway。
"$VENV/bin/pip" install -q \
  "click>=8.3.3,<9" "dora-rs==0.4.1" "fastapi>=0.141.1,<1" "uvicorn[standard]>=0.30,<1" \
  "forge-msgs==1.3.0" "opencv-python>=4.10,<5" "pillow>=12.3,<13" "pyyaml>=6,<7" \
  pyinstaller
"$VENV/bin/python" -c "import sys; sys.path.insert(0, 'src'); import forge_gateway; print('gateway import ok')"

export TMPDIR=$EVAL/tmp
mkdir -p "$TMPDIR"
# forge_gateway 是源码布局（pyproject [tool.uv] package=false，未装成包），
# --collect-all 找不到发行版数据 → 必须 --add-data 显式把 resources 打进
# _MEIPASS/forge_gateway/resources。缺了它 gateway 启动即死：
# 'resources/static does not exist'。
"$VENV/bin/pyinstaller" --onefile --clean -y \
  --name gateway \
  --paths src \
  --add-data "$PWD/src/forge_gateway/resources:forge_gateway/resources" \
  --distpath "$DIST/_gw_build" \
  --workpath "$TMPDIR/gw_work" \
  --specpath "$TMPDIR" \
  main.py >/dev/null

rm -f "$DIST/gateway-1.1.1-linux-x86_64.tar.gz"
tar -C "$DIST/_gw_build" -czf "$DIST/gateway-1.1.1-linux-x86_64.tar.gz" gateway
sha256sum "$DIST/gateway-1.1.1-linux-x86_64.tar.gz" | awk '{print $1}' > "$DIST/gateway.sha256"
echo "gateway built: $(cat "$DIST/gateway.sha256")"
"$DIST/_gw_build/gateway" --version 2>/dev/null | head -2 || true
rm -rf "$DIST/_gw_build"
