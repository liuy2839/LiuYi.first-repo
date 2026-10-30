#!/usr/bin/env bash
# zy 上安装 robodojo-bench Skill 与节点制品（本地 resolver dev 路径，
# 仿 rekep-skill-paos/scripts/install_nodes_local.sh）。
#
# 用法（在 zy 上）：
#   source /data0/liuyi/miniconda3/etc/profile.d/conda.sh && conda activate paos
#   bash /data0/liuyi/forge_robodojo_eval/scripts/install_robodojo_skill.sh
set -euo pipefail

EVAL="${EVAL:-/data0/liuyi/forge_robodojo_eval}"
PAOS_HOME="${PAOS_HOME:-$HOME/.PhyAgentOS}"
VERSION="0.1.0"
NODES_ROOT="$PAOS_HOME/forge_runtime/nodes"
DIST="$EVAL/dist"

command -v paos >/dev/null || { echo "需要 conda env paos（paos 在 PATH）" >&2; exit 1; }

# 1) Skill bundle 安装（幂等）
paos skill install "$DIST/skills/robodojo-bench-$VERSION.tar.gz" --local --yes

# 2) 节点制品放置（镜像 NodeInstaller 产物）
place() { # node_id artifact_id entrypoint archive
    local node_id="$1" artifact_id="$2" entrypoint="$3" archive="$4"
    local target="$NODES_ROOT/$node_id/versions/$artifact_id"
    # 已存在则改名归档（可回退），不做删除
    if [ -d "$target" ]; then
        mv "$target" "${target}.bak.$(date +%s)"
        echo "archived old $node_id version dir"
    fi
    mkdir -p "$target"
    tar -xzf "$archive" -C "$target"
    chmod +x "$target/$entrypoint"
    local asha bsha
    asha="$(sha256sum "$archive" | awk '{print $1}')"
    bsha="$(sha256sum "$target/$entrypoint" | awk '{print $1}')"
    cat > "$target/.paos-node.json" <<EOF
{"schema_version":1,"node_id":"$node_id","artifact_id":"$artifact_id","artifact_type":"executable_tar_gz","entrypoint":"$entrypoint","archive_sha256":"$asha","binary_sha256":"$bsha"}
EOF
    echo "installed node $node_id ($artifact_id)"
}

place gateway gateway-1.1.1-linux-x86_64 gateway "$DIST/nodes/gateway-1.1.1-linux-x86_64.tar.gz"
place bench_tool_node "bench_tool_node-$VERSION-linux-x86_64" bench_tool_node \
    "$DIST/nodes/bench_tool_node-$VERSION-linux-x86_64.tar.gz"
place bench_data_node "bench_data_node-$VERSION-linux-x86_64" bench_data_node \
    "$DIST/nodes/bench_data_node-$VERSION-linux-x86_64.tar.gz"
place policy_node "policy_node-$VERSION-linux-x86_64" policy_node \
    "$DIST/nodes/policy_node-$VERSION-linux-x86_64.tar.gz"
place rekep_policy_node "rekep_policy_node-$VERSION-linux-x86_64" rekep_policy_node \
    "$DIST/nodes/rekep_policy_node-$VERSION-linux-x86_64.tar.gz"

paos skill inspect robodojo-bench || true
echo "INSTALL_DONE"
