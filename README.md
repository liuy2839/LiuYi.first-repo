# PAOS 评测链路：RoboDojo 策略评测接入

本仓库整理 **RoboDojo 操作策略评测链路接入 PhyAgentOS (PAOS)** 的架构与实现，
供团队同学查看、部署，以及**把各自训练的新策略接入评测、跑出结果**。

- 评测环境：RoboDojo（Isaac Sim 仿真，ARX X5 双臂）
- 执行框架：Forge / Dora 数据流（`forge_runtime` 节点体系）
- 治理与编排：PhyAgentOS（Skill Bundle + HTTP Forge Tool API + AgentTask）

## 整体架构

```
paos agent / 评测脚本
   │  HTTP Forge Tool API (127.0.0.1:19021)
   ▼
gateway 节点  (forge-gateway 1.1.1, PyInstaller onefile)
   │  dora: tool_in / tool_out (ToolMessage)
   ▼
bench_tool 节点  (py3.12 venv: forge-tool 1.0.0 + forge-msgs 1.3.0 + dora-rs 0.4.1)
   │  endpoint benchmark.controller：describe=Query / run=Session(max_concurrency=1)
   │  dora: bench_command ↓   benchmark_status / benchmark_result / benchmark_capabilities ↑
   ▼
bench_data 节点  (RoboDojo conda env, Isaac Sim)  ←—— serve 循环（serve.py）
   │  观测→动作 1:1 锁步（proprio_state / image/*  ↔  action）
   ▼
policy 节点  (G05 裸节点 / ReKep 等经 forge_policy SDK；新策略接入点，见下文)
```

**双进程的原因**：forge-tool 要求 Python ≥3.12，而 RoboDojo/Isaac Sim 固定在
py3.10/3.11。工具面（bench_tool）与数据面（bench_data）因此拆成两个 dora 节点，
经 `bench_command` / `benchmark_status` / `benchmark_result` / `benchmark_capabilities`
四条 JSON 通道协作；观测定时锁步的数据面协议与裸 dora 跑法完全一致。

评测的调用语义（对齐 Forge Tool API）：

- `benchmark.describe`（Query）：返回能力快照（suite、tasks、camera_ids、joint_names、默认参数、护栏）
- `benchmark.run`（Session）：参数 `task_ids / init_state_ids / num_runs / max_steps / seed`，
  一个 Session = 一个批次；`stop` 在 episode 边界落停；`max_concurrency=1`
- 结果：`benchmark_execution_result_v1` JSON 落盘 `results_paos/<suite>_<invocation_id>.json`，
  同时以七字段 ToolResult 回传（run_id / status / completed_episodes / total_episodes /
  success_rate / result_file / failure_summary）

## 仓库结构

```
forge_robodojo_eval/robodojo/       评测节点源码（与服务器部署布局一致）
├── _core/                          通用评测核心（BenchmarkRunner + serve 支撑 session.py）
├── bench_tool/                     工具面：benchmark.controller 端点
├── serve.py                        数据面 serve 循环（bench_command ↔ status/result）
├── main.py                         入口：cfg.serve 分流 autorun / serve
├── adapters/robodojo_adapter.py    RoboDojo 适配层：官方 eval env ↔ _core 协议
│                                   （含 is_episode_end 副作用补偿与 reproducibility 口径）
├── policy_node.py                  G05 策略节点（裸节点路径示例）
├── config*.yaml / dataflow*.yaml   各策略的评测配置与裸 dora 拓扑
│                                   （config.rekep/dataflow.rekep 为 ReKep 的对应版本）
├── tests/                          单测（serve 状态机 / autorun 回归 / 消息 round-trip；
│                                   _stubs.py 优先真实依赖、缺失才回退桩）
└── NEW_POLICY_GUIDE.md             新策略接入指南（详见下节）
policy/                             策略侧 SDK：forge_policy
├── src/forge_policy/base.py        Policy 四方法抽象（reset_episode/set_instruction/step/shutdown）
├── src/forge_policy/contract.py    策略线协议（PolicyContract：观测 topic、动作分量、哨兵）
├── src/forge_policy/runtime.py     PolicyRuntime：帧聚合、命令分发、动作发送、状态上报
└── tests/                          协议与生命周期单测
nodes/policies/rekep/               策略实现：ReKep 关系关键点约束操作
├── rekep_alg/                      算法核心（关键点感知 / 约束规划 + AST 沙箱 / 执行状态机）
├── src/rekep_policy/               统一策略节点（episode 时序 core / 动作空间 profile /
│                                   IK 适配 / 感知适配 / 算法装配 / 入口 main）
├── rekep_profile.arx_x5_ee.json    ARX X5 机器人 profile（动作命名 + 关节限位）
├── skill/                          PAOS manifest 与 robodojo-rekep dataflow
└── tests/                          算法与节点单测
skill-src/robodojo-bench/           PAOS Skill Bundle 源
├── skill.yaml                      manifest v2（g05 / rekep 两个 profile，startup_timeout_s: 900）
├── SKILL.md                        Agent 使用说明 + AgentTask verification 合同
├── launchers/                      薄壳启动器（bench_data / bench_tool / policy / rekep_policy）
└── profiles/g05|rekep/             每 profile 的 dataflow / gateway / bench_config
                                   （g05：joint 动作；rekep：ee 动作 + scene_objects 感知）
scripts/                            构建 / 部署 / 探针与校验
├── build_gateway_robodojo.sh       gateway onefile（官方 adapter-forge-gateway）
├── build_nodes_skill_robodojo.sh   launcher 节点构建 + skill.yaml sha 回填 + 打包
├── build_skill_robodojo.sh         Skill 全量构建入口（节点制品 + bundle 打包）
├── deploy_robodojo_skill.sh        评测侧部署：代码同步 → venv 准备 → 构建 → 安装
├── deploy_rekep_policy.sh          策略侧部署：forge_policy/ReKep 同步 + 策略 venv
│                                   + reference 求解器运行时 + 单测冒烟
├── install_robodojo_skill.sh       paos skill install + 节点制品放置
├── gateway_probe_robodojo.py       Tool API 探针（tools→describe→run，支持 --smoke）
└── check_skill_yaml.py             Skill 树自洽校验（多 profile）
```

## 部署与使用

```bash
# 0) 前置：PAOS v1.1.1 conda env `paos`、RoboDojo conda env、dora-cli 0.4.1、
#    bench_tool venv（forge-tool 1.0.0 / forge-msgs 1.3.0 / dora-rs 0.4.1 / pyinstaller）
#    注意：所有路径以服务器实际部署为准（skill.yaml / profiles 内为示例机器路径，换机需替换）

# 1) 部署（评测侧 + 策略侧）
bash scripts/deploy_robodojo_skill.sh       # 评测代码/构建/安装一条龙
bash scripts/deploy_rekep_policy.sh         # ReKep 策略 venv + reference 求解器

# 2) 构建与安装（deploy 已包含；单独重跑时）
bash scripts/build_skill_robodojo.sh        # 节点制品 + Skill bundle

# 3) 启动（Isaac 冷启动需要数分钟；profile 已设 startup_timeout_s: 900）
dora up && paos skill start robodojo-bench --profile g05     # 或 --profile rekep

# 4) 探针
python scripts/gateway_probe_robodojo.py          # GET /tools + describe
python scripts/gateway_probe_robodojo.py --smoke  # 追加单集 run
```

## 新策略接入

目标：把你训练好的策略（VLA / 行为策略 / 自定义 skill 算法）接入评测链路跑分。

**路径 A：forge_policy SDK**（策略代码在本仓库 `policy/` + `nodes/policies/`）
实现 4 个方法 + 一份契约，dora 收发、帧聚合、命令分发、chunk_end 哨兵全部由
`forge_policy.PolicyRuntime` 托管：

```python
from forge_policy import Policy, PolicyContract, PolicyRuntime

class MyPolicy(Policy):
    def reset_episode(self): ...          # episode 边界：清缓存 / 重置内部状态
    def set_instruction(self, text): ...  # 接收任务指令
    def step(self, obs) -> np.ndarray | None: ...  # 一帧观测 → 一个动作
    def shutdown(self): ...

contract = PolicyContract(
    action_names=[...],          # 与 adapter.action_component_names() 逐字一致（ARX X5 joint = 14 维）
    cameras=["image/head", "image/left_wrist", "image/right_wrist"],
    emit_chunk_end=True,         # RoboDojo 必须开
)
PolicyRuntime(MyPolicy(), contract).run()
```

**路径 B：裸 policy 节点**——需要完全控制 rollout 节奏（如复刻官方 rollout 的
动作 chunk 节奏）时直接写 dora 节点，自己处理观测聚合与动作发送，参照
`forge_robodojo_eval/robodojo/policy_node.py`（G05 做法）。

**仓库内示例**
- G05：裸 policy 节点 `forge_robodojo_eval/robodojo/policy_node.py`，profile `profiles/g05/`
- ReKep：`nodes/policies/rekep/`（SDK 路径完整实现），launcher
  `launchers/rekep_policy_node.py` + profile `profiles/rekep/`
  （`action_type: ee` + `publish_scene_objects: true`）

**加 profile 的步骤**
1. 策略代码放进 `nodes/policies/<你的策略>/`（路径 A，参照 rekep）或复用
   `policy_node.py` 换 env（路径 B），准备对应 Python 环境（与 benchmark 侧
   **必须分环境**，websockets 版本冲突）；
2. 复制 `skill-src/robodojo-bench/profiles/g05/` 为 `profiles/<你的名字>/`，改
   `bench_config.yaml` 的 `policy_id / action_type`、`dataflow.yaml` 的 policy 节点 env；
3. `skill.yaml` 的 `profiles` 下注册新 profile，替换 environment 中的 python/权重路径；
   artifacts.nodes 增加对应 launcher 的 lock；
4. 重新打包安装（scripts 里的构建链），`paos skill start robodojo-bench --profile <你的名字>`。

硬约束与常见坑（详见 `NEW_POLICY_GUIDE.md`）：观测驱动（tick 只当心跳）、
`policy_command` 通道 queue_size ≥ 32、动作分量名逐字匹配、图像必须 RGB、
动作维度走 `get_robot_action_dim_info(env_cfg_type)`、chunk 末动作 effort 填
`CHUNK_END_SENTINEL`。

## 工程注意事项

1. **dora 节点 env 是替换语义**：launcher 必须自建完整环境（LD_LIBRARY_PATH / PYTHONPATH /
   CUDA），不能依赖父进程继承。
2. **`TMPDIR`**：isaaclab 的日志/缓存走 `tempfile.gettempdir()`；共享机 `/tmp/isaaclab`
   被他人占用会令 env 构造 `PermissionError`。launcher 已设私有 TMPDIR。
3. **`sys.path` 顺序**：部署目录里若残留旧版 `_core`，会遮蔽包内的新核心（表现为
   `serve: true` 不生效、静默走 autorun）。`main.py` 已按「包内优先」固定顺序。
4. **PyInstaller + 源码布局包**：`forge_gateway` 的 `[tool.uv] package=false`，
   `--collect-all` 无效，必须 `--add-data "$PWD/src/forge_gateway/resources:forge_gateway/resources"`。
5. **PAOS 默认 30s 健康超时**：Isaac 冷启动要数分钟 → profile 需 `startup_timeout_s: 900`。
6. **skill.yaml 的 sha256**：每次重建节点后从 `dist/nodes/*.sha256` 回填（占位符会被安装器拒绝）。
7. **bench_tool 以 `-m bench_tool.main` 运行**（相对 import 才成立）。
8. **dora-cli 0.4.1 端口隔离**：需 shim 以参数重写（`--coordinator-port` 等），环境变量不生效。