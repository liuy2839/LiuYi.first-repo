# 新策略接入 RoboDojo 评测链路指南

> 目标读者：训练出新策略（policy/skill）、想在 RoboDojo 上跑分的同学。
> 评测链路已接入 PhyAgentOS（PAOS）：`robodojo-bench` Skill 起来之后，
> `paos agent` 一句话就能跑批出结果。本文讲你的策略怎么接进来。

## 全景

```
paos agent ──HTTP── gateway ──dora── bench_tool(py3.12, 工具面)
                                      │ bench_command / benchmark_status|result|capabilities
                              bench_data(py3.10, Isaac Sim 数据面, _core runner)
                                      │ proprio_state / image/* / policy_command
                                policy 节点 ← 你的策略在这里
```

- **数据面协议不变**：观测驱动 1:1 锁步（发观测→收 action→step），这是已和
  官方 runner 对齐过（45/50 vs 46/50, Fisher p=1.0）的口径。
- 你的策略 = dataflow 里的一个 policy 节点 + skill.yaml 的一个 profile。

## 路径 A：forge_policy SDK（推荐，ReKep 即此路径）

实现 4 个方法 + 一份契约，其余（dora 收发、帧聚合、命令分发、chunk_end
哨兵、policy_status 上报）SDK 全包：

```python
from forge_policy import Policy, PolicyContract, PolicyRuntime

class MyPolicy(Policy):
    def reset_episode(self): ...          # episode 边界：清缓存/重置内部状态
    def set_instruction(self, text): ...  # 接收任务指令
    def step(self, obs) -> np.ndarray | None: ...  # 一帧观测→一个动作（或 None 丢弃）
    def shutdown(self): ...

contract = PolicyContract(
    action_names=[...],     # 必须与 adapter.action_component_names() 逐字一致（ARX X5 joint 模式 14 维）
    cameras=["image/head", "image/left_wrist", "image/right_wrist"],
    # depth_cameras / seg_cameras / scene_objects / camera_info 按需声明
    emit_chunk_end=True,    # RoboDojo 必须开：chunk 末动作补 is_episode_end 查询
)
PolicyRuntime(MyPolicy(), contract).run()
```

参考实现：`packages/nodes/policies/rekep/src/rekep_policy/`（main.py 装配 +
core.py 算法 + profile.py 动作名映射）。

## 路径 B：裸 policy 节点（需要完全控制 rollout 节奏时）

参照 `packages/nodes/benchmarks/robodojo/policy_node.py`（G05 的做法：
逐字复刻官方 rollout 节奏）。要自己处理的硬约束：

1. **观测驱动，绝不 tick 驱动**——tick 只当心跳，在 tick 上推理直接错。
2. `policy_command` 输入通道 **queue_size ≥ 32**（episode 边界连发两条命令）。
3. `JointCommand.name` 与 `action_component_names()` **逐字一致**，否则整集
   判失败（`action_name_mismatch`，这是保护不是刁难）。
4. 图像一律 RGB；BGR→RGB 的遗漏是最常见的静默 bug。
5. 动作维度走 `XPolicyLab.utils.process_data.get_robot_action_dim_info(env_cfg_type)`，
   不要写死（ARX X5 joint 模式 = 14，不是想当然的 7）。
6. policy 与 benchmark **必须不同 Python 环境**（websockets 版本钉死冲突）。
7. chunk 最后一个动作的 `effort[0]` 填 `CHUNK_END_SENTINEL`（=1.0，见
   `forge_policy` SDK；自己写节点时对齐）。

## 加一个 profile 的步骤

1. 策略代码放进 `packages/nodes/policies/<你的策略>/`（路径 A）或复用
   `policy_node.py` 换 env（路径 B）。
2. `skill-src/robodojo-bench/profiles/<你的profile>/`：拷贝 `g05/` 三件
   （dataflow.yaml / gateway.yaml 可原样；bench_config.yaml 改 policy_id）。
3. `skill.yaml`：profiles 下加一段（environment 换成你的 python/权重路径；
   required_binaries 里 policy_node 的 launcher 同一个即可）。
4. `bash scripts/deploy_robodojo_skill.sh`（同步→构建→安装一条龙）。
5. 冒烟：`paos skill start robodojo-bench --profile <你的profile>` → 单集
   （`task_ids=[0], init_state_ids=[0], max_steps=100`）→ 对着结果 JSON 检查
   `termination` 与 `num_steps` 是否合理。
6. 正式跑：`paos agent -m "用 robodojo-bench 跑 <suite> 全量评测并汇报成功率"`。

## 结果在哪、怎么读

- 权威文件：`bench_config.yaml` 的 `result_dir` 下
  `<suite>_<invocation_id>.json`（schema `benchmark_execution_result_v1`），
  含每集 success/termination/num_steps/score/layout_id 与视频路径。
- ToolResult 七字段：`run_id / status / completed_episodes / total_episodes /
  success_rate / result_file / failure_summary`（status 里 `partial` = stop
  落停时已有完成集）。
- 对照口径：`reproducibility` 字段（env_cfg_type / action_type /
  init_states_per_task / seed_is_layout_id）标明跑法，跨策略比较先核这个。

## 常见坑速查

| 症状 | 原因 |
|---|---|
| 每集都 `action_name_mismatch` | 动作分量名不一致（看错误信息里期望 vs 收到） |
| 分数恒 0 但机器人会动 | 分量顺序错位 / BGR 图像 / 维度写死 |
| 双向挂死无报错 | 跨通道乱序死锁（数据面有 20s 看门狗重发，若仍挂查 policy 是否吞观测） |
| 首集分数异常低 | 懒加载首帧慢于 20s 看门狗（正常，看 obs_republished 计数） |
| 第一步就 success | 没跑 `run_reward()` 判定链（用官方 adapter 不会发生） |
