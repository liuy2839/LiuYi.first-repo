# forge-policy

[English](README.md) | [简体中文](README_zh.md)

统一策略节点传输协议与 Policy Runtime SDK：任何策略实现 `Policy` 接口后，即可
接入 Forge 评测链路（当前验证目标：RoboDojo benchmark），无需改动评测端。

## Node、Endpoint 与 Tool API

| 项目 | 当前实现 |
| --- | --- |
| 生产 Node | 0（本包是 SDK/库，不是独立 Node；策略 Node 由使用方 + main 脚本组成） |
| Tool 支持 | 仅数据平面（proprio_state / image / depth / seg / policy_command → action / policy_status） |
| Endpoint 数 | 0 |
| operation 数 | 0 |
| 直接 HTTP Tool API | 0（由 dataflow 内消息承载） |
| Gateway Tool 数 | 0 |

## 协议摘要

- 输入：`proprio_state`(JointState)、`image/<cam>`(rgb8)、`depth/<cam>`(32FC1 米，可选)、
  `seg/<cam>`(16UC1，可选)、`policy_command`(queue_size ≥32)、`tick`(仅心跳)。
- 输出：`action`(JointCommand，name/position 与评测端期望逐字一致)、
  `policy_status`(JSON 状态机)。
- 生命周期：`start` → `set_instruction` → `reset_scene`(每 episode) → `stop/cancel`；
  观测驱动推理（一帧完整观测至多一个 action，与评测端 1:1 锁步一致）。
- 策略实现 `forge_policy.base.Policy`：`reset_episode/set_instruction/step/shutdown`。

## 开发与测试

```bash
cd packages/policy
uv venv && uv pip install --python .venv <forge-msgs 本地检出> numpy pyarrow pytest
.venv/Scripts/python -m pytest tests -q   # 或 linux 下 .venv/bin/python
```

无 GitLab 凭据的开发机：用本地 `forge-forge-msgs-v1.0.1/packages/msgs` path 源
（字段布局与根 uv.lock 锁定的 rev 一致）；目标部署环境可切回 git 源。
