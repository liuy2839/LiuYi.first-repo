# ReKep Policy（RoboDojo 评测 profile）· 中文说明

[English](README.md) | [简体中文](README_zh.md)

本 Skill 把 ReKep 作为**数据平面策略节点**接入 RoboDojo 评测链路：不引入 Tool/Gateway
调用面，评测端按统一策略契约（观测进、`JointCommand` 出）直接消费策略输出。

## 节点与环境要求

| 组件 | 来源 | 说明 |
| --- | --- | --- |
| `robodojo_benchmark` | 宿主 `${ROBODOJO_ENV}/bin/python` + `${ROBODOJO_BENCHMARK_DIR}` | Isaac Sim 环境，不随 Skill 分发 |
| `rekep_policy` | Skill 制品 `rekep_policy-<ver>-linux-x86_64.tar.gz` | 通过 `${FORGE_RUNTIME_BIN}` 解析 |
| reference 算法源码 | 资产 `${REKEP_REF_RUNTIME}` | `subgoal_solver/path_solver/utils/transform_utils` 等 |
| 策略 profile | 资产 `rekep_profile.arx_x5_ee.json` | 动作分量名必须与评测端 `action_component_names()` 逐字一致 |

## 环境变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `ROBODOJO_ENV` | 是 | Isaac Sim conda 环境根目录 |
| `ROBODOJO_ROOT` | 是 | RoboDojo 仓库根 |
| `ROBODOJO_BENCHMARK_DIR` | 是 | forge 的 robodojo benchmark 节点目录（含 main.py、_core、adapters） |
| `ROBODOJO_ASSETS_DIR` | 是 | RoboDojo `Assets` 目录（X5 URDF） |
| `REKEP_PROFILE` | 是 | 策略 profile JSON 绝对路径 |
| `REKEP_REF_RUNTIME` | 是 | reference runtime 目录绝对路径 |
| `REKEP_TMP_DIR` | 是 | 私有临时目录（共享机上 IsaacLab 日志目录必须指向它） |
| `REKEP_IK` | 否 | `x5`（默认，真实 IK）或 `stub` |
| `REKEP_PLAN_SOURCE` | 否 | `template`（默认）或 `vlm` |
| `REKEP_PERCEPTION` | 否 | `scene_objects`（默认）或 `dino`（需深度与 DINOv2 资产） |

## 运行与验收

```bash
dora run dataflow.yaml
```

单集结果写入 `result_dir`（默认 `results_rekep/`）。验收看结果 JSON 的
`success_rate`、`episodes[].termination`、`mean_policy_latency_ms` 与录像文件；
不要以节点自身日志推断成功。

## 已知边界

- RoboDojo `arx_x5` 是双臂 4 瓶 + 交接任务；本策略只驱动左臂，右臂保持当前位姿。
- 深度注解器在部分 Isaac Sim 5.1 + 驱动组合上会触发 CUDA 崩溃；遇到时用
  `REKEP_PERCEPTION=scene_objects`。
- 规划默认模板；启用 VLM 需注入 `REKEP_VLM_*` 并设 `REKEP_PLAN_SOURCE=vlm`。