# ReKep policy on RoboDojo（Skill 数据平面 profile）

两节点 dataflow：RoboDojo benchmark（Isaac Sim，环境由宿主提供）↔ ReKep 策略节点
（`rekep_policy`，单文件制品，统一策略契约）。

## 节点与环境要求

| 组件 | 来源 | 说明 |
| --- | --- | --- |
| `robodojo_benchmark` | 宿主 `${ROBODOJO_ENV}/bin/python` + `${ROBODOJO_BENCHMARK_DIR}` | Isaac Sim 环境，不随 Skill 分发 |
| `rekep_policy` | Skill 制品 `rekep_policy-<ver>-linux-x86_64.tar.gz` | 通过 `${FORGE_RUNTIME_BIN}` 解析 |
| reference 算法源码 | 资产 `${REKEP_REF_RUNTIME}` | `subgoal_solver/path_solver/utils/transform_utils` 等 |
| 策略 profile | 资产 `rekep_profile.arx_x5_ee.json` | 动作分量名与评测端 `action_component_names()` 逐字一致 |

## 环境变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `ROBODOJO_ENV` | 是 | Isaac Sim conda 环境根（如 `/data0/liuyi/miniconda3/envs/RoboDojo`） |
| `ROBODOJO_ROOT` | 是 | RoboDojo 仓库根 |
| `ROBODOJO_BENCHMARK_DIR` | 是 | forge 的 robodojo benchmark 节点目录（含 main.py/_core/adapters） |
| `ROBODOJO_ASSETS_DIR` | 是 | RoboDojo `Assets` 目录（X5 URDF 在此） |
| `REKEP_PROFILE` | 是 | 策略 profile JSON 的绝对路径 |
| `REKEP_REF_RUNTIME` | 是 | reference runtime 目录绝对路径 |
| `REKEP_IK` | 否 | `x5`（默认，真实 IK）或 `stub` |
| `REKEP_PLAN_SOURCE` | 否 | `template`（默认）或 `vlm` |
| `REKEP_DINOV2_REPO` / `REKEP_DINOV2_WEIGHTS` | 否 | 仅 `REKEP_PERCEPTION=dino` 时需要 |
| `TMPDIR` / `TEMP` / `TMP` | 是 | 共享机上 IsaacLab 日志目录需指向私有路径 |

## 运行

```bash
dora run dataflow.yaml
```

结果写入 `config.rekep.yaml` 的 `result_dir`（默认 `results_rekep/`），
每集产出 `episode_end_calls/layout_id/score/video` 指标。
