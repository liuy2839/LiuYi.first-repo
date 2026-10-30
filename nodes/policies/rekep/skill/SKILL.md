# ReKep Policy（RoboDojo 评测 profile）

本 Skill 把 ReKep 作为**数据平面策略节点**接入 RoboDojo 评测链路：不引入 Tool/Gateway
调用面，评测端按统一策略契约（观测进、`JointCommand` 出）直接消费策略输出。

## 激活后做什么

1. 启动 `profiles/robodojo-rekep/dataflow.yaml`：
   - `robodojo_benchmark`：宿主 Isaac Sim 环境，产出 `proprio_state`、`image/*`、
     `depth/*`、`scene_objects`、`camera_info`、`policy_command`
   - `rekep_policy`：Skill 制品节点，消费上述观测与控制命令，输出 `action`（16 维 ee 动作）
2. 评测端按 1:1 锁步驱动 episode；单集结束写入 `result_dir` 的结果 JSON 与录像。

## 使用规则

- **不要**修改 `rekep_policy` 的动作分量名或顺序：必须与评测端
  `action_component_names()` 逐字一致（见 `rekep_profile.arx_x5_ee.json`）。
- 策略只在收到完整一帧观测后产出一个动作；`tick` 仅作心跳。
- `reset_scene` 清空当前 episode 的计划与队列；新指令触发重新感知与规划。
- 求解失败或计划结束时会持续输出"保持位姿"动作，而不是停止发动作——保证评测端
  能走到步数上限并正常结算。
- 感知来源由 `REKEP_PERCEPTION` 选择：
  - `scene_objects`（默认）：评测端物体真值位姿 → 中心关键点，稳定但非视觉；
  - `dino`：深度反投影 + 连通域伪分割 + DINOv2 关键点，需要 `depth/*` 与
    `camera_info`，以及 `REKEP_DINOV2_*` 资产。
- 结果只以评测端 `ToolResult`/结果 JSON 为准，不要在对话里自行宣称成功率。

## 已知边界

- 当前 RoboDojo `arx_x5` 是双臂 4 瓶 + 交接任务；本策略只驱动左臂，右臂保持当前位姿。
- 深度注解器（`distance_to_image_plane`）在部分 Isaac Sim 5.1 + 驱动组合上会触发
  CUDA 崩溃；遇到时请用 `REKEP_PERCEPTION=scene_objects`。
- 规划默认 `template`；启用 VLM 需注入 `REKEP_VLM_*` 密钥并设置
  `REKEP_PLAN_SOURCE=vlm`。
