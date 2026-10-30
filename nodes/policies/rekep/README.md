# rekep_policy

[English](README.md) | [简体中文](README_zh.md)

Unified policy node that plugs ReKep (relational keypoint constraint manipulation)
into the Forge/RoboDojo evaluation chain. To the evaluator it is just another policy
node (observations in, `JointCommand` out); internally it runs perception → planning
→ staged solve-and-execute.

## Status

| Capability | State |
| --- | --- |
| Unified policy contract (lifecycle / action / status) | ✅ wired; two-process Dora smoke passed |
| Full RoboDojo episode | ✅ 700 steps completed, result JSON + video written (eval5) |
| ReKep solvers (reference subgoal/path) | ✅ run for real; warmup failures tolerated, solve failures degrade to hold-pose |
| Action spaces | ✅ `ee` (evaluator solves IK) and `joint` |
| Real X5 IK (solver feasibility) | ✅ loads `X5A.urdf`, IK error ~0 mm |
| Visual perception (depth + DINOv2) | ⚠️ code ready; Isaac 5.1 depth annotator crashes CUDA on this host, disabled by default |
| VLM constraint planning | ⚠️ code ready (`REKEP_PLAN_SOURCE=vlm`), needs keys; template is default |
| Template plan modes | ✅ regions present → 3-stage grab/place; no region (e.g. `general_pickup`) → 2-stage grab-and-lift-10cm, target chosen from instruction words |
| Action safety guards | ✅ finite/quaternion-normalized/clipped poses; obstacle radius and floor clearance on path waypoints |

## Layout

```text
src/rekep_policy/
├── core.py                # ReKepPolicy: perceive+plan once per episode, step execution per frame
├── profile.py             # action-name mapping; joint/ee action spaces
├── ik_adapters.py         # StubIK / X5IK (MuJoCo numeric IK)
├── perception_adapters.py # SceneObjectPerception (ground truth) / DepthDinoPerception (vision)
├── assembly.py            # real assembly (rekep_alg + IK + planner)
└── main.py                # node entrypoint driven by PolicyRuntime
rekep_alg/                 # algorithm core (perception / planner+sandbox / executor state machine)
skill/                     # PAOS manifest v2 / SKILL.md / profile dataflow
build_node.sh              # PyInstaller single-file node build
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `REKEP_PERCEPTION` | `scene_objects` (default, stable) or `dino` (needs depth) |
| `REKEP_IK` | `x5` (default) or `stub` |
| `REKEP_PLAN_SOURCE` | `template` (default) or `vlm` |
| `REKEP_X5_URDF` / `REKEP_X5_BASE_POS` / `REKEP_X5_BASE_QUAT_WXYZ` | X5 kinematics |
| `REKEP_REF_RUNTIME` | reference runtime dir (required for the packaged binary) |
| `REKEP_SUBGOAL_MAXFUN` / `REKEP_PATH_MAXFUN` / `REKEP_MINIMIZER_MAXITER` | solver budget |

## Run

From source: `dora run dataflow.rekep.yaml` (see `skill/profiles/robodojo-rekep`).
Single-file artifact: `bash build_node.sh` produces
`dist/rekep_policy-<ver>-linux-x86_64.tar.gz` and `skill/skill.yaml`.

## Known boundaries

- RoboDojo `arx_x5` is a dual-arm, four-bottle handover task; this policy drives the
  left arm only and holds the right arm, so current scores are not a ReKep capability
  measurement.
- The `distance_to_image_plane` annotator is unstable on this host (CUDA illegal
  address / device lost), so `dino` perception is not enabled by default.
- Grasp success is not observable in the evaluator observations; the policy advances
  assuming the grasp succeeded (recorded in the plan phase).

## Test log (2026-10-29, zyserver)

- **Stable path**: `put_bottles_into_dustbin` + `scene_objects` perception + real X5 IK
  completed 700 steps twice (`valid_episodes=1`, ~15 ms mean policy latency, result and
  video written).
- **Control experiment**: in `general_pickup`, a hold-only policy ran the full 200 steps;
  as soon as the arm moves, `CUDA_ERROR_ILLEGAL_ADDRESS` appears within 2–17 steps
  (reproduced with a single object, so clutter count is not the trigger). Action guards
  were added; retest pending a stable environment/driver.
- **Concurrent work**: a PAOS `bench_tool` refactor is in progress in the same repo and
  host, so the eval entrypoint may change. Once stable, run through the Skill flow in
  `NEW_POLICY_GUIDE.md` (`paos skill start robodojo-bench --profile rekep`) instead of a
  bare `dora run`.
