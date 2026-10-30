# RoboDojo Benchmark Node

This package connects the RoboDojo evaluation environment to Forge through a
two-node Dora dataflow. The benchmark node owns Isaac Sim and the policy node
loads an XPolicyLab policy, exchanging typed observations and named joint
actions. Keep the two nodes in their documented Python environments: Isaac Sim
and XPolicyLab pin incompatible `websockets` versions.

## Two modes

- **autorun** (default, unchanged): `dora run dataflow.<policy>.yaml` runs the
  batch from `config.yaml` to completion. All existing deployment scripts
  keep working untouched.
- **serve** (`serve: true` in config): the node waits for `bench_command`
  inputs and runs one batch per command, reporting progress and terminal
  results over `benchmark_status` / `benchmark_result`. A separate
  `bench_tool` process (Python 3.12, `forge-tool` 1.x) exposes these as the
  Forge Tool API endpoint `benchmark.controller` (`describe` = Query,
  `run` = Session) so PhyAgentOS can drive evaluation through a gateway —
  see `bench_tool/` and `skill-src/robodojo-bench/` at the repository root.
  Adding a new policy = a new profile; see `NEW_POLICY_GUIDE.md`.

Both nodes use `adapters/messages.py` for the four Arrow message types in this
dataflow. It matches the `forge-msgs` 1.0.1 field schemas without requiring that
Python >=3.11 package in the G05 Python 3.10 environment. Other benchmark nodes
still use their installed `forge-msgs`. Run `tests/test_messages.py` in both
Python environments, then verify transport with a two-process Dora smoke run
before trusting scores; unit round trips alone do not test Dora delivery.
Install this benchmark node's dependencies from its own `pyproject.toml`, not
from the repository root workspace, which still contains unrelated private
Forge dependencies.

## Portable setup

The repository contains no checkpoint, simulator checkout, proxy setting, or
machine-specific output. Set these values in the deployment environment (or
replace the placeholders in a private copy of `dataflow.example.yaml`):

```text
ROBODOJO_ROOT       = /path/to/robodojo/repo
XPOLICYLAB_ROOT     = /path/to/robodojo/repo/XPolicyLab
G05_ROOT            = /path/to/G05                 # optional, if not vendored
G05_CKPT_PATH       = /path/to/OpenGalaxea/g05-robodojo
G05_PROCESSOR_PATH  = /path/to/qwen3_5_2b_base_processor
ROBODOJO_ENV        = /path/to/isaac-sim/environment
ROBODOJO_POLICY_ENV = /path/to/xpolicylab/environment
```

`G05_CKPT_PATH` may point to a released `.pt` file or to its extracted run
directory. `G05_PROCESSOR_PATH` is required for G05 so an upstream deploy file
cannot silently reintroduce a machine-specific processor path; it overrides any
processor path from that file. Select the Isaac device with `device_id` in the private
benchmark YAML. The policy process follows the normal CUDA visibility rules;
there is no hard-coded GPU number or proxy in this node.

Run from this directory after making a private config copy:

```bash
BENCHMARK_CONFIG=/path/to/config.yaml \
ROBODOJO_ROOT=/path/to/robodojo/repo \
dora run dataflow.example.yaml
```

The example uses `put_bottles_into_dustbin`, `arx_x5`, joint actions, and two
layouts as a small smoke run. Increase `init_states_per_task` only when a full
benchmark is explicitly required. Result and video paths are resolved relative
to the node directory and are ignored by git.

## Validation record

The formal Benchmark Node comparison used the public `OpenGalaxea/g05-robodojo`
checkpoint on `put_bottles_into_dustbin` with the same ordered layouts
`layout_id=0..49` on both sides. The public RoboDojo leaderboard reports 94%
success and 96.3% score for this checkpoint/task.

| runner | success | Wilson 95% CI | normalized mean score |
| --- | ---: | --- | ---: |
| Official | 46/50 (92%) | [0.8116, 0.9685] | 0.946 |
| Forge | 45/50 (90%) | [0.7864, 0.9565] | 0.934 |

Fisher exact two-sided `p=1.0`; the confidence intervals overlap. Under
Benchmark acceptance manual v3 criterion A this is a **PASS**.

The deterministic bonus probe matched state, metadata, instruction, and image
shape, but image SHA values differed across Isaac processes. Strict whole-record
JSONL equality is therefore **NOT PASS**. The intentional bad-action negative
control produced an immediate state mismatch and is **PASS**.

Do not place raw episode videos, traces, or benchmark result directories in a
commit. Keep only a small summary if an external audit requires one.
