# RoboDojo Benchmark Node

This package connects the RoboDojo evaluation environment to Forge through a
two-node Dora dataflow. The benchmark node owns Isaac Sim and the policy node
loads an XPolicyLab policy, exchanging typed observations and named joint
actions. Keep the two nodes in their documented Python environments: Isaac Sim
and XPolicyLab pin incompatible `websockets` versions.

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

On a shared machine, give Isaac Lab a writable temporary directory through
`TMPDIR` when launching Dora. Check any upstream RoboDojo robot/CuRobo asset
paths against the local checkout before running; those files are not part of
this package. Set `PYTHONPATH` for the benchmark process if RoboDojo's
`client_server` package is not otherwise importable. If using
`CUDA_VISIBLE_DEVICES`, set `device_id` to the index within the visible set
and verify that Isaac Sim's Vulkan device selection agrees with the intended
GPU. These are deployment settings, not portable defaults.

After filling private copies of both example YAML files, run from this
directory with a writable temp directory:

```bash
TMPDIR=/path/to/writable/tmp dora run dataflow.yaml
```

The example uses `put_bottles_into_dustbin`, `arx_x5`, joint actions, and two
layouts as a small smoke run. Increase `init_states_per_task` only when a full
benchmark is explicitly required. Result and video paths are resolved relative
to the node directory and are ignored by git.

## Validation record

On 2026-10-24, the deployed copy completed a two-layout
`put_bottles_into_dustbin` smoke run with G05: `status=succeeded`,
`completed_episodes=2`, `valid_episodes=2`, and `successes=2`. This establishes
that the simulator, policy, Dora messages, and result writer worked together
for those two layouts. It is not a full benchmark or a reliable estimate of
the policy's overall success rate.

After that run, the policy node was changed to exit on `stop` and `cancel`.
Its message handling passed local tests, and a lightweight two-process Dora
probe delivered `stop` and exited automatically. A full simulator run has not
yet revalidated automatic shutdown because a later attempt hit an unrelated
Vulkan out-of-device-memory error during startup.

Do not place raw episode videos, traces, or benchmark result directories in a
commit. Keep only a small summary if an external audit requires one.
