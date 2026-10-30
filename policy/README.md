# forge-policy

[English](README.md) | [简体中文](README_zh.md)

Unified policy-node wire contract and Policy Runtime SDK. Any policy implementing
the `Policy` interface plugs into the Forge evaluation dataflow (current target:
RoboDojo benchmark) without touching the evaluator.

## Node / Endpoint / Tool API

| Item | Current |
| --- | --- |
| Production Nodes | 0 (SDK library; policy Nodes are composed by consumers + a main script) |
| Tool support | Data-plane only (proprio_state / image / depth / seg / policy_command → action / policy_status) |
| Endpoints | 0 |
| Operations | 0 |
| Direct HTTP Tool API | 0 (carried by dataflow messages) |
| Gateway Tools | 0 |

## Contract summary

- Inputs: `proprio_state` (JointState), `image/<cam>` (rgb8), `depth/<cam>`
  (32FC1 meters, optional), `seg/<cam>` (16UC1, optional), `policy_command`
  (queue_size ≥ 32), `tick` (heartbeat only).
- Outputs: `action` (JointCommand whose name/position match the evaluator's
  expectation verbatim), `policy_status` (JSON state machine).
- Lifecycle: `start` → `set_instruction` → `reset_scene` (per episode) →
  `stop/cancel`; observation-driven inference (at most one action per complete
  observation frame, matching the evaluator's 1:1 lockstep).
- Policies implement `forge_policy.base.Policy`:
  `reset_episode/set_instruction/step/shutdown`.

## Development & tests

```bash
cd packages/policy
uv venv && uv pip install --python .venv <local forge-msgs checkout> numpy pyarrow pytest
.venv/bin/python -m pytest tests -q
```

On machines without GitLab credentials, use the local
`forge-forge-msgs-v1.0.1/packages/msgs` path source (same field layout as the
rev pinned in the root uv.lock); deployments can switch back to the git source.
