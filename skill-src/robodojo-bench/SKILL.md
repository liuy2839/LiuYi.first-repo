---
name: robodojo-bench
description: Run RoboDojo manipulation-policy benchmarks (Isaac Sim, ARX X5) through the Forge Tool API and report authoritative success rates.
metadata: {"PhyAgentOS":{"requires":{"runtime":["robodojo-bench"]}}}
---

# RoboDojo benchmark runner

Use this Skill only with the active profile (`g05`: G05 policy on
`put_bottles_into_dustbin`, ARX X5, Isaac Sim). It drives batch evaluation of a
manipulation policy and returns authoritative, file-backed results.

## Required order

1. Call `benchmark.describe` with `{}`. It returns the capability snapshot:
   `suite`, `tasks` (each with `task_id`, `num_init_states`, `name`,
   `language`), `joint_names`, `camera_ids`, `defaults` (`num_runs` / `seed` /
   `max_steps`) and `limits` (`max_total_episodes` / `max_max_steps`).
   Never invent task IDs or episode counts; derive them from this snapshot.
2. Call `benchmark.run` with the batch you want:
   `task_ids` (array or `null` = all), `init_state_ids` (array or `null` =
   min across selected tasks), `num_runs`, `max_steps`, `seed`. Omitted
   fields fall back to `defaults`. The call returns 202 with an
   `invocation_id` — admission is not completion.
3. Poll the invocation `status`/`result` until terminal. Progress events fire
   on episode boundaries (`episode_index`, `successes`). Cold start can take
   minutes (Isaac Sim loads the scene on the first run of a task).
4. Read the terminal result. The authoritative fields are `run_id`, `status`
   (`succeeded` / `failed` / `partial` / `stopped`), `completed_episodes`,
   `total_episodes`, `success_rate`, `result_file` (a
   `benchmark_execution_result_v1` JSON with per-episode records and videos),
   `failure_summary`. A completed invocation is not success unless its output
   says so; `partial` means a stop landed with episodes already done.
5. Stop cleanly with the session `stop` operation if the user cancels; it
   lands at the next episode boundary and yields a `partial` result with the
   episodes completed so far.

## Constraints

- One benchmark runs at a time (`max_concurrency=1`); a second concurrent
  `run` is rejected with `FORGE_ENDPOINT_BUSY` — retry later, do not queue.
- Episode counts above `limits.max_total_episodes` are rejected synchronously
  with `BENCH_INVALID_ARGUMENT`; narrow `task_ids` / `init_state_ids` /
  `num_runs`.
- `benchmark.run` arguments with a given `invocation_id` must be identical on
  replay; changing arguments under the same key is a conflict.
- Report numbers exactly as returned (`success_rate`, per-episode
  `termination`); do not recompute or round them.
- The result file is on the server filesystem (`result_file` path); quote it
  verbatim, never guess local paths.

## AgentTask verification contract

```json
{"mode": "audit", "goal": "run the requested RoboDojo benchmark batch and report the authoritative result", "success_criteria": ["terminal ToolResult for benchmark.run received", "success_rate and completed/total episodes reported from the ToolResult outputs", "result_file path reported"], "evidence_policy": {"minimum_association": "best_effort"}}
```
