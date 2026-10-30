"""服务模式：一次 benchmark.controller/run session 的封装（数据面进程内）。

本模块**只描述一次 session 是什么**，不含事件循环——循环在各 benchmark 的
serve 入口里。与官方 benchmark-libero 的同名模块同构：capabilities 供
describe 直读，BenchSession 把一次 run 的参数落成一次性的 BenchmarkRunner。

⚠️ **不 import forge_tool。** Forge 协议整层（descriptor/endpoint/租约/事件循环）
在工具面（bench_tool 包，py3.12 进程）；_core 保持零 forge_tool 依赖，
autorun 的独立跑法不为此拖进任何 tool 运行时。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner


def bench_limits(cfg: BaseBenchmarkConfig) -> dict[str, int]:
    """serve 模式的批次护栏，0 = 不限。endpoint 的 plan 归一化按这个拒超预算计划。"""
    return {
        "max_total_episodes": int(getattr(cfg, "max_total_episodes", 0) or 0),
        "max_max_steps": int(getattr(cfg, "max_max_steps", 0) or 0),
    }


# ---------------------------------------------------------------- capabilities


def _camera_ids(cfg: BaseBenchmarkConfig, adapter: Any) -> list[str]:
    """观测图像的 dora 通道名。

    不同 benchmark 的来源不一样：有的在 config 里配 `camera_map`（通道→相机名），
    有的 adapter 暴露 `camera_ids()` 钩子。两条都试，都没有就给空表——
    **不要为此改 adapter 协议**，describe 少一个字段不影响调用方构造 run。
    """
    mapping = getattr(cfg, "camera_map", None)
    if isinstance(mapping, dict) and mapping:
        return sorted(mapping)
    hook = getattr(adapter, "camera_ids", None)
    if callable(hook):
        try:
            return sorted(str(x) for x in hook())
        except Exception:  # noqa: BLE001 拿不到相机名不该挡住 describe
            return []
    return []


def capabilities_payload(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    *,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """benchmark.controller/describe 的数据源，进程内直读 —— 没有缓存/心跳/过期
    问题，一次 describe 就是一次真实快照。

    benchmark 特有的口径（RoboDojo 的 env_cfg_type/action_type 之类）一律走
    `adapter.reproducibility()` 原样透传，_core 不认识它们。
    """
    names = getattr(adapter, "tasks", None)
    tasks: list[dict[str, Any]] = []
    for task_id in range(adapter.num_tasks):
        entry: dict[str, Any] = {
            "task_id": task_id,
            "num_init_states": int(adapter.num_init_states(task_id)),
        }
        if names is not None and task_id < len(names):
            entry["name"] = str(names[task_id])
        try:
            entry["language"] = str(adapter.task_language(task_id))
        except Exception:  # noqa: BLE001 拿不到指令不该挡住 describe
            entry["language"] = ""
        tasks.append(entry)

    repro = getattr(adapter, "reproducibility", None)
    try:
        reproducibility = dict(repro()) if callable(repro) else {}
    except Exception:  # noqa: BLE001
        reproducibility = {}

    return {
        "benchmark": cfg.benchmark,
        "suite": cfg.suite,
        "policy_id": cfg.policy_id,
        "num_tasks": int(adapter.num_tasks),
        "tasks": tasks,
        "joint_names": list(adapter.state_joint_names()),
        "camera_ids": _camera_ids(cfg, adapter),
        "defaults": {
            "num_runs": cfg.num_runs,
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
        },
        "limits": dict(limits or {}),
        "reproducibility": reproducibility,
    }


# ------------------------------------------------------------------- 一次执行


class BenchSession:
    """一次 benchmark.controller/run session（数据面进程内）。

    `BenchmarkRunner` 是**一次性**的——`__init__` 就把 `cfg.episode_plan()` 绑死，
    `finished` 是不可逆的单向开关。所以每次 session 新建一个，cfg 用
    `replace()` 克隆，绝不改模板 cfg（改了会污染下一次）。
    """

    def __init__(
        self,
        node: Any,
        cfg_template: BaseBenchmarkConfig,
        adapter: Any,
        plan: dict[str, Any],
        invocation_id: str,
        log=print,
    ) -> None:
        self.invocation_id = invocation_id
        self._log = log
        self.cfg = replace(
            cfg_template,
            task_ids=[int(t) for t in plan["task_ids"]],
            init_state_ids=[int(i) for i in plan["init_state_ids"]],
            num_runs=int(plan["num_runs"]),
            seed=int(plan["seed"]),
            max_steps=int(plan["max_steps"]),
        )
        # 权威兜底校验。endpoint 侧拿 capabilities 也校验了一遍，但那份可能
        # 与 adapter 的真实清单不一致；这里抛出来会被上层转成同步拒绝，
        # 而不是"跑起来之后失败"。
        resolve_and_validate_plan(self.cfg, adapter)

        # **adapter 会回读 cfg，必须让它和 runner 看的是同一个对象。**
        # RoboDojoAdapter 在构造时存了 `self.cfg`，并在 reset_episode /
        # _ensure_env 里回读（env_cfg_type / device_id / init_states_per_task）。
        # 不指过去的话，run 传的参数到不了 adapter —— 分数照出、不报任何错。
        # 判据是「有没有 cfg 这个属性」，不是「cfg 是不是 None」：后者会让 cfg
        # 初始为 None 的 adapter 静默拿不到回写通道。
        if hasattr(adapter, "cfg"):
            adapter.cfg = self.cfg

        self.runner = BenchmarkRunner(
            node,
            self.cfg,
            adapter,
            log=log,
            # 文件名的时间戳只到秒，同一秒内两次 session 会写到同一路径
            result_tag=invocation_id,
            # 工具面进程的生命周期长于单次 session，关了 env 下一次 run
            # 就要重载整个场景；释放交给进程退出时统一做。
            close_adapter_on_finish=False,
        )
        self.started_at = time.time()
        self._last_progress = -1

    # ---- 生命周期 ----

    def start(self) -> None:
        self.runner.policy_command("start")
        self.runner.advance()

    def on_action(self, command: Any) -> None:
        self.runner.on_action(command)

    def request_stop(self, reason: str | None = None) -> None:
        """请求在 episode 边界停。消费点在 runner.start_next_episode：
        当前集跑完后不再开下一集，批次以 stopped 收尾（已有完成集作 partial）。"""
        self.runner.request_stop(reason)

    def fail(self, code: str, message: str) -> None:
        self.runner.fail_current(code, message)

    @property
    def finished(self) -> bool:
        return self.runner.finished

    @property
    def result_path(self) -> str | None:
        return self.runner.result_path

    # ---- 回执 ----

    def progress_report(self) -> dict[str, Any] | None:
        """episode 边界才发一条，不要每步发。没跨界就返回 None。"""
        index = self.runner.plan_index
        if index == self._last_progress:
            return None
        self._last_progress = index
        return {
            "invocation_id": self.invocation_id,
            "episode_index": index,
            "total": len(self.runner.plan),
            "successes": self.runner.result.successes,
        }


__all__ = ["BenchSession", "bench_limits", "capabilities_payload"]
