"""benchmark.controller 的两个 operation 实现：describe(Query)与 run(Session)。

双进程桥接版（与官方 benchmark-libero 的同文件不同）：工具面（本进程，
py3.12）不持有仿真器；run 的真实执行发生在数据面进程（py3.10，Isaac Sim）。
本模块职责：

- describe：返回启动时数据面上报的 capabilities（工具面缓存，进程内直读）；
- run：**登记 + 转发**。校验参数 → 登记进 InvocationTable → 经 bench_command
  通道把 {run, plan} 发给数据面。终态不在这里产生——node.py 收到数据面的
  benchmark_result 回执后调 on_finished 落终态；
- stop：经 bench_command 转发 {stop}，数据面在 episode 边界落停。

**start() 只登记不等待**：任何 dora 发送都在事件循环的 dispatch 之外做
（node.py 把 pending 拾取放在每轮 dispatch 前），这是从官方双节点架构继承
的死锁纪律。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from forge_tool import (
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolEventEmitter,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)

from .descriptor import (
    ERR_BUSY,
    ERR_CONFLICT,
    ERR_INVALID_ARGUMENT,
    ERR_NOT_READY,
    ERR_UNKNOWN_INVOCATION,
    endpoint_error,
    tool_error,
)
from .invocations import InvocationTable
from .result_shape import outputs_from_payload, stopped_summary

# describe 的输出白名单。capabilities 里还有 benchmark 内部的其它字段
# （reproducibility 之类），直接透传等于把内部细节漏给 Agent，而且 schema 会对不上。
_DESCRIBE_FIELDS = (
    "benchmark",
    "suite",
    "policy_id",
    "num_tasks",
    "tasks",
    "joint_names",
    "camera_ids",
    "defaults",
    "limits",
    "reproducibility",
)

# run 参数 → data 面 plan 的字段映射（与 _core/config 的 episode_plan 对齐）
_PLAN_KEYS = ("task_ids", "init_state_ids", "num_runs", "max_steps", "seed")


@dataclass
class SessionHandle:
    """一次 run session 在端点侧与循环侧之间传递的唯一把手。"""

    key: ToolExecutionKey
    plan: dict[str, Any]
    arguments_json: str
    emitter: Any = None  # forge_tool 的 ToolEventEmitter；start() 时由循环侧存入
    stop_reason: str | None = None
    reported: bool = False
    # 发给数据面的 bench_command 是否已投递（node.py 拾取 pending 时置 True）
    dispatched: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class BenchDescribeEndpoint:
    """QueryToolEndpoint：进程内直读缓存的 capabilities，永不等待。"""

    def __init__(self, caps_provider: Callable[[], dict[str, Any] | None]) -> None:
        self._caps_provider = caps_provider

    async def query(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        caps = self._caps_provider() or {}
        if not caps:
            return ToolResult(
                status="failed",
                error=tool_error(
                    ERR_NOT_READY,
                    "数据面尚未上报 capabilities（Isaac Sim 可能还在启动）",
                ),
            )
        outputs = {name: caps.get(name) for name in _DESCRIBE_FIELDS if name in caps}
        return ToolResult(status="succeeded", outputs=outputs)


class BenchSessionEndpoint:
    """SessionToolEndpoint：start 登记+转发，stop 转发，终态由回执驱动。"""

    def __init__(
        self,
        table: InvocationTable,
        caps_provider: Callable[[], dict[str, Any] | None],
        *,
        send_bench_command: Callable[[dict[str, Any]], None],
        log: Callable[[str], None] = print,
    ) -> None:
        self.table = table
        self._caps_provider = caps_provider
        self._send = send_bench_command
        self._log = log
        self._handles: dict[str, SessionHandle] = {}
        self._pending: list[SessionHandle] = []

    # ---- 循环侧接口（同步，不进 tool 分发）----

    def take_pending(self) -> list[SessionHandle]:
        items, self._pending = self._pending, []
        return items

    def handle_for(self, invocation_id: str) -> SessionHandle | None:
        return self._handles.get(invocation_id)

    def active_handle(self) -> SessionHandle | None:
        for handle in self._handles.values():
            if not handle.reported:
                return handle
        return None

    def on_status_event(self, payload: dict[str, Any]) -> None:
        """数据面 benchmark_status 的结构化消息（started/busy/rejected）。"""
        command = payload.get("command")
        invocation_id = str(payload.get("invocation_id") or "")
        if command == "started" and invocation_id:
            self.table.mark_running(invocation_id, {"phase_detail": "running"})
        elif command == "rejected" and invocation_id:
            # 数据面权威兜底校验拒绝 → 终态 failed
            self.table.finish(
                invocation_id,
                "failed",
                outputs={"run_id": invocation_id},
                error=tool_error(
                    ERR_INVALID_ARGUMENT,
                    str(payload.get("error") or "数据面拒绝该计划"),
                ),
            )
            handle = self._handles.get(invocation_id)
            if handle is not None:
                handle.reported = True
        elif command == "busy":
            self._log(f"数据面忙：{payload}")

    def on_result_payload(self, payload: dict[str, Any]) -> None:
        """数据面 benchmark_result 回执 → 落终态。**先落表，后发终态事件**由调用方做。"""
        invocation_id = str(payload.get("invocation_id") or "")
        if not invocation_id:
            self._log(f"benchmark_result 缺 invocation_id，丢弃：{str(payload)[:200]}")
            return
        handle = self._handles.get(invocation_id)
        if handle is None:
            self._log(f"benchmark_result 对应的 session 不在本地表里，丢弃：{invocation_id}")
            return
        if handle.reported:
            return
        handle.reported = True

        status = str(payload.get("status") or "failed")
        completed = int(payload.get("completed_episodes") or 0)
        total = int(payload.get("total_episodes") or 0)
        outputs = outputs_from_payload(run_id=invocation_id, payload=payload)

        phase = {"succeeded": "completed", "failed": "failed", "stopped": "stopped"}.get(
            status, "failed"
        )
        error = None
        failure_summary = outputs.get("failure_summary")
        if phase == "failed":
            failure_summary = failure_summary or "批次未正常结束"
            error = tool_error(
                "BENCH_RUN_FAILED",
                str(failure_summary),
                details={"status": status},
            )
        self.table.finish(invocation_id, phase, outputs=outputs, error=error)

    # ---- SessionToolEndpoint SPI ----

    async def start(
        self,
        request: ToolRequest,
        context: ToolContext,
        events: ToolEventEmitter,
    ) -> ToolAccepted:
        key = context.execution_key
        arguments_json = json.dumps(
            dict(request.arguments),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        existing = self._handles.get(key.invocation_id)
        if existing is not None:
            if existing.arguments_json != arguments_json:
                # handler 的重复 key 缓存承接的是同参数重放；异参重放必须在这里
                # 拒掉，绝不能把新参数塞进一个已经登记的 session。
                raise endpoint_error(
                    ERR_CONFLICT,
                    "execution key 已用于一次参数不同的 session；重复 key 必须带相同参数",
                )
            return ToolAccepted(
                details={"invocation_id": key.invocation_id, "idempotent": True}
            )

        # 先查忙，再校验参数：忙的时候连参数都不用看，而且 retryable 语义不同。
        active = self.table.active()
        if active is not None:
            raise endpoint_error(
                ERR_BUSY,
                f"已有 session {active.key.invocation_id} 在跑；一台机器只有一个仿真器",
                details={"active_invocation_id": active.key.invocation_id},
            )

        caps = self._caps_provider()
        if not caps:
            raise endpoint_error(
                ERR_NOT_READY,
                "数据面尚未上报 capabilities（Isaac Sim 可能还在启动），稍后再试",
            )

        plan = normalize_plan(dict(request.arguments), caps)

        # ⚠️ 这行以下就有副作用了（session 进队列、表已建），
        # 再往后不许抛 ToolEndpointError。
        self.table.create(key, plan)
        handle = SessionHandle(
            key=key, plan=plan, arguments_json=arguments_json, emitter=events
        )
        self._handles[key.invocation_id] = handle
        self._pending.append(handle)
        return ToolAccepted(
            details={
                "invocation_id": key.invocation_id,
                "total_episodes": plan["total_episodes"],
            }
        )

    async def stop(
        self,
        key: ToolExecutionKey,
        reason: str | None = None,
    ) -> ToolControlResponse:
        item = self.table.get(key)
        if item is None:
            return ToolControlResponse(
                command="stop",
                status="rejected",
                error=tool_error(
                    ERR_UNKNOWN_INVOCATION, f"没有 session {key.invocation_id!r} 的记录"
                ),
            )
        if item.terminal:
            # 已经终结：不是错误，只是没得可停。
            return ToolControlResponse(command="stop", status="terminal")
        if not item.stop_requested:
            item.stop_requested = True
            self.table.mark_stopping(key.invocation_id, {"phase_detail": "stopping"})
            handle = self._handles.get(key.invocation_id)
            if handle is not None:
                handle.stop_reason = reason
        # accepted 只表示"停止请求被收下"，不表示已经停了 —— 调用方必须继续看
        # status/result。这是 ToolControlResponse 文档里写死的语义。
        # 真正的 {stop} bench_command 由 node.py 在 dispatch 之外补发（那里才能安全发 dora）。
        return ToolControlResponse(command="stop", status="accepted", details={"reason": reason})

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        return self.table.status_for(key)

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        return self.table.result_for(key)

    # ---- 循环侧：dispatch 之外的动作 ----

    def dispatch_pending(self) -> None:
        """把新登记的 session 经 bench_command 发给数据面（事件循环里调）。"""
        for handle in self.take_pending():
            try:
                self._send(
                    {
                        "command": "run",
                        "invocation_id": handle.key.invocation_id,
                        "plan": handle.plan,
                    }
                )
                handle.dispatched = True
            except Exception as exc:  # noqa: BLE001 发不出去要把 session 收成终态
                self._log(f"bench_command 发送失败：{exc}")
                self.table.finish(
                    handle.key.invocation_id,
                    "failed",
                    outputs={"run_id": handle.key.invocation_id},
                    error=tool_error("BENCH_RUN_FAILED", f"bench_command 发送失败：{exc}"),
                )
                handle.reported = True

    def dispatch_stop(self, invocation_id: str, reason: str | None) -> None:
        """把 stop 请求经 bench_command 发给数据面（事件循环里调）。"""
        try:
            self._send(
                {"command": "stop", "invocation_id": invocation_id, "reason": reason}
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"stop bench_command 发送失败（忽略）：{exc}")


# ---- 计划归一化与校验 ----
#
# 这段逻辑和 _core/config.py:resolve_and_validate_plan 是重复的，**这是刻意的**：
# endpoint 拿到 capabilities 就能挡住绝大多数越界，把错误变成同步的
# BENCH_INVALID_ARGUMENT，而不是"接受了、跑起来了、然后失败"。数据面那份是
# 权威兜底（对 adapter 的真实清单），不能删。


def normalize_plan(arguments: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    tasks = capabilities.get("tasks") or []
    if not tasks:
        raise endpoint_error(ERR_INVALID_ARGUMENT, "capabilities 里没有任何 task")
    counts = {int(t["task_id"]): int(t["num_init_states"]) for t in tasks}

    task_ids = arguments.get("task_ids")
    if task_ids is None:
        task_ids = sorted(counts)
    task_ids = [int(t) for t in task_ids]
    if not task_ids:
        raise endpoint_error(ERR_INVALID_ARGUMENT, "task_ids 不能是空数组；要全集请传 null")
    bad = [t for t in task_ids if t not in counts]
    if bad:
        raise endpoint_error(
            ERR_INVALID_ARGUMENT,
            f"越界 task_ids: {bad};可用范围 0..{max(counts)}",
            details={"available_task_ids": sorted(counts)},
        )

    init_state_ids = arguments.get("init_state_ids")
    if init_state_ids is None:
        init_state_ids = list(range(min(counts[t] for t in task_ids)))
    init_state_ids = [int(i) for i in init_state_ids]
    if not init_state_ids:
        raise endpoint_error(ERR_INVALID_ARGUMENT, "init_state_ids 不能是空数组；要全集请传 null")
    for t in task_ids:
        # 逐 task 校验：不同 task 的 init state 数未必相同，拿 task_ids[0]
        # 代表全体是错的（_core 侧那份注释里踩过）。
        bad = [i for i in init_state_ids if not 0 <= i < counts[t]]
        if bad:
            raise endpoint_error(
                ERR_INVALID_ARGUMENT,
                f"task {t} 只有 {counts[t]} 个 init state，越界 init_state_ids: {bad}",
            )

    defaults = capabilities.get("defaults") or {}
    num_runs = int(arguments.get("num_runs", defaults.get("num_runs", 1)))
    max_steps = int(arguments.get("max_steps", defaults.get("max_steps", 300)))
    seed = int(arguments.get("seed", defaults.get("seed", 0)))
    if num_runs < 1:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"num_runs 必须 ≥1，收到 {num_runs}")
    if max_steps < 1:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"max_steps 必须 ≥1，收到 {max_steps}")

    limits = capabilities.get("limits") or {}
    total = len(task_ids) * len(init_state_ids) * num_runs
    cap_total = int(limits.get("max_total_episodes", 0) or 0)
    if cap_total and total > cap_total:
        # 没有这道闸，Agent 会一口气开满然后跑 8 小时，而且不报错。
        raise endpoint_error(
            ERR_INVALID_ARGUMENT,
            f"本次将跑 {total} 集，超过上限 {cap_total}；请缩小 task_ids/init_state_ids/num_runs",
            details={"requested_episodes": total, "max_total_episodes": cap_total},
        )
    cap_steps = int(limits.get("max_max_steps", 0) or 0)
    if cap_steps and max_steps > cap_steps:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"max_steps {max_steps} 超过上限 {cap_steps}")

    return {
        "task_ids": task_ids,
        "init_state_ids": init_state_ids,
        "num_runs": num_runs,
        "seed": seed,
        "max_steps": max_steps,
        "total_episodes": total,
    }


__all__ = [
    "BenchDescribeEndpoint",
    "BenchSessionEndpoint",
    "SessionHandle",
    "normalize_plan",
]
