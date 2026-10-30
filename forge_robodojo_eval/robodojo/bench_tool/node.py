"""工具面（py3.12）的 dora 事件循环：forge_tool 协议 ↔ bench_command 桥。

职责（与官方 benchmark-libero 的 bench_tool/node.py 对应，但数据面在另一进程）：
- ToolEndpointHandler + DoraToolEndpointBinding：tool_in → 分发 → tool_out；
- EndpointLease：空闲时靠 node.next(timeout=) 驱动续租；
- run 登记拾取：dispatch 之外把 {run, plan} 经 bench_command 发给数据面；
- 数据面回执：benchmark_status（结构化 started/rejected/busy + 文本进度）与
  benchmark_result（终态 JSON）→ InvocationTable + 进度/终态 ToolEvent；
- 数据面 capabilities：等 BENCH_CAPABILITIES_CHANNEL 上的第一条 JSON，缓存。

关键纪律：endpoint 方法在 dispatch 内同步跑到底，只登记不执行；任何 dora
发送（bench_command、tool_out）都在 dispatch 之外。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable

from forge_msgs import ToolMessage
from forge_tool import ToolEndpointHandler, ToolEnvelope, ToolEvent
from forge_tool.dora import (
    DoraToolEndpointBinding,
    tool_envelope_to_message,
    tool_message_to_envelope,
)

from .descriptor import BENCH_DESCRIPTOR, OP_DESCRIBE, OP_RUN
from .endpoints import BenchDescribeEndpoint, BenchSessionEndpoint, SessionHandle
from .invocations import InvocationTable

TOOL_INPUT_ID = "tool_in"
TOOL_OUTPUT_ID = "tool_out"
# 数据面 ↔ 工具面 的桥接通道
BENCH_COMMAND_OUTPUT_ID = "bench_command"
BENCH_STATUS_INPUT_ID = "benchmark_status"
BENCH_RESULT_INPUT_ID = "benchmark_result"
BENCH_CAPABILITIES_INPUT_ID = "benchmark_capabilities"

# node.next(timeout=) 的周期。空闲时 tool_in 低频，续租与 pending 拾取全靠它。
# lease_ttl_ms 由 gateway 配置（本 skill 的 profile 调到 120000：episode reset
# 会阻塞数据面事件循环数十秒——工具面不受影响，但保守起见同值）；EndpointLease
# 在 ttl/3 处重注册，所以 tick 周期必须 ≤ ttl/3，取 1s。
TICK_PERIOD_S = 1.0


def _is_timeout(event: Any) -> bool:
    """dora 把 `next(timeout=)` 的超时表达成 ERROR 事件，不是返回 None。"""
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


def _text_from_value(value: Any) -> str | None:
    """dora 文本通道的事件 value → str。容忍 string array / scalar / bytes。"""
    if value is None:
        return None
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        value = as_py()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    # pa.array(["..."]) 的 value 可能是整条 array
    try:
        import pyarrow as pa

        if isinstance(value, pa.Array) and len(value) > 0:
            return str(value[0].as_py())
    except Exception:  # noqa: BLE001
        pass
    return None


def _json_from_value(value: Any) -> dict[str, Any] | None:
    text = _text_from_value(value)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class BenchServeLoop:
    """工具面状态机。**不碰 dora 发送**（发送集中在本类之外由循环做），可单测。"""

    def __init__(
        self,
        node: Any,
        log: Callable[[str], None] = print,
        *,
        endpoint_instance_id: str | None = None,
    ) -> None:
        self.node = node
        self._log = log
        self.endpoint_instance_id = endpoint_instance_id or str(uuid.uuid4())
        self.caps: dict[str, Any] | None = None
        self.table = InvocationTable()
        self._async = asyncio.Runner()
        self.endpoint = BenchSessionEndpoint(
            self.table,
            caps_provider=lambda: self.caps,
            send_bench_command=self._queue_bench_command,
            log=log,
        )
        self.handler = ToolEndpointHandler(
            BENCH_DESCRIPTOR,
            endpoint_instance_id=self.endpoint_instance_id,
            operations={
                OP_DESCRIBE: BenchDescribeEndpoint(caps_provider=lambda: self.caps),
                OP_RUN: self.endpoint,
            },
        )
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        from .lease import EndpointLease

        self.lease = EndpointLease(BENCH_DESCRIPTOR, self.endpoint_instance_id)
        self._outbound: list[tuple[str, Any]] = []
        self._closed = False

    # ---- 出站（只在事件循环轮次之间真正发送）----

    async def _send_batch(self, batch: Any) -> None:
        self.node.send_output(TOOL_OUTPUT_ID, batch)

    def _queue_bench_command(self, payload: dict[str, Any]) -> None:
        """endpoints 侧调用的"发送"其实只是入队；真正的 dora 发送在循环轮次里。"""
        import pyarrow as pa

        self._outbound.append(
            (BENCH_COMMAND_OUTPUT_ID, pa.array([json.dumps(payload, ensure_ascii=False)]))
        )

    def flush_outbound(self) -> None:
        for channel, value in self._outbound:
            self.node.send_output(channel, value)
        self._outbound.clear()

    def _send_envelope(self, envelope: ToolEnvelope) -> None:
        self.node.send_output(TOOL_OUTPUT_ID, tool_envelope_to_message(envelope).to_arrow())

    def _drive_lease(self) -> None:
        request = self.lease.poll(time.monotonic())
        if request is not None:
            self._send_envelope(request)

    def _emit_event(self, handle: SessionHandle, event: ToolEvent) -> None:
        emitter = getattr(handle, "emitter", None)
        if emitter is None:
            return
        try:
            self._async.run(emitter.emit(event))
        except Exception:  # noqa: BLE001 事件发不出去不该打死批次
            self._log(f"tool 事件发送失败（{event.type}），忽略")

    # ---- 入站 ----

    def _handle_tool(self, value: Any) -> None:
        message = ToolMessage.from_arrow(value)
        envelope = tool_message_to_envelope(message)
        if envelope.message_type == "endpoint.registry.response":
            # 漏了这一步的症状：gateway 起得来、GET /tools 也有，但 ready 永远
            # false，而且不报任何错。
            self.lease.acknowledge(envelope, time.monotonic())
            return
        for batch in self._async.run(self.binding.dispatch_input(value)):
            self.node.send_output(TOOL_OUTPUT_ID, batch)

    # ---- 数据面回执 ----

    def _on_capabilities(self, payload: dict[str, Any]) -> None:
        if self.caps is None:
            self._log(
                f"数据面 capabilities 已就绪：benchmark={payload.get('benchmark')} "
                f"suite={payload.get('suite')} num_tasks={payload.get('num_tasks')}"
            )
        self.caps = payload

    def _on_status(self, text: str) -> None:
        payload = _json_from_value(text)
        if payload is not None and "command" in payload:
            command = payload.get("command")
            invocation_id = str(payload.get("invocation_id") or "")
            if command == "progress" and invocation_id:
                handle = self.endpoint.handle_for(invocation_id)
                if handle is not None:
                    self.table.update_details(invocation_id, dict(payload))
                    self._emit_event(
                        handle,
                        ToolEvent(type="progress", data=dict(payload)),
                    )
                return
            self.endpoint.on_status_event(payload)
            return
        # 纯文本进度（数据面的 episode 开始/结束日志）
        self._log(f"[数据面] {text}")

    def _on_result(self, payload: dict[str, Any]) -> None:
        invocation_id = str(payload.get("invocation_id") or "")
        handle = self.endpoint.handle_for(invocation_id)
        # 先落表，再发终态事件（顺序是 forge_tool handler 的硬约束）
        self.endpoint.on_result_payload(payload)
        if handle is not None and not getattr(handle, "_terminal_event_sent", False):
            item = self.table.by_id(invocation_id)
            if item is not None and item.terminal:
                handle._terminal_event_sent = True
                event_type = {
                    "completed": "executor_completed",
                    "failed": "executor_failed",
                    "stopped": "stopped",
                }.get(item.phase, "executor_failed")
                self._emit_event(
                    handle,
                    ToolEvent(
                        type=event_type,  # type: ignore[arg-type]
                        data={
                            "invocation_id": invocation_id,
                            "status": item.result.status if item.result else "unknown",
                            "result_file": payload.get("result_path"),
                        },
                    ),
                )
                self._log(f"session {invocation_id} 终态 {item.phase}，回到 idle")

    # ---- stop 的 bench_command 补发 ----

    def _dispatch_stops(self) -> None:
        for invocation_id, item in self.table._items.items():
            if item.stop_requested and not getattr(item, "_stop_sent", False):
                handle = self.endpoint.handle_for(invocation_id)
                reason = handle.stop_reason if handle is not None else None
                self.endpoint.dispatch_stop(invocation_id, reason)
                item._stop_sent = True  # type: ignore[attr-defined]

    # ---- 事件循环 ----

    def process_event(self, event: Any) -> int | None:
        """处理一个 dora 事件。返回非 None 表示外层循环该退出，值就是退出码。"""
        if event is None or _is_timeout(event):
            self._drive_lease()
            self.endpoint.dispatch_pending()
            self._dispatch_stops()
            self.flush_outbound()
            return None

        etype = event.get("type") if isinstance(event, dict) else None
        self._drive_lease()
        if etype == "STOP":
            unregister = self.lease.stop()
            if unregister is not None:
                self._send_envelope(unregister)
            self._async.close()
            self._closed = True
            return 0
        if etype == "ERROR":
            self._log(f"dora error: {event.get('error')}")
            self._async.close()
            self._closed = True
            return 1
        if etype != "INPUT":
            return None

        self.endpoint.dispatch_pending()
        self._dispatch_stops()
        eid = event.get("id")
        try:
            if eid == TOOL_INPUT_ID:
                self._handle_tool(event.get("value"))
            elif eid == BENCH_CAPABILITIES_INPUT_ID:
                payload = _json_from_value(event.get("value"))
                if payload is not None:
                    self._on_capabilities(payload)
            elif eid == BENCH_STATUS_INPUT_ID:
                text = _text_from_value(event.get("value"))
                if text:
                    self._on_status(text)
            elif eid == BENCH_RESULT_INPUT_ID:
                payload = _json_from_value(event.get("value"))
                if payload is not None:
                    self._on_result(payload)
            # 其它输入只当心跳
        except (TypeError, ValueError) as exc:
            self._log(f"忽略非法输入：{exc}")
        finally:
            self.flush_outbound()
        return None

    def shutdown(self) -> None:
        """进程退出前调一次（幂等）。"""
        if not self._closed:
            self._async.close()
            self._closed = True


def run_bench_tool(log: Callable[[str], None] = print, *, node: Any = None) -> int:
    """工具面入口。独立 dora 节点进程（py3.12 venv）。"""
    from dora import Node

    dora_node = node if node is not None else Node()
    loop = BenchServeLoop(dora_node, log=log)
    log(
        f"工具面就绪：endpoint {BENCH_DESCRIPTOR.endpoint_id} 等待 tool_in；"
        f"等数据面 capabilities（{BENCH_CAPABILITIES_INPUT_ID}）"
    )
    rc = 0
    try:
        while True:
            # **必须带 timeout**：tool_in 是低频输入，空闲时若无限阻塞，
            # 续租与 pending 拾取都发不出去。
            event = dora_node.next(timeout=TICK_PERIOD_S)
            outcome = loop.process_event(event)
            if outcome is not None:
                rc = outcome
                break
    finally:
        loop.shutdown()
    return rc


__all__ = [
    "BENCH_CAPABILITIES_INPUT_ID",
    "BENCH_COMMAND_OUTPUT_ID",
    "BENCH_RESULT_INPUT_ID",
    "BENCH_STATUS_INPUT_ID",
    "BenchServeLoop",
    "TICK_PERIOD_S",
    "TOOL_INPUT_ID",
    "TOOL_OUTPUT_ID",
    "run_bench_tool",
]
