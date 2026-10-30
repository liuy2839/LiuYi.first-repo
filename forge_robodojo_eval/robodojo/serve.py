"""RoboDojo 数据面的 serve 事件循环（py3.10 进程，与 Isaac Sim 同进程）。

与官方 benchmark-libero 的 bench_tool/node.py（工具面+数据面同进程）不同，
RoboDojo 的工具面跑在独立的 py3.12 进程（forge-tool 依赖 Python ≥3.12，而
Isaac Sim 钉死 3.10）。两进程经两条 dora 通道协作：

    工具面 ──bench_command──► 本模块（run/stop，JSON 文本通道）
    本模块 ──benchmark_status / benchmark_result──► 工具面（进度与终态）

数据面职责：等 bench_command；run 到来 → 建 BenchSession 开批；action 锁步
推进（与 autorun 完全相同的 1:1 协议）；stop 到来 → episode 边界落停；批次
结束 → 终态 JSON 经 benchmark_result 发回工具面，由它落成 ToolResult。

本文件**不 import forge_tool**——它是 py3.10 侧唯一要多认识的东西就是两条
文本通道，保持 Isaac Sim 环境零额外依赖。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from _core.session import BenchSession

# bench_command 的载荷类型
CMD_RUN = "run"
CMD_STOP = "stop"

# 等 action 的看门狗周期（秒）。与 autorun 的 _WATCHDOG_S 相同：明显大于
# 最慢策略的单步推理耗时（G05 首帧秒级），又不至于把真死锁挂太久。
_WATCHDOG_S = 20.0

# 工具面从 benchmark_result 收到终态后的宽限：数据面进程不退出（还要服务
# 下一次 run），但把 session 置空、回到等待 bench_command 的 idle 态。


def _is_timeout(event: Any) -> bool:
    """dora 把 `next(timeout=)` 的超时表达成 ERROR 事件，不是返回 None。"""
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


class ServeDataLoop:
    """数据面状态机。**不碰 tool 协议**，事件由外面喂，可以单测。

    状态只有一个：`self.session`。None = idle（等 bench_command），
    非 None = 正在跑一次 run session。
    """

    def __init__(
        self,
        node: Any,
        cfg: Any,
        adapter: Any,
        log: Callable[[str], None] = print,
    ) -> None:
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self._log = log
        self.session: BenchSession | None = None
        self._result_payload: dict[str, Any] | None = None

    # ---- 出站 ----

    def _emit(self, channel: str, payload: dict[str, Any] | str) -> None:
        import pyarrow as pa

        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.node.send_output(channel, pa.array([text]))

    def _emit_status(self, text: str) -> None:
        self._emit("benchmark_status", text)
        self._log(text)

    # ---- 入站：bench_command ----

    def on_bench_command(self, text: str) -> int | None:
        """处理一条 bench_command。返回非 None 表示进程该退出。"""
        try:
            cmd = json.loads(text)
        except (TypeError, ValueError) as exc:
            self._log(f"忽略非法 bench_command：{exc}")
            return None
        kind = cmd.get("command")
        if kind == CMD_RUN:
            return self._start_session(cmd)
        if kind == CMD_STOP:
            return self._request_stop(cmd.get("reason"))
        self._log(f"未知 bench_command：{kind!r}，忽略")
        return None

    def _start_session(self, cmd: dict[str, Any]) -> int | None:
        if self.session is not None and not self.session.finished:
            # 工具面侧 max_concurrency=1 已挡住并发；这里是数据面兜底。
            active = self.session.invocation_id
            self._log(f"run 拒绝：session {active} 还在跑（数据面兜底）")
            self._emit("benchmark_status", {
                "command": "busy",
                "active_invocation_id": active,
            })
            return None

        invocation_id = str(cmd["invocation_id"])
        plan = cmd.get("plan") or {}
        try:
            session = BenchSession(
                self.node, self.cfg, self.adapter, plan, invocation_id, log=self._log
            )
        except (ValueError, KeyError) as exc:
            # 权威兜底校验失败：同步拒绝，让工具面把 ToolResult 写成 failed。
            self._emit("benchmark_status", {
                "command": "rejected",
                "invocation_id": invocation_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return None

        self.session = session
        self._result_payload = None
        self._emit(
            "benchmark_status",
            {
                "command": "started",
                "invocation_id": invocation_id,
                "total_episodes": len(session.runner.plan),
            },
        )
        # episode 边界把结构化进度补发到 benchmark_status（工具面转 progress
        # ToolEvent 给 Agent）。包一层而不是改 runner：runner 是 _core 共用件。
        runner = session.runner
        _orig_emit_status = runner.emit_status

        def _emit_status_with_progress(text: str) -> None:
            _orig_emit_status(text)
            progress = session.progress_report()
            if progress is not None:
                self._emit("benchmark_status", {"command": "progress", **progress})

        runner.emit_status = _emit_status_with_progress
        session.start()
        if session.finished:
            self._on_finished()
        return None

    def _request_stop(self, reason: str | None) -> None:
        if self.session is None or self.session.finished:
            return
        self._emit_status(f"stop 请求已收下（{reason}），在 episode 边界落停")
        self.session.request_stop(reason)

    # ---- 数据面推进（与 autorun 同协议）----

    def on_action(self, value: Any) -> None:
        session = self.session
        if session is None or session.finished:
            return  # 上一批的迟到动作，丢掉
        try:
            session.on_action(value)
        except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
            session.fail("step_error", f"{type(exc).__name__}: {exc}")
        if session.finished:
            self._on_finished()

    def republish_if_stalled(self, timeout_s: float) -> None:
        session = self.session
        if session is None or session.finished:
            return
        session.runner.republish_if_stalled(timeout_s)

    # ---- 终态 ----

    def _on_finished(self) -> None:
        session = self.session
        if session is None:
            return
        runner = session.runner
        payload = runner.result.to_dict()
        payload["result_path"] = runner.result_path
        payload["invocation_id"] = session.invocation_id
        self._result_payload = payload
        self._emit("benchmark_result", payload)
        self._log(
            f"session {session.invocation_id} 终态 {payload['status']}，回到 idle"
        )
        self.session = None

    # ---- 事件循环 ----

    def process_event(self, event: Any) -> int | None:
        if event is None or _is_timeout(event):
            self.republish_if_stalled(_WATCHDOG_S)
            return None

        etype = event.get("type") if isinstance(event, dict) else None
        if etype == "STOP":
            if self.session is not None and not self.session.finished:
                self.session.runner.finish("stopped", failure_summary="节点被 dora 停止")
                self._on_finished()
            return 0
        if etype == "ERROR":
            self._log(f"dora error: {event.get('error')}")
            if self.session is not None and not self.session.finished:
                self.session.runner.finish("failed", failure_summary=f"dora error: {event.get('error')}")
                self._on_finished()
            return 1
        if etype != "INPUT":
            return None

        eid = event.get("id")
        if eid == "bench_command":
            text = event.get("value")
            # dora 文本通道来的是 pyarrow string array，取第一个元素
            if hasattr(text, "as_py"):
                text = text.as_py()
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", errors="replace")
            if isinstance(text, str):
                return self.on_bench_command(text)
            # pa.array(["..."]) 的事件 value 可能是整条 array
            try:
                import pyarrow as pa

                if isinstance(text, pa.Array):
                    text = text[0].as_py()
                    return self.on_bench_command(text)
            except Exception:  # noqa: BLE001
                pass
            self._log(f"bench_command 载荷类型异常：{type(text)}，忽略")
            return None
        if eid == "action":
            session = self.session
            if session is None or session.finished:
                return None
            messages = session.runner.messages
            self.on_action(messages.JointCommand.from_arrow(event.get("value")))
            return None
        # 其它输入只当心跳，绝不推进 episode
        return None

    def run(self) -> int:
        from dora import Node

        node = self.node
        # 启动即把能力包发给工具面（describe 的数据源）。发多次没关系——
        # 工具面以最后一条为准（config 热改后重启 dataflow 也能对上）。
        from _core.session import bench_limits, capabilities_payload

        self._emit(
            "benchmark_capabilities",
            capabilities_payload(self.cfg, self.adapter, limits=bench_limits(self.cfg)),
        )
        self._emit_status(
            f"serve 数据面就绪：benchmark={self.cfg.benchmark} suite={self.cfg.suite} "
            f"{self.adapter.num_tasks} 个 task，等待 bench_command"
        )
        rc = 0
        while True:
            # 必须带 timeout：空闲时（idle 等命令）若无限阻塞，看门狗与
            # dora 心跳都无从驱动。
            event = node.next(timeout=_WATCHDOG_S)
            outcome = self.process_event(event)
            if outcome is not None:
                rc = outcome
                break
        return rc


def run_serve_data(cfg: Any, adapter: Any, log: Callable[[str], None] = print, *, node: Any = None) -> int:
    """serve 模式数据面入口（从 main.py 调，app 已起、adapter 已建）。"""
    from dora import Node

    dora_node = node if node is not None else Node()
    loop = ServeDataLoop(dora_node, cfg, adapter, log=log)
    rc = 0
    try:
        rc = loop.run()
    finally:
        adapter.close()
    return rc
