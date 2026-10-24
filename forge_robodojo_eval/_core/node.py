"""Dora 节点入口。各 benchmark 的 main.py 只需调用 run_node()。"""

from __future__ import annotations

from typing import Any, Callable

from dora import Node

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner

# 等 action 的看门狗周期（秒）。取值要**明显大于**最慢策略的单步推理耗时，
# 否则会在正常推理途中误重发：pi05(3B) 实测首帧约 2 s，这里留足余量。
_WATCHDOG_S = 20.0


def _is_timeout(event: Any) -> bool:
    """dora 把 next(timeout=) 的超时表达成 ERROR 事件而不是 None。"""
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


def run_node(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    """事件循环。无论怎么退出，结果都必须落盘。"""
    # fail fast：越界的 task/init_state 必须在建 Node 之前炸掉
    resolve_and_validate_plan(cfg, adapter)

    node = Node()
    runner = BenchmarkRunner(node, cfg, adapter, log=log)
    log(
        f"benchmark={cfg.benchmark} suite={cfg.suite} tasks={cfg.task_ids} "
        f"init_states={len(cfg.init_state_ids)} num_runs={cfg.num_runs} "
        f"seed={cfg.seed} → {len(runner.plan)} episodes"
    )

    rc = 0
    try:
        runner.policy_command("start")
        runner.advance()

        while not runner.finished:
            event = node.next(timeout=_WATCHDOG_S)

            # 超时：要么 policy 真的很慢，要么这一份观测丢了/被清了。两种情况
            # 重发一次都无害（policy 只是拿到同一帧再推一次），不重发的话后者
            # 会双向死锁 —— 见 runner.republish_if_stalled 的注释。
            #
            # **坑**：dora 的 next(timeout=) 超时不返回 None，而是抛一个 ERROR
            # 事件 "Timeout event stream error: Receiver timed out"。照抄
            # 文档里的 `if event is None` 会让超时走进致命错误分支，把整批判失败
            # （实测就这么丢过一轮 20 集）。所以两种形态都要认。
            if event is None or _is_timeout(event):
                runner.republish_if_stalled(_WATCHDOG_S)
                continue

            if event["type"] == "INPUT":
                if event["id"] == "action":
                    try:
                        runner.on_action(runner.messages.JointCommand.from_arrow(event["value"]))
                    except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
                        runner.fail_current("step_error", f"{type(exc).__name__}: {exc}")
            elif event["type"] == "STOP":
                runner.finish("cancelled")
                break
            elif event["type"] == "ERROR":
                log(f"dora error: {event.get('error')}")
                runner.finish("failed")
                rc = 1
                break
    finally:
        runner.finish("cancelled")

    return rc
