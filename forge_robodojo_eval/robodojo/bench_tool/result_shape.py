"""benchmark.controller/run 的 ToolResult 输出契约（七字段）。

字段集合与 Skill Bundle 里 gateway.yaml 的 output_schema 一一对应：
多一个少一个都会让下游 schema 校验失败，改这里必须同时改 gateway.yaml。

outputs.status 是**给 Agent 看的批次结论**，枚举 {succeeded, failed, partial,
stopped}；ToolResult.status 是 forge 的执行结论，只能是 {succeeded, failed,
stopped}。批次被 stop 时两者分叉：completed_episodes>0 → partial，
一集没跑完 → stopped。
"""

from __future__ import annotations

from typing import Any


def stopped_summary(*, completed: int, total: int, reason: str | None) -> str:
    summary = f"stopped after {completed}/{total} episodes"
    return f"{summary}（{reason}）" if reason else summary


def build_outputs(
    *,
    run_id: str,
    batch_status: str,
    completed_episodes: int,
    total_episodes: int,
    success_rate: float,
    result_file: str | None,
    failure_summary: str | None,
) -> dict[str, Any]:
    status = batch_status
    if batch_status == "stopped":
        status = "partial" if completed_episodes > 0 else "stopped"
    return {
        "run_id": run_id,
        "status": status,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "success_rate": success_rate,
        "result_file": result_file,
        "failure_summary": failure_summary,
    }


def outputs_from_payload(
    *,
    run_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """从数据面经 benchmark_result 通道发来的结果 JSON 组装七字段输出。

    payload 就是 benchmark_execution_result_v1 + result_path/invocation_id。
    """
    completed = int(payload.get("completed_episodes") or 0)
    total = int(payload.get("total_episodes") or 0)
    status = str(payload.get("status") or "failed")
    # 数据面把批次跑完是 succeeded；半路停是 stopped；异常是 failed。
    # execution JSON 的 status 枚举与 ToolResult 对齐靠 result_shape 外的
    # 映射（endpoints.on_finished），这里只管七字段。
    failure_summary = payload.get("failure_summary")
    if not failure_summary and status == "stopped":
        failure_summary = stopped_summary(
            completed=completed, total=total, reason=payload.get("stop_reason")
        )
    return build_outputs(
        run_id=run_id,
        batch_status=status,
        completed_episodes=completed,
        total_episodes=total,
        success_rate=float(payload.get("success_rate") or 0.0),
        result_file=payload.get("result_path"),
        failure_summary=failure_summary,
    )


__all__ = ["build_outputs", "outputs_from_payload", "stopped_summary"]
