"""zy 上对 robodojo-bench 的 gateway 冒烟探针（tools→describe→小批 run→轮询 result）。

用法（zy, conda paos）：
    python3 /data0/liuyi/forge_robodojo_eval/scripts/gateway_probe_robodojo.py [--smoke]
默认只列 tools + describe；--smoke 额外跑单集批次（task0/init0/max_steps 100）。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:19021"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def req(method: str, path: str, body=None, timeout: float = 60.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    smoke = "--smoke" in sys.argv
    payload = req("GET", "/tools")
    tools = [t.get("tool_id") for t in payload.get("data", {}).get("tools", [])]
    print("TOOLS:", tools)
    assert "benchmark.describe" in tools and "benchmark.run" in tools, tools

    desc = req(
        "POST",
        "/tools/benchmark.controller/describe:invoke",
        {"arguments": {}, "caller_id": "probe", "timeout_ms": 30000},
    )
    data = desc.get("data", {})
    print("DESCRIBE:", json.dumps(data, ensure_ascii=False)[:400])

    if not smoke:
        print("PROBE_OK (no --smoke, skip run)")
        return 0

    run = req(
        "POST",
        "/tools/benchmark.run:invoke",
        {
            "arguments": {
                "task_ids": [0],
                "init_state_ids": [0],
                "num_runs": 1,
                "max_steps": 100,
                "seed": 0,
            },
            "caller_id": "probe",
            "timeout_ms": 1800000,
        },
    )
    invocation = run.get("data", {}).get("invocation_id")
    print("RUN invocation:", invocation)
    for _ in range(240):  # 最多 20 分钟
        result = req("GET", f"/invocations/{invocation}/result")
        status = result.get("data", {}).get("status")
        if status == "available":
            terminal = result["data"]["result"]
            print("RESULT:", json.dumps(terminal.get("outputs", {}), ensure_ascii=False))
            return 0 if terminal.get("status") == "succeeded" else 1
        time.sleep(5)
    print("RUN_TIMEOUT")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
