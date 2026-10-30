"""serve 数据面状态机单测（不依赖 dora/forge_tool/isaac，全桩）。

zy 服务器跑法：
    cd packages/nodes/benchmarks
    python -m pytest robodojo/tests/test_serve_data.py -q
或直接 python robodojo/tests/test_serve_data.py

覆盖：
1. run 命令 → 建 session → started 状态、终态 benchmark_result（含 result_path）
2. stop 路径无崩溃
3. 非法/未知 bench_command 容忍
4. capabilities_payload 字段齐全
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 自适应布局：仓库 benchmarks/{_core,robodojo}/tests 与 zy 部署 robodojo/{_core,tests}。
# 以 serve.py 定位包目录；_core 可能在包内（zy）或包的上一级（仓库）。
CANDIDATES = [HERE.parent, HERE.parent.parent]
PKG = next(c for c in CANDIDATES if (c / "serve.py").exists())
CORE_ROOT = PKG if (PKG / "_core" / "config.py").exists() else PKG.parent
for p in (str(CORE_ROOT), str(PKG), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pyarrow_stub  # noqa: E402  (tests 目录内的桩)

sys.modules.setdefault("dora", type(sys)("dora"))
sys.modules.setdefault("pyarrow", pyarrow_stub)
sys.modules.setdefault("forge_msgs", pyarrow_stub.Msgs())

from robodojo_config_stub import StubConfig  # noqa: E402
from serve import CMD_RUN, CMD_STOP, ServeDataLoop  # noqa: E402


class FakeNode:
    def __init__(self):
        self.outputs = []

    def send_output(self, channel, value):
        self.outputs.append((channel, value))

    def next(self, timeout=None):
        raise KeyboardInterrupt


class FakeAdapter:
    @property
    def num_tasks(self):
        return 2

    def num_init_states(self, task_id):
        return 2

    def task_language(self, task_id):
        return "stub task"

    def state_joint_names(self):
        return ["j0", "j1"]

    def reproducibility(self):
        return {"stub": True}

    def reset_episode(self, task_id, init_state_id, run_index, episode_seed):
        return {"instruction": f"task{task_id}"}

    def step(self, action, chunk_end=False):
        class O:
            obs = {}
            reward = 1.0
            done = True
            success = True
            next_instruction = None
            metrics = {"success": True}

        return O()

    def to_state(self, obs):
        return [0.0, 0.1]

    def to_images(self, obs):
        return {}

    def close(self):
        pass


class _JointCommand:
    def __init__(self, names, position):
        self.name = names
        self.position = position
        self.effort = []


def text_of(value):
    arr = pyarrow_stub.array
    return value[0] if isinstance(value, arr) else value


def drive_to_finish(loop, names=("j0", "j1")):
    for _ in range(100):
        session = loop.session
        if session is None or session.finished:
            return
        loop.on_action(_JointCommand(list(names), [0.0, 0.0]))
    raise AssertionError("100 步还没跑完，疑似死循环")


def test_run_to_succeeded():
    cfg = StubConfig()
    node = FakeNode()
    adapter = FakeAdapter()
    loop = ServeDataLoop(node, cfg, adapter, log=lambda m: None)

    rc = loop.on_bench_command(
        json.dumps(
            {
                "command": CMD_RUN,
                "invocation_id": "inv-1",
                "plan": {
                    "task_ids": [0, 1],
                    "init_state_ids": [0],
                    "num_runs": 1,
                    "max_steps": 10,
                    "seed": 0,
                },
            }
        )
    )
    assert rc is None
    drive_to_finish(loop)
    statuses = [(c, text_of(v)) for c, v in node.outputs if c == "benchmark_status"]
    assert any("started" in s for _, s in statuses), statuses
    results = [v for c, v in node.outputs if c == "benchmark_result"]
    assert results
    payload = json.loads(text_of(results[-1]))
    assert payload["status"] == "succeeded"
    assert payload["completed_episodes"] == 2
    assert payload["invocation_id"] == "inv-1"
    assert payload["result_path"]


def test_stop_and_malformed():
    cfg = StubConfig()
    node = FakeNode()
    loop = ServeDataLoop(node, cfg, FakeAdapter(), log=lambda m: None)
    rc = loop.on_bench_command(
        json.dumps(
            {
                "command": CMD_RUN,
                "invocation_id": "inv-2",
                "plan": {
                    "task_ids": [0],
                    "init_state_ids": [0, 1],
                    "num_runs": 1,
                    "max_steps": 10,
                    "seed": 0,
                },
            }
        )
    )
    drive_to_finish(loop)
    assert loop.on_bench_command(json.dumps({"command": CMD_STOP, "reason": "t"})) is None
    assert loop.on_bench_command("{bad json") is None
    assert loop.on_bench_command(json.dumps({"command": "wat"})) is None


def test_capabilities_payload():
    from _core.session import capabilities_payload

    caps = capabilities_payload(
        StubConfig(), FakeAdapter(), limits={"max_total_episodes": 50}
    )
    for k in (
        "benchmark",
        "suite",
        "num_tasks",
        "tasks",
        "joint_names",
        "camera_ids",
        "defaults",
        "limits",
        "reproducibility",
    ):
        assert k in caps, f"caps 缺 {k}"
    assert caps["num_tasks"] == 2 and caps["tasks"][0]["num_init_states"] == 2


if __name__ == "__main__":
    test_run_to_succeeded()
    print("PASS run → succeeded")
    test_stop_and_malformed()
    print("PASS stop / malformed")
    test_capabilities_payload()
    print("PASS capabilities_payload")
    print("ALL PASS")
