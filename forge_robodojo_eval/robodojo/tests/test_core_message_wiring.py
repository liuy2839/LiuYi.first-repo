from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _core.runner import BenchmarkRunner  # noqa: E402
from adapters import messages  # noqa: E402


class FakeNode:
    def __init__(self) -> None:
        self.outputs: dict[str, object] = {}

    def send_output(self, name: str, value: object) -> None:
        self.outputs[name] = value


def test_runner_uses_robodojo_messages_without_forge_msgs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "forge_msgs", None)
    node = FakeNode()
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    adapter = SimpleNamespace(
        messages=messages,
        state_joint_names=lambda: ["left_arm", "right_arm"],
        action_component_names=lambda: ["left_arm", "right_arm"],
        step=lambda action, chunk_end=False: None,
        to_state=lambda obs: np.array([0.25, -0.5]),
        to_images=lambda obs: {"image/head": frame},
    )
    cfg = SimpleNamespace(
        episode_plan=lambda: [],
        benchmark="RoboDojo",
        suite="smoke",
        max_steps=1,
        seed=0,
        record_dir=None,
        policy_id="G05",
    )

    runner = BenchmarkRunner(node, cfg, adapter)
    runner.policy_command("set_instruction", {"instruction": "stack bowls"})
    runner._publish_obs({})

    assert runner.messages is messages
    assert messages.PolicyCommand.from_arrow(node.outputs["policy_command"]).command == "set_instruction"
    assert messages.JointState.from_arrow(node.outputs["proprio_state"]).position == [0.25, -0.5]
    np.testing.assert_array_equal(messages.Image.from_arrow(node.outputs["image/head"]).to_numpy(), frame)


def test_other_benchmarks_keep_forge_msgs_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_msgs = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "forge_msgs", fake_msgs)
    cfg = SimpleNamespace(
        episode_plan=lambda: [],
        benchmark="other",
        suite="smoke",
        max_steps=1,
        seed=0,
        record_dir=None,
    )
    adapter = SimpleNamespace(state_joint_names=lambda: [], step=lambda action: None)

    runner = BenchmarkRunner(FakeNode(), cfg, adapter)
    assert runner.messages is fake_msgs


def test_episode_start_failure_logs_traceback() -> None:
    logs: list[str] = []
    statuses: list[str] = []
    cfg = SimpleNamespace(
        episode_plan=lambda: [(0, 0, 0)],
        benchmark="RoboDojo",
        suite="smoke",
        max_steps=1,
        seed=0,
        record_dir=None,
    )
    adapter = SimpleNamespace(
        messages=messages,
        state_joint_names=lambda: [],
        step=lambda action: None,
        task_language=lambda task_id: (_ for _ in ()).throw(ValueError("bad asset path")),
    )
    runner = BenchmarkRunner(FakeNode(), cfg, adapter, log=logs.append)
    runner.finish = statuses.append

    runner.advance()

    assert statuses == ["failed"]
    assert any("启动下一 episode 失败：ValueError: bad asset path" in line for line in logs)
    assert any("Traceback (most recent call last):" in line for line in logs)
    assert any("test_core_message_wiring.py" in line for line in logs)
    assert any("ValueError: bad asset path" in line for line in logs)
