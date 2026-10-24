from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("command", ["stop", "cancel"])
def test_policy_exits_on_terminal_command(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    robodojo_dir = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(robodojo_dir))

    class FakeNode:
        def __iter__(self):
            yield {"type": "INPUT", "id": "policy_command", "value": command}
            raise AssertionError("policy requested another event after terminal command")

    fake_dora = ModuleType("dora")
    fake_dora.Node = FakeNode
    monkeypatch.setitem(sys.modules, "dora", fake_dora)

    spec = importlib.util.spec_from_file_location("robodojo_policy_shutdown_test", robodojo_dir / "policy_node.py")
    assert spec is not None and spec.loader is not None
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)

    logs: list[str] = []
    monkeypatch.setattr(policy, "_build_model", lambda: object())
    monkeypatch.setattr(policy, "_build_state_spec", lambda *_: (("arm", 1),))
    monkeypatch.setattr(policy, "_build_obs_spec", lambda *_: (("arm", 1),))
    monkeypatch.setattr(policy.PolicyCommand, "from_arrow", lambda value: SimpleNamespace(command=value))
    monkeypatch.setattr(policy, "log", logs.append)

    assert policy.main() == 0
    assert f"收到 {command}，退出" in logs
    assert "退出，共发出 0 个动作" in logs
