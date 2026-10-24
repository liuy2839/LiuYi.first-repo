from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from adapters.messages import Image, JointCommand, JointState, PolicyCommand


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_policy_command_schema_and_round_trip(as_bytes: bool) -> None:
    sent = PolicyCommand.from_inputs(
        policy_id="G05", command="set_instruction", inputs={"instruction": "stack bowls"}
    )
    batch = sent.to_arrow()
    assert batch.schema == pa.schema(
        [(name, pa.string()) for name in ("policy_id", "command", "request_id", "inputs_json")]
    )
    received = PolicyCommand.from_arrow(_ipc_bytes(batch) if as_bytes else batch)
    assert received.command == "set_instruction"
    assert received.policy_id == "G05"
    assert received.inputs_json == '{"instruction":"stack bowls"}'


@pytest.mark.parametrize("as_bytes", [False, True])
def test_joint_state_schema_and_round_trip(as_bytes: bool) -> None:
    sent = JointState.from_np(np.array([0.25, -0.5]), ["left_arm", "right_arm"], "position")
    batch = sent.to_arrow()
    assert batch.schema == pa.schema(
        [
            ("name", pa.list_(pa.string())),
            ("position", pa.list_(pa.float64())),
            ("velocity", pa.list_(pa.float64())),
            ("effort", pa.list_(pa.float64())),
        ]
    )
    received = JointState.from_arrow(_ipc_bytes(batch) if as_bytes else batch)
    assert received.name == ["left_arm", "right_arm"]
    assert received.position == [0.25, -0.5]


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("chunk_end", [False, True])
def test_joint_command_schema_and_chunk_marker(as_bytes: bool, chunk_end: bool) -> None:
    sent = JointCommand(
        name=["left_arm", "right_arm"],
        mode="position",
        position=[0.1, -0.2],
        effort=[1.0, 1.0] if chunk_end else [],
    )
    batch = sent.to_arrow()
    assert batch.schema == pa.schema(
        [
            ("name", pa.list_(pa.string())),
            ("mode", pa.string()),
            *((name, pa.list_(pa.float64())) for name in ("position", "velocity", "effort", "kp", "kd")),
        ]
    )
    received = JointCommand.from_arrow(_ipc_bytes(batch) if as_bytes else batch)
    assert received.position == pytest.approx([0.1, -0.2])
    assert (received.effort == [1.0, 1.0]) is chunk_end


@pytest.mark.parametrize("as_bytes", [False, True])
def test_rgb_image_schema_and_round_trip(as_bytes: bool) -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    batch = Image.from_numpy(frame, "rgb8").to_arrow()
    assert batch.schema == pa.schema(
        [
            ("height", pa.uint32()),
            ("width", pa.uint32()),
            ("encoding", pa.string()),
            ("step", pa.uint32()),
            ("data", pa.large_binary()),
        ]
    )
    received = Image.from_arrow(_ipc_bytes(batch) if as_bytes else batch)
    np.testing.assert_array_equal(received.to_numpy(), frame)


def test_invalid_joint_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="match the number"):
        JointCommand(name=["left", "right"], position=[0.1])
