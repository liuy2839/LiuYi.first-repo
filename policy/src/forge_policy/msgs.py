"""消息载体：优先使用 forge_msgs 包；不可得时回退到字段级兼容的本地实现。

本地回退对齐 forge-msgs 1.0.1 的 Arrow 字段布局（与
benchmarks/robodojo/adapters/messages.py 的镜像一致），仅用于无法安装
私有 git 依赖的开发/测试环境；目标部署环境安装 forge_msgs 后自动走正式包。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa

try:  # 正式包优先
    from forge_msgs import Image, JointCommand, JointState, PolicyCommand  # type: ignore

    USING_FORGE_MSGS = True
except ImportError:  # pragma: no cover - 开发机回退
    USING_FORGE_MSGS = False

    def _batch(value: Any) -> pa.RecordBatch:
        if isinstance(value, pa.RecordBatch):
            return value
        if isinstance(value, pa.Table):
            batches = value.to_batches()
            if not batches:
                raise ValueError("Arrow table is empty")
            return batches[0]
        if isinstance(value, pa.StructArray):
            return pa.RecordBatch.from_arrays(
                [value.field(i) for i in range(value.type.num_fields)],
                names=[value.type.field(i).name for i in range(value.type.num_fields)],
            )
        if isinstance(value, bytes):
            return pa.ipc.open_stream(value).read_next_batch()
        raise TypeError(f"Expected Arrow batch/table/struct/bytes; got {type(value)}")

    def _cell(batch: pa.RecordBatch, name: str) -> Any:
        if batch.num_rows != 1:
            raise ValueError(f"Expected one {name} message, got {batch.num_rows} rows")
        return batch[name][0].as_py()

    def _names(names: list[str]) -> None:
        if not names or len(set(names)) != len(names):
            raise ValueError("Joint names must be nonempty and unique")

    def _vector(names: list[str], values: list[float], name: str) -> None:
        if values and len(values) != len(names):
            raise ValueError(f"{name} must be empty or match the number of joint names")

    @dataclass
    class PolicyCommand:  # type: ignore[no-redef]
        policy_id: str
        command: str
        request_id: str = ""
        inputs_json: str = "{}"

        def __post_init__(self) -> None:
            if not self.policy_id or not re.fullmatch(r"[a-z][a-z0-9_]*", self.command):
                raise ValueError("Policy command needs a policy ID and snake_case command")
            if not isinstance(json.loads(self.inputs_json), dict):
                raise ValueError("inputs_json must be a JSON object")

        @classmethod
        def from_inputs(
            cls,
            *,
            policy_id: str,
            command: str,
            inputs: dict | None = None,
            request_id: str = "",
        ):
            return cls(
                policy_id,
                command,
                request_id=request_id,
                inputs_json=json.dumps(inputs or {}, separators=(",", ":")),
            )

        def to_arrow(self) -> pa.RecordBatch:
            return pa.RecordBatch.from_pydict(
                {
                    "policy_id": pa.array([self.policy_id], type=pa.string()),
                    "command": pa.array([self.command], type=pa.string()),
                    "request_id": pa.array([self.request_id], type=pa.string()),
                    "inputs_json": pa.array([self.inputs_json], type=pa.string()),
                }
            )

        @classmethod
        def from_arrow(cls, value: Any):
            batch = _batch(value)
            return cls(*(_cell(batch, n) for n in ("policy_id", "command", "request_id", "inputs_json")))

    @dataclass
    class JointState:  # type: ignore[no-redef]
        name: list[str]
        position: list[float] = field(default_factory=list)
        velocity: list[float] = field(default_factory=list)
        effort: list[float] = field(default_factory=list)

        def __post_init__(self) -> None:
            _names(self.name)
            for key in ("position", "velocity", "effort"):
                _vector(self.name, getattr(self, key), key)

        def to_arrow(self) -> pa.RecordBatch:
            return pa.RecordBatch.from_pydict(
                {
                    "name": pa.array([self.name], type=pa.list_(pa.string())),
                    "position": pa.array([self.position], type=pa.list_(pa.float64())),
                    "velocity": pa.array([self.velocity], type=pa.list_(pa.float64())),
                    "effort": pa.array([self.effort], type=pa.list_(pa.float64())),
                }
            )

        @classmethod
        def from_arrow(cls, value: Any):
            batch = _batch(value)
            return cls(**{n: _cell(batch, n) or [] for n in ("name", "position", "velocity", "effort")})

    @dataclass
    class JointCommand:  # type: ignore[no-redef]
        name: list[str]
        mode: str = "position"
        position: list[float] = field(default_factory=list)
        velocity: list[float] = field(default_factory=list)
        effort: list[float] = field(default_factory=list)
        kp: list[float] = field(default_factory=list)
        kd: list[float] = field(default_factory=list)

        def __post_init__(self) -> None:
            _names(self.name)
            if self.mode not in ("position", "velocity", "effort", "hybrid"):
                raise ValueError(f"Unsupported joint command mode: {self.mode}")
            for key in ("position", "velocity", "effort", "kp", "kd"):
                _vector(self.name, getattr(self, key), key)

        def to_arrow(self) -> pa.RecordBatch:
            return pa.RecordBatch.from_pydict(
                {
                    "name": pa.array([self.name], type=pa.list_(pa.string())),
                    "mode": pa.array([self.mode], type=pa.string()),
                    "position": pa.array([self.position], type=pa.list_(pa.float64())),
                    "velocity": pa.array([self.velocity], type=pa.list_(pa.float64())),
                    "effort": pa.array([self.effort], type=pa.list_(pa.float64())),
                    "kp": pa.array([self.kp], type=pa.list_(pa.float64())),
                    "kd": pa.array([self.kd], type=pa.list_(pa.float64())),
                }
            )

        @classmethod
        def from_arrow(cls, value: Any):
            batch = _batch(value)
            return cls(
                name=_cell(batch, "name") or [],
                mode=_cell(batch, "mode") or "position",
                **{n: _cell(batch, n) or [] for n in ("position", "velocity", "effort", "kp", "kd")},
            )

    @dataclass
    class Image:  # type: ignore[no-redef]
        height: int
        width: int
        encoding: str
        step: int
        data: bytes

        def __post_init__(self) -> None:
            if self.height < 0 or self.width < 0:
                raise ValueError("Image dimensions must be nonnegative")
            if self.step < self.width * 3 and self.encoding == "rgb8":
                raise ValueError("Image step is inconsistent with its dimensions")
            if len(self.data) != self.step * self.height:
                raise ValueError("Image data length is inconsistent with its dimensions")

        @classmethod
        def from_numpy(cls, frame: np.ndarray, encoding: str = "rgb8"):
            if encoding != "rgb8" or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Image.from_numpy expects an HxWx3 uint8 rgb8 frame")
            height, width, _ = frame.shape
            return cls(height, width, encoding, width * 3, np.ascontiguousarray(frame).tobytes())

        def to_numpy(self) -> np.ndarray:
            if self.encoding == "32FC1":
                return np.frombuffer(self.data, dtype=np.float32).reshape(self.height, self.width).copy()
            if self.encoding == "mono16":
                return np.frombuffer(self.data, dtype=np.uint16).reshape(self.height, self.width).astype(np.int32).copy()
            if self.step == self.width * 3:
                return np.frombuffer(self.data, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
            return np.ndarray(
                (self.height, self.width, 3),
                dtype=np.uint8,
                buffer=self.data,
                strides=(self.step, 3, 1),
            ).copy()

        def to_arrow(self) -> pa.RecordBatch:
            return pa.RecordBatch.from_arrays(
                [
                    pa.array([self.height], type=pa.uint32()),
                    pa.array([self.width], type=pa.uint32()),
                    pa.array([self.encoding], type=pa.string()),
                    pa.array([self.step], type=pa.uint32()),
                    pa.array([self.data], type=pa.large_binary()),
                ],
                names=["height", "width", "encoding", "step", "data"],
            )

        @classmethod
        def from_arrow(cls, value: Any):
            batch = _batch(value)
            return cls(*(_cell(batch, n) for n in ("height", "width", "encoding", "step", "data")))


__all__ = ["Image", "JointCommand", "JointState", "PolicyCommand", "USING_FORGE_MSGS"]
