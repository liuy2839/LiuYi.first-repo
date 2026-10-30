"""策略抽象接口：算法实现这四个方法，Dora 收发全部由 PolicyRuntime 承担。

obs（一帧完整观测）的结构：
    {
        "state": np.ndarray[float64],          # 关节状态向量
        "state_names": list[str],              # 与 state 对齐的分量名
        "images": {cam: np.ndarray[H,W,3] uint8},   # cam = contract.cameras 中的名字
        "depths": {cam: np.ndarray[H,W] float32},   # meters，可选
        "segs":   {cam: np.ndarray[H,W] int},       # 实例标签，可选
    }
step 返回：
    np.ndarray[len(action_names)]  # 本帧动作（position 语义）
    None                           # 本帧不动作（如规划中），评测端 watchdog 会重发观测
抛异常：
    PolicyError                    # 结构化失败：runtime 停发动作并上报 policy_status=failed
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PolicyError(RuntimeError):
    """策略的结构化失败（不可恢复到本 episode 内），带稳定错误码。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Policy:
    """算法侧唯一需要实现的接口。"""

    def reset_episode(self) -> None:
        """评测端 reset_scene：清空 episode 内的计划/观测缓存/动作队列。
        不重载模型、不退出进程。"""

    def set_instruction(self, text: str) -> None:
        """接收本 episode 的自然语言任务指令；下一次 step 前应触发重规划。"""

    def step(self, obs: dict[str, Any]) -> np.ndarray | None:
        """消费一帧完整观测，返回一个动作向量或 None。"""
        raise NotImplementedError

    def shutdown(self) -> None:
        """stop/cancel 后的清理；进程即将退出。"""
