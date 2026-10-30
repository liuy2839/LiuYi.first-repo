# -*- coding: utf-8 -*-
"""ServeDataLoop 测试用的配置桩：字段与 RoboDojoConfig 的共有部分对齐。"""
from dataclasses import dataclass, field


@dataclass
class StubConfig:
    benchmark: str = "robodojo"
    suite: str = "stub_suite"
    task_ids: list = field(default_factory=list)
    init_state_ids: list = field(default_factory=list)
    num_runs: int = 1
    max_steps: int = 10
    seed: int = 0
    policy_id: str = "STUB"
    send_instruction: bool = False
    result_dir: str = r"C:\Users\LiuYi\AppData\Local\Temp\paos_check\results"
    record_dir: str = None
    serve: bool = True
    max_total_episodes: int = 0
    max_max_steps: int = 0
    camera_map: dict = field(default_factory=lambda: {"image/head": "cam_head"})

    def episode_plan(self):
        return [
            (t, i, r)
            for t in self.task_ids
            for i in self.init_state_ids
            for r in range(self.num_runs)
        ]
