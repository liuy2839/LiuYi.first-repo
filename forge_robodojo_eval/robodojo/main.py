#!/usr/bin/env python3
"""RoboDojo benchmark 节点（Dora）。调度核心在 _core，这里只负责组装。

**启动顺序有硬约束**：Isaac Sim 的 `AppLauncher` 必须在 import 任何
isaaclab/omni 相关模块之前跑完，否则 Kit 起不来。所以本文件的 import 是分段的，
不要为了"整洁"把下半段的 import 提到顶上。官方 src/eval_client/main.py 同理。
"""

from __future__ import annotations

import argparse
import os
import sys

# Benchmark package and shared core import setup.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
# _HERE 必须优先（插到最前）：部署布局 robodojo/_core 才是最新核心；
# _PARENT 只做兜底（仓库布局 benchmarks/_core），用 append 而不是 insert(0)——
# 否则 EVAL 级残留的旧 _core 会遮蔽包内的新版（serve 字段被吃掉 → 误入
# autorun，实测就是这么丢的）。
if _PARENT not in sys.path:
    sys.path.append(_PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _log(msg: str) -> None:
    print(f"[robodojo_bench] {msg}", flush=True)


def main() -> int:
    # **先把配置路径定死成绝对路径**：下面会 os.chdir 到 RoboDojo 仓库根，
    # 之后任何相对路径都会解析到那边去。dora 传进来的 BENCHMARK_CONFIG 通常写
    # `./config.yaml`，不先解析就会变成去 RoboDojo 仓库根找，报 FileNotFoundError。
    config_path = os.environ.get("BENCHMARK_CONFIG", "config.yaml")
    if not os.path.isabs(config_path):
        config_path = os.path.join(_HERE, config_path)

    # RoboDojo 仓库根：env/ task/ utils/ src/ 都在那儿，必须进 sys.path。
    dojo_root = os.environ.get("ROBODOJO_ROOT")
    if not dojo_root or not os.path.isdir(dojo_root):
        raise RuntimeError("必须设置 ROBODOJO_ROOT 指向 RoboDojo 仓库根目录")
    if dojo_root not in sys.path:
        sys.path.insert(0, dojo_root)
    # 官方代码里到处用相对路径（eval_result/ 等），CWD 必须是仓库根。
    os.chdir(dojo_root)

    from config import load_config  # noqa: E402  (本仓库的 config.py)

    cfg = load_config(config_path)
    # 输出路径同理：chdir 之后相对路径会落到 RoboDojo 仓库里去，一律先解析成绝对路径。
    for attr in ("record_dir", "result_dir"):
        v = getattr(cfg, attr, None)
        if isinstance(v, str) and v and not os.path.isabs(v):
            setattr(cfg, attr, os.path.normpath(os.path.join(_HERE, v)))

    # ---- 第一段结束：下面开始碰 Isaac Sim ----
    from isaaclab.app import AppLauncher  # noqa: E402

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = True
    app_args.device = f"cuda:{cfg.device_id}"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    simulation_app = AppLauncher(app_args).app
    _log(f"Isaac Sim 已启动 (device=cuda:{cfg.device_id}, headless)")

    # ---- 第二段：Kit 起来了，现在才能 import 依赖它的东西 ----
    from _core.node import run_node  # noqa: E402
    from adapters.robodojo_adapter import RoboDojoAdapter  # noqa: E402

    adapter = RoboDojoAdapter(cfg, simulation_app, log=_log)
    _log(
        f"suite={cfg.suite} → {adapter.num_tasks} 个 task | "
        f"env_cfg={cfg.env_cfg_type} | 每 task {cfg.init_states_per_task} 个 seed"
    )
    try:
        if getattr(cfg, "serve", False):
            # serve 模式：工具面（bench_tool，py3.12 进程）经 bench_command 驱动。
            # 本进程只做数据面：跑批次、回状态/结果。
            from serve import run_serve_data  # noqa: E402

            return run_serve_data(cfg, adapter, log=_log)
        return run_node(cfg, adapter, log=_log)
    finally:
        adapter.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
