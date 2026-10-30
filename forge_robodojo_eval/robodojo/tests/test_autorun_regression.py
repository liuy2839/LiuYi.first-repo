# -*- coding: utf-8 -*-
"""回归：autorun 行为零破坏。
- _core.node.run_node 的函数体逻辑不变（serve 分流在 robodojo/main.py，_core 不动）
- config 新增 serve 字段默认关闭，老 config.yaml 不含该键时行为不变
- runner 的 finish() 默认关 adapter（autorun 原行为）
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 自适应布局：仓库 benchmarks/{_core,robodojo}/tests 与 zy 部署 robodojo/{_core,tests}。
# 以 serve.py 定位包目录，_core 可能在包内（zy）或包的上一级（仓库）。
CANDIDATES = [HERE.parent, HERE.parent.parent]
PKG = next(c for c in CANDIDATES if (c / "serve.py").exists())
CORE_ROOT = PKG if (PKG / "_core" / "config.py").exists() else PKG.parent
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(HERE))

sys.modules.setdefault("dora", type(sys)("dora"))
_dora_stub = sys.modules["dora"]
if not hasattr(_dora_stub, "Node"):
    _dora_stub.Node = type("Node", (), {})
import pyarrow_stub  # noqa: E402

sys.modules.setdefault("pyarrow", pyarrow_stub)
sys.modules.setdefault("forge_msgs", pyarrow_stub.Msgs())

from _core.config import BaseBenchmarkConfig, fill_base  # noqa: E402


def main():
    # 1) 默认字段
    cfg = BaseBenchmarkConfig()
    assert cfg.serve is False
    assert cfg.max_total_episodes == 0
    assert cfg.max_max_steps == 0
    print("PASS serve 字段默认关闭")

    # 2) 老 config.yaml（无 serve 键）加载后仍关闭
    cfg2 = BaseBenchmarkConfig()
    fill_base(cfg2, {"suite": "x", "num_runs": 1, "max_steps": 5, "seed": 0})
    assert cfg2.serve is False
    print("PASS 老 config 不开启 serve")

    # 3) 新 config 开启
    cfg3 = BaseBenchmarkConfig()
    fill_base(cfg3, {"serve": True, "max_total_episodes": 50, "max_max_steps": 1050})
    assert cfg3.serve is True and cfg3.max_total_episodes == 50
    print("PASS serve 可显式开启")

    # 4) runner 默认 close_adapter_on_finish=True（autorun 行为）
    import inspect
    from _core import runner as runner_mod

    sig = inspect.signature(runner_mod.BenchmarkRunner.__init__)
    assert sig.parameters["close_adapter_on_finish"].default is True
    assert sig.parameters["result_tag"].default is None
    assert "failure_summary" in inspect.signature(runner_mod.BenchmarkRunner.finish).parameters
    print("PASS runner 默认值 = autorun 原行为")

    # 5) node.run_node 仍是唯一 autorun 入口且签名未变
    from _core import node as node_mod

    sig2 = inspect.signature(node_mod.run_node)
    assert list(sig2.parameters) == ["cfg", "adapter", "log"], list(sig2.parameters)
    print("PASS run_node 签名未变")

    print("ALL PASS")


if __name__ == "__main__":
    main()
