# -*- coding: utf-8 -*-
"""测试辅助：优先使用真实依赖，缺失时才回退到 tests 目录内的桩。

背景：原先用 ``sys.modules.setdefault("pyarrow", 桩)``，当真实 pyarrow 尚未被导入时
会先把 ``pyarrow`` 注册成桩，同进程后续用例（如 test_messages.py）就拿到桩并报
``module 'pyarrow_stub' has no attribute 'RecordBatch'``。这里显式「先尝试真实包」。
"""

from __future__ import annotations

import importlib
import sys


def ensure_module(name, fallback, *, require=()):
    """确保 sys.modules[name] 可用；真实包缺失或缺关键属性时才安装 fallback。"""
    mod = sys.modules.get(name)
    if mod is None:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 缺包/导入失败都回退
            mod = None
    if mod is None or any(not hasattr(mod, attr) for attr in require):
        mod = fallback()
        sys.modules[name] = mod
    return mod