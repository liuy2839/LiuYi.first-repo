"""PyInstaller 入口：打包后的单文件节点从这里启动。

用法与源码运行一致：rekep_policy --profile ... --cameras image/head
"""

from __future__ import annotations

import sys

from rekep_policy.main import main

if __name__ == "__main__":
    sys.exit(main())
