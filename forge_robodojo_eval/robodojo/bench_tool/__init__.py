"""bench_tool：RoboDojo 评测的 Forge Tool 工具面（py3.12 进程）。

数据面（Isaac Sim，py3.10）在 robodojo 包的 serve.py；两进程经
bench_command / benchmark_status / benchmark_result / benchmark_capabilities
四条 dora 通道协作。gateway 把 HTTP Tool API 翻译到本端点。
"""
