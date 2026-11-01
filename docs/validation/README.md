# 验证记录索引

公开报告保留各次实验的源码版本、环境和硬件范围；测试跳过与实际执行分别统计。
当前性能结论以 2026-11-01、源码 `567ecf8` 的固定四卡拓扑 B 为准。

| 日期 / 记录 | 范围 |
|---|---|
| [公开整理回归](public-cleanup-20261101.json) | 文档与命名整理后的 CPU 回归：105 项测试、41 项启动器检查通过，4 项 GPU 用例跳过 |
| [四卡报告](four-gpu-20261101.md) / [JSON](four-gpu-20261101.json) | 4×RTX 4090，拓扑 A/B 功能验收；拓扑 B 的 NCCL 并发 1/2/4、Gloo 并发 2 正式基准 |
| [实现与修正记录](implementation-notes.md) | 交接、输入边界、探针退出判定、并发路由与缓存 batch 形状修复 |
| [并发路由摘要](parallel-routing-20261031.json) | 本机并发、背压、SSE 与进程回收；历史单卡模型冒烟 |
| [设备池摘要](device-pools-20261031.json) | 设备池展开、候选边和单卡回环，不能替代真实四卡验收 |
| [架构回归摘要](architecture-review-20261031.json) | 当时的 CPU/控制面测试和探针回归 |
| [三阶段实现摘要](three-gpu-implementation-20261031.json) | 历史独立设备映射与通信实现验证 |

原始 JSON/JSONL、GPU 遥测和日志保留在实验机，未随仓库分发；摘要中的绝对路径仅用于追溯原环境。
仓库不包含模型权重。重新运行请使用 [多卡验收工具](../../utils/validate_vpd_multigpu.py) 和新的 `metrics/` 输出目录。
本机单卡/CPU 验证不能推导跨卡性能，NCCL 通过也不代表直接 GPU P2P 或 NVLink。
