# 文档导航

从 [项目首页](../README.md) 了解能力、实测结果与快速启动，再按需要阅读下列文档。

| 文档 | 内容 |
|---|---|
| [架构设计](architecture.md) | 阶段权重、设备池、路由、交接协议、数据所有权与失败处理 |
| [单卡与接口使用](vpd_local.md) | 图片请求、HTTP/SSE、资源限额及单卡正确性验证 |
| [多卡部署与验收](vpd_multigpu.md) | Gloo/NCCL、设备映射、验收探针和并发基准命令 |
| [目录与代码导航](project_structure.md) | 主线模块、共享设施、保留的上游与参考路径 |
| [验证记录索引](validation/README.md) | 当前四卡实测、机器可读摘要及历史实现验证 |
| [历史单卡缓存消融](vpd_benchmark.md) | 旧版本、RTX 4060 Laptop / Gloo / 并发 1 的独立结果 |
| [早期文本 PD 后端](legacy/pd_local.md) | Qwen3 双进程 Gloo 参考路径 |
| [上游来源与部署](upstream/README.md) | ModServe 原实现、依赖环境、配置与贡献边界 |

`docs/validation/` 保留版本、日期和硬件条件明确的公开摘要。
原始实验日志与 JSONL 位于实验机器的 `metrics/`，未随仓库分发；不能把摘要中记录的历史路径当作克隆后已有的文件。
个人简历、学习笔记和面试材料保留在本地忽略目录，不属于项目文档。
