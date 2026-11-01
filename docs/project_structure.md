# 目录与代码导航

项目名称为 xixi-server，Python 包路径保留 `src/lmm_serving/`。当前主线是 SmolVLM V/P/D 三阶段服务。

## 根目录

| 路径 | 职责 |
|---|---|
| `src/lmm_serving/` | 服务、模型阶段、路由和传输协议 |
| `tests/` | 模型对齐、通信、设备池、背压、SSE、进程回收及保留路径的回归 |
| `utils/` | 验收、基准、传输探针、回归运行器及上游部署工具 |
| `docs/` | 架构、接口、部署、验证报告；`upstream/` 与 `legacy/` 单独保存参考路径 |
| `configs/` | 上游 ModServe/Monolith 拓扑和 CPU 模拟模式配置 |
| `llm-backend/` | 上游固定版本 vLLM 的构建脚本与补丁 |
| `client/`、`traces/` | 上游轨迹回放客户端与小型输入样本 |
| `profiles/` | 上游容量参考数据，不能作为本后端实测 |
| `requirements-vpd-cuda.txt` | 当前 CUDA 主线的已验证依赖 |
| `requirements-pd-local.txt` | 随机微型模型、PD/VPD 正确性测试的 CPU 依赖 |
| `requirements-learning.txt` | 早期 CPU 模拟链路依赖 |
| `requirements.txt`、`pixi.toml`、`pixi.lock` | 保留的上游环境，与当前主线分开 |
| `pyproject.toml` | `xixi-server` 发行包元数据，兼容原 Pixi editable 安装 |
| `LICENSE` | 保留的上游 MIT 许可证 |
| `AGENTS.md` | 面向开发工具的项目约定和验证入口 |

## 主线阅读顺序

| 文件 | 内容 |
|---|---|
| `vpd_service.py`、`vpd_options.py` | HTTP 适配、启动选项与接纳配置 |
| `vpd_local.py` | 输入准备、模型阶段、worker 执行与请求编排 |
| `vpd_router.py` | 空闲副本选择、整条路由预留、阶段槽位释放 |
| `vpd_transport.py` | 视觉/KV 布局、张量通道、接收检查和 ACK |
| `vision_cache.py` | CPU 字节级 LRU 与独立副本管理 |
| `pd_local.py`、`pd_service.py` | 共享进程/HTTP 基础设施及早期文本 PD 后端 |
| `monolithic_local.py` | 持久单体模型对照，使用相同输入准备和生成策略 |
| `stream_benchmark.py` | HTTP/SSE 逐请求测量与结果记录 |

模块均位于 `src/lmm_serving/`。先看 [架构时序](architecture.md)，再沿交接边界阅读实现。

## 保留路径

`modserve/`、`monolith/`、`profiler/` 为原版服务及其派生修改，依赖打补丁的 vLLM；
`common/` 包含这些路径的共享调度、传输预算和扩缩容逻辑。
`launcher/` 同时支持原版、早期文本 PD 与当前 VPD 入口。
`learning/` 是 CPU/Gloo 模拟链路，保留源码和回归测试，其输出不属于真实模型性能证据。
来源、部署和范围见 [上游说明](upstream/README.md)。

## 本地文件

`.venv*/`、`metrics/`、`.local/`、模型下载和编辑器配置由 `.gitignore` 排除。
个人笔记放 `.local/notes/`，私人完整简历保持本地且不跟踪。
实验原始数据和已安装环境继续保留；它们不能当作普通缓存删除。
`__pycache__/` 等可重新生成的缓存不属于公开项目内容。
