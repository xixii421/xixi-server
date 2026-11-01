# xixi-server 项目上下文

本文件是 Codex 在 VSCode、桌面端和 CLI 中工作的项目入口。开始修改前检查当前分支和工作区状态；下列状态记录到 2026-11-01，代码与最新验证结果优先。

## 项目与工作位置

- 本项目从 Azure/ModServe 派生，新增 PyTorch / Transformers 的 SmolVLM 三阶段后端；不要把上游论文成果或大模型 vLLM 路径当成本后端的实测成果。
- 项目和发行包统一命名 `xixi-server`，Python 导入路径保留 `lmm_serving`；工作目录以当前 checkout 为准。
- Git 远端：`https://github.com/xixii421/xixi-server.git`；整理时的开发分支为 `codex/pd-local`，操作前仍须检查实际分支。
- 最新已验证的功能提交为 `567ecf8`（保持视觉缓存 encoder batch 形状）；继续工作时用 Git 确认后续提交，不要重置用户修改。

## 目标与用户约定

- 目标是可运行、可解释、可复现实测的 AI Infra 项目，最终为简历提供真实证据。
- 用中文交流，通信和调度代码的关键所有权、时序与设计取舍加中文注释。
- 逻辑简洁，避免重复验证和不必要的防御性分支；保留输入、通信接收、预算和生命周期边界检查。
- 教学、租卡平台操作指导在对话里讲，不新增租卡教程或学习笔记污染仓库。必要的 README 和架构说明随代码更新。
- 不编造收益，不把跳过测试算作通过，不把单卡回环称为真实多卡验收。
- `docs/resume_full.md` 是本机私人简历，可能包含联系方式，保持本地，不加入提交或推送。
- 个人项目简历、面试稿和学习笔记已移至忽略目录 `.local/notes/`；不要重新加入公开文档。
- 公开文档入口为 `docs/README.md`；上游说明在 `docs/upstream/`，早期文本 PD 在 `docs/legacy/`，实现修正在 `docs/validation/implementation-notes.md`。
- `metrics/` 原始证据和 `.venv*/` 已安装环境保留在本地，不作为目录清理的删除对象。

## 当前架构

- V = 视觉 Transformer + Connector；P = 完整语言 Prefill + 首 Token；D = 恢复 prompt KV 后逐 Token Decode。
- 设备池与阶段数解耦，每个配置设备对应一个 worker。默认四卡布局为 1V + 1P + 2D，可配置其他非空 V/P/D 池。
- Gloo 传元数据和 ACK；张量通道可选 CPU/Gloo 或 CUDA/NCCL。NCCL 每个 worker 必须独占一张不同的可见 GPU。
- `vpd_router.py` 的 StageRouter 从空闲副本中轮询选择，原子预留整条路由。V/P 在交接 ACK 后释放槽位；D 在生成完成后释放。
- 路由提前预留 D，活动路由数最多为 D 副本数；1V1P2D 支持 A 在 D2 解码时，B 经 V/P 后进入 D3；单 D 配置仍只有一条活动路由。
- 默认 `--max-inflight 4`，包含排队、执行和尚未返回的消费者；超额 429，阶段不足则在主进程 FIFO 等待。
- 事件泵是 worker 事件队列的唯一消费者，核对请求 ID 与路由 rank，再投递各请求收件箱；各请求独立重排 Token 和执行 SSE 回调。
- worker 内顺序执行；没有连续批处理、同 worker 的计算通信 stream 重叠、TP 或多机能力。
- 预处理明确拒绝和未获配路由的排队失败可局部回收。活动请求异常采用整组停止，唤醒在途请求并回收全部 worker，不自动重试半次交接。
- 每个 V 有独立 CPU 字节级视觉特征 LRU。缓存按请求图数隔离，部分命中或去重后补齐原 encoder batch，避免 BF16 批量形状变化导致输出漂移；实际 encoder 槽位和补齐成本单独记录。缓存和完整载荷校验可能产生 CPU 搬运，不能统称零拷贝。

## 阅读与验证入口

- 修改架构或调度前读 `docs/architecture.md`；启动与参数见 `README.md`。
- 当前四卡证据：`docs/validation/four-gpu-20261101.json` 和同名 Markdown；历史并发摘要：`docs/validation/parallel-routing-20261031.json`。多卡验收入口：`utils/validate_vpd_multigpu.py`。
- 相关模块：`vpd_local.py`（阶段与编排）、`vpd_router.py`（槽位）、`vpd_transport.py`（载荷/KV/ACK）、`vpd_service.py`（HTTP 适配）。共享 PD 基础设施改动需验证旧 PD 路径。
- WSL 已验证环境为 `.venv-pd-cuda/bin/python`，Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.51.3。先检查环境是否仍存在，不擅自升级依赖。
- 真实模型路径由使用者配置，公开命令使用 `/path/to/SmolVLM-256M-Instruct`。此前本机 GPU 为 RTX 4060 Laptop 8GB，每次运行仍须重新探测设备；不能在单卡完成四卡 NCCL 验收。
- 功能版本的本机验证通过 106 项单元/集成测试、41 项启动器检查，3 项真实多卡用例跳过。云端 4×RTX4090 拓扑 A/B 分别完整执行并通过 16 项设备池测试，Gloo/NCCL 探针及真实模型 HTTP/SSE 对齐通过。
- 2026-11-01 公开整理后使用 `.venv-pd/bin/python`（PyTorch 2.5.1+cpu）通过 105 项单元/集成测试、41 项启动器检查，4 项 GPU 用例跳过；当前会话无可用 CUDA。首轮因沙箱回环通信限制失败，允许 localhost 后重跑通过，原始日志均保留。公开摘要见 `docs/validation/public-cleanup-20261101.json`。
- 拓扑 A 保留两组并发 1 正式基准和 180 条并发冒烟；拓扑 B 已完成同一物理布局上的 NCCL 并发 1/2/4、Gloo 并发 2 四组正式基准。修复版本共 3420 条测量，其中 2280 条 VPD 请求逐 Token 对齐，HTTP 失败和 Token 不一致均为 0。吞吐比较只在拓扑 B 内进行。
- 原云端拓扑为 V0/P1/D2 在 NUMA0、D3 在 NUMA1；重启后变为 V0/P1 在 NUMA0、D2/D3 在 NUMA1，且 `/dev/shm` 一度从 10GiB 变为 64MiB，NCCL 初始化失败。平台恢复 10GiB 后已通过完整补测；容器本身不能重挂载。后续开关机仍须核对共享内存及 UUID/PCI 拓扑。CUDA peer access 均为 false，调试通道确认为 SHM，不称为直接 P2P 或 NVLink 验收。
- 按改动选择必要套件，已有运行器示例：

```bash
PYTHONPATH=src .venv-pd-cuda/bin/python utils/run_regressions.py \
  --suites vpd_router vpd_parallel vpd_transport vpd vpd_pools vpd_tools vision_cache pd_local pd_service stream_benchmark launcher \
  --timeout 180 --output-dir metrics/<新的运行目录>
```

- 并发基准为 `utils/benchmark_vpd.py --concurrency 2 --max-inflight 4`，结合 README 提供模型、设备池和新输出目录；默认客户端并发仍为 1。
- 原始日志放在新的 `metrics/` 子目录，不覆盖历史证据。摘要区分实际执行、跳过和硬件范围。
- 历史 TTFT 降低 48.5% 只属于旧单卡、并发 1、特定缓存命中负载，不能移用于新四卡版本。当前公开指标采用拓扑 B：同四卡 NCCL 冷图无缓存、客户端并发 1→2，吞吐 23.9→44.9 Token/s（+87.5%）；并发 1、命中率 90% 的混合复用合成负载，TTFT P50 144.2→126.4ms（降低 12.3%），该并发冷图约增加 0.7%。条件与边界见 `docs/validation/four-gpu-20261101.md`，不称为单卡到四卡的等预算扩容收益。
- 架构、默认参数或验证状态改变后同步维护本文件相关条目，避免陈旧记忆覆盖新事实。
