# xixi-server 项目上下文

本文件是 Codex 在 VSCode、桌面端和 CLI 中工作的项目入口。开始修改前检查当前分支和工作区状态；下列状态是 2026-10-31 的记录，代码与最新验证结果优先。

## 项目与工作位置

- 本项目从 Azure/ModServe 派生，新增 PyTorch / Transformers 的 SmolVLM 三阶段后端；不要把上游论文成果或大模型 vLLM 路径当成本后端的实测成果。
- 本机实际开发、测试仓库位于 WSL Ubuntu-22.04：`/home/xixii/project/ModServe-SingleGPU`。
- Git 远端：`https://github.com/xixii421/xixi-server.git`；当前开发分支：`codex/pd-local`。
- Windows 的 `D:\重要文档\AI Infra\tmp\xixi-server` 是文件镜像，Git 状态可能落后；避免把它误当作实际 WSL 工作仓库。其他电脑以当前实际 checkout 为准。
- 最新已验证的功能提交为 `958be6e`（有界并发路由）；继续工作时用 Git 确认后续提交，不要重置用户修改。

## 目标与用户约定

- 目标是可运行、可解释、可复现实测的 AI Infra 项目，最终为简历提供真实证据。
- 用中文交流，通信和调度代码的关键所有权、时序与设计取舍加中文注释。
- 逻辑简洁，避免重复验证和不必要的防御性分支；保留输入、通信接收、预算和生命周期边界检查。
- 教学、租卡平台操作指导在对话里讲，不新增租卡教程或学习笔记污染仓库。必要的 README 和架构说明随代码更新。
- 不编造收益，不把跳过测试算作通过，不把单卡回环称为真实多卡验收。
- `docs/resume_full.md` 是本机私人简历，可能包含联系方式，保持本地，不加入提交或推送。

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
- 每个 V 有独立 CPU 字节级视觉特征 LRU。缓存和完整载荷校验可能产生 CPU 搬运，不能统称零拷贝。

## 阅读与验证入口

- 修改架构或调度前读 `docs/architecture.md`；启动与参数见 `README.md`。
- 当前验证摘要：`docs/validation/parallel-routing-20261031.json`；多卡验收入口：`utils/validate_vpd_multigpu.py`。
- 相关模块：`vpd_local.py`（阶段与编排）、`vpd_router.py`（槽位）、`vpd_transport.py`（载荷/KV/ACK）、`vpd_service.py`（HTTP 适配）。共享 PD 基础设施改动需验证旧 PD 路径。
- WSL 已验证环境为 `.venv-pd-cuda/bin/python`，Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.51.3。先检查环境是否仍存在，不擅自升级依赖。
- 真实模型本机目录：`/home/xixii/huggingface/SmolVLM-256M-Instruct`。本机只有 RTX 4060 Laptop 8GB，不能在此完成四卡 NCCL P2P 验收。
- 本轮通过 103 项单元/集成测试、41 项启动器检查；3 项真实多卡用例跳过。真实模型并发 2 的 18 条测量请求输出对齐，属于单卡功能冒烟，不能充当多卡性能收益。
- 按改动选择必要套件，已有运行器示例：

```bash
PYTHONPATH=src .venv-pd-cuda/bin/python utils/run_regressions.py \
  --suites vpd_router vpd_parallel vpd_transport vpd vpd_pools vpd_tools vision_cache pd_local pd_service stream_benchmark launcher \
  --timeout 180 --output-dir metrics/<新的运行目录>
```

- 并发基准为 `utils/benchmark_vpd.py --concurrency 2 --max-inflight 4`，结合 README 提供模型、设备池和新输出目录；默认客户端并发仍为 1。
- 原始日志放在新的 `metrics/` 子目录，不覆盖历史证据。摘要区分实际执行、跳过和硬件范围。
- 历史 TTFT 降低 48.5% 只属于旧单卡、并发 1、特定缓存命中负载，不能移用于新四卡版本。简历文案见 `docs/resume_project2.md`，指标待真实硬件填写。
- 架构、默认参数或验证状态改变后同步维护本文件相关条目，避免陈旧记忆覆盖新事实。
