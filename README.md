# xixi-server

**面向 SmolVLM 的视觉编码 / Prefill / Decode 三阶段推理服务。**

在 [ModServe](https://github.com/Azure/ModServe) 派生仓库中新增 PyTorch / Transformers 后端，将真实多模态模型拆为三个阶段，按设备池启动各阶段副本，交接视觉特征与 KV Cache，提供 HTTP/SSE 接口、CPU 视觉特征 LRU，以及单机多卡 NCCL 传输实现。

当前主线支持按空闲阶段槽位分配路由、有界并发和背压，阶段设备可独立配置。单卡真实模型与协议路径已经验证，**多卡点对点通信和性能待真实硬件验收**。

## 架构

```text
HTTP / SSE ── 主进程：请求入口、调度、Token 输出
                    │ multiprocessing.Queue：命令与事件
             ┌──────┴───────────┬─────────────────┐
             ▼                  ▼                 ▼
          V / rank 0 ───────► P / rank 1 ───────► D / rank 2
          视觉编码与投影       多模态 Prefill        逐 Token Decode
          CPU 特征 LRU         产生首 Token         恢复 KV 并续算
                 视觉特征                打包 KV
```

阶段间的元数据和 ACK 走默认 Gloo 组；张量可选 CPU/Gloo 或 CUDA/NCCL。每个阶段可配置多个设备，每个设备运行一个副本，发送端持有数据直到接收端确认接管。职责、时序和设计取舍见 [架构设计](docs/architecture.md)。

## 安装与启动

以下命令在仓库根目录执行，目标环境为 Linux/WSL、Python 3.10。准备本地 [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) 目录，包含权重、模型配置和 processor/tokenizer 文件。

容器内启动 NCCL 前检查 `/dev/shm`。本轮四 worker/NCCL 2.21.5 在 10GiB 共享内存下通过，
平台重启后恢复为 64MiB 时初始化失败；需先恢复足够容量。GPU 逻辑编号也可能在重启后对应不同物理卡，
性能对照须记录并固定 UUID/PCI 和 NUMA 布局。

```bash
python3.10 -m venv .venv-vpd
.venv-vpd/bin/python -m pip install -r requirements-vpd-cuda.txt
```

单卡运行三个进程，使用 CPU 中转的 Gloo 张量通道：

```bash
PYTHONPATH=src .venv-vpd/bin/python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct \
  --device cuda:0 --transport gloo --port 8099
```

三卡部署：

```bash
PYTHONPATH=src .venv-vpd/bin/python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct \
  --transport nccl --stage-devices cuda:0 cuda:1 cuda:2 --port 8099
```

上述配置启动三个工作进程，无需额外使用 `torchrun`。设备编号以 `CUDA_VISIBLE_DEVICES` 映射后的逻辑编号为准。启动日志返回服务地址、模型名、阶段设备和传输模式。

四卡推荐配置为 **1V + 1P + 2D**。阶段数仍为三个，工作进程数由设备池展开生成：

```bash
PYTHONPATH=src .venv-vpd/bin/python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct --transport nccl \
  --vision-devices cuda:0 --prefill-devices cuda:1 --decode-devices cuda:2 cuda:3
```

三个设备池都须非空，不能与旧 `--stage-devices V P D` 混用。设备可任意重新编号；
NCCL 下每个副本独占一张可见 GPU。请求从各池轮询选择空闲副本，固定整条路由；
V/P 在交接确认后释放槽位，D 在生成完成后释放，因此 A 在 D2 解码时，B 可经过 V/P 交给 D3。
每次路由提前预留一个 D，活动路由数最多为 D 副本数；其余已接纳请求在主进程 FIFO 等待。
单 worker 顺序执行，不提供连续批处理或同 worker 的计算通信重叠，吞吐变化需在目标硬件实测。
`backend_metrics.route`、`stage_pools` 和 `worker_info` 返回实际分配。

没有 GPU 时，可按 [CPU 依赖](requirements-pd-local.txt) 创建独立环境，用 `--device cpu` 运行；省略 `--model` 会使用随机初始化的微型测试模型，只用于功能验证。

## 接口与常用选项

服务监听 `127.0.0.1:8099`：`GET /health` 查看状态，`POST /v1/chat/completions` 进行生成。支持文本、内联 PNG/JPEG、多图与 SSE。`model` 字段应与启动日志一致，下面假设模型目录名是 `SmolVLM-256M-Instruct`：

```bash
curl -N http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolVLM-256M-Instruct","messages":[{"role":"user","content":"Explain KV cache briefly."}],"max_tokens":8,"stream":true}'
```

图片请求示例和完整限制见 [使用说明](docs/vpd_local.md)。当前支持 greedy、`n=1`、固定长度生成，最多 32 个输出 Token；最多两张图。默认最多接纳 4 个请求，计入排队、执行及尚未返回的流式消费者，超出容量返回 429。事件泵按请求 ID 投递到独立收件箱，各 SSE 流分别按 Token index 输出。

| 选项 | 用途 |
|---|---|
| `--vision-cache-mib 16` | CPU 视觉特征 LRU 预算，默认 16MiB；设为 `0` 关闭 |
| `--transport gloo` | CPU 张量交接，支持单卡或指定的阶段设备 |
| `--transport nccl` | CUDA 张量交接，至少三张可见 GPU，每个 worker 独占一张 |
| `--stage-devices V P D` | 分别指定视觉、Prefill、Decode 设备 |
| `--vision-devices … --prefill-devices … --decode-devices …` | 为各阶段配置非空设备池，进程数由配置决定 |
| `--max-inflight 4` | 已接纳请求总上限；空闲阶段不足时在主进程排队 |
| `--no-handoff-checksum` | 关闭完整载荷 SHA256，保留元数据和 ACK 检查 |

完整载荷校验会把 CUDA 数据复制到 CPU 做哈希；视觉 LRU 也存于 CPU。性能实验需记录这两个开关，不能将默认路径描述成“全程零拷贝”。

## 验证与实验

本机回归：

```bash
PYTHONPATH=src .venv-vpd/bin/python utils/run_regressions.py \
  --suites vpd_router vpd_parallel vpd_transport vpd vpd_pools vpd_tools vision_cache pd_local pd_service stream_benchmark launcher \
  --timeout 180 --output-dir metrics/local-check-001
```

真实三卡机器的一键验收：

```bash
PYTHONPATH=src .venv-vpd/bin/python utils/validate_vpd_multigpu.py \
  --model /path/to/SmolVLM-256M-Instruct \
  --stage-devices cuda:0 cuda:1 cuda:2 \
  --output-dir metrics/three-gpu-check-001
```

验收记录硬件拓扑、源码/模型摘要、传输探针、模型输出一致性及 HTTP/SSE 结果；输出目录须为新目录。加 `--benchmark` 后执行单体 / 三阶段无缓存 / 三阶段有缓存的 HTTP 对照，覆盖热图、冷图与混合复用负载。

四卡或其它副本分配使用同一个验收脚本，把 `--stage-devices` 换成上面的三组设备池参数。
传输探针覆盖所有 V×P 和 P×D 候选连接，模型验收要求每个副本实际处理请求。
本机设备池回归和四进程单卡回环结果见 [验证摘要](docs/validation/device-pools-20261031.json)，不代表真实四卡验收。
并发压测为 `benchmark_vpd.py` 增加 `--concurrency 2 --max-inflight 4`，或在一键验收时同时加 `--benchmark --concurrency 2`。并发度默认仍为 1，须显式设置；吞吐按整组请求的墙钟时间计算。
并发路由的 [验证摘要](docs/validation/parallel-routing-20261031.json) 记录 103 项单元/集成测试、41 项启动器检查及真实 SmolVLM 并发冒烟；3 项多卡用例因硬件不足跳过。

2026-11-01 的 [四卡报告](docs/validation/four-gpu-20261101.md) 和 [摘要](docs/validation/four-gpu-20261101.json)
记录缓存 batch 形状修复、本机 106 项单元/集成测试及 41 项启动器检查，以及真实 4×RTX4090 验收。
修复版本共 3420 条测量请求，其中 2280 条 VPD 请求逐 Token 对齐，HTTP 失败与 Token 不一致均为 0。
固定拓扑 B 已完成 NCCL 并发 1/2/4、Gloo 并发 2 正式对照；同四卡、冷图无缓存时，
客户端并发 1→2 的有效吞吐为 23.9→44.9 Token/s（+87.5%）。NCCL 并发 1、命中率 90% 的混合复用
合成负载下，TTFT P50 为 144.2→126.4ms（降低 12.3%）；该并发的冷图约增加 0.7%。
这些是条件明确的合成负载结果，不是单卡到四卡的成本/扩容收益，也不能套用历史单卡 48.5% 数字。

- [多卡设备池、验收和测量方法](docs/vpd_multigpu.md)
- [历史单卡消融实验](docs/vpd_benchmark.md)：保留当时源码和原始记录，不能作为当前三卡结果。
- [项目简历文案](docs/resume_project2.md)
- [实测指标填写表](docs/resume_measurements.md)
- [本轮代码复查](docs/code_review.md)
- [代码阅读与面试准备](docs/vpd_interview.md)

单卡测试中的 NCCL 组初始化不代表跨卡通信已经通过。三卡用例默认跳过，显式三卡验收要求足够硬件；不以跳过用例充当通过。单体占一张 GPU、三阶段占三张 GPU 时，也不能直接推导等 GPU 预算下的扩容收益。

## 代码导航

| 文件 / 目录 | 职责 |
|---|---|
| `src/lmm_serving/vpd_service.py` | 服务配置与 HTTP 适配 |
| `src/lmm_serving/vpd_local.py` | 模型阶段、输入准备、V/P/D 请求编排 |
| `src/lmm_serving/vpd_router.py` | 空闲副本选择、路由预留与阶段槽位释放 |
| `src/lmm_serving/vpd_transport.py` | 张量通道、KV 打包、接收恢复与 ACK |
| `src/lmm_serving/vision_cache.py` | 由 V 进程独占的 CPU 字节级 LRU |
| `src/lmm_serving/pd_local.py`、`pd_service.py` | 共享进程/HTTP 基础设施及早期 Qwen3 文本 PD 参考路径 |
| `src/lmm_serving/monolithic_local.py` | 持久单体模型对照 |
| `utils/validate_vpd_multigpu.py`、`probe_vpd_transport.py` | 多卡设备池验收与全候选边传输探针 |
| `utils/benchmark_vpd.py`、`src/lmm_serving/stream_benchmark.py` | HTTP 负载与流式测量 |

## 与上游的关系

本仓库保留 [ModServe 原实现](https://github.com/Azure/ModServe)、模型适配、vLLM 补丁和提交历史。当前 SmolVLM V/P/D 后端是派生工程，未接入原版 vLLM 的批处理调度。早期 Qwen3 文本 PD 后端继续保留，见 [文本 PD 说明](docs/pd_local.md)。

原版服务的部署与配置见 [配置说明](docs/configuration.md)、[拓扑说明](docs/topology.md)、[原版可靠性扩展](docs/modserve_reliability.md) 和 [实验指南](docs/experiments.md)。原版大模型路径与本后端的验证范围分别记录，论文结果属于原作者。

- [ModServe 论文](https://dl.acm.org/doi/10.1145/3698038.3698540)
- [上游请求轨迹](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md)
