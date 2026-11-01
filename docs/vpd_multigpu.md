# 单机多卡 V/P/D 设备池与验收

当前代码按 V/P/D 三个非空设备池启动副本，每个设备对应一个 spawn 工作进程。
旧三卡配置仍兼容；四卡推荐 1V+1P+2D，也可配置更多副本和不同设备编号。
本机只有 RTX 4060 Laptop 一张卡；2026-11-01 已在租用的 4×RTX4090 拓扑 A 上完成四卡 NCCL 交接验收，
另在固定拓扑 B 上完成 NCCL 并发 1/2/4、Gloo 并发 2 正式基准，见 [四卡记录](validation/four-gpu-20261101.md)。
修复版本共 3420 条测量请求、2280 条 VPD 请求逐 Token 对齐。CUDA peer access 均为 false，
调试通道为 SHM，不能把 NCCL 交接称为直接 GPU P2P。
本机通过的测试与待多卡验收的项目必须分别记录，不能把跳过用例算作通过。

## 部署与依赖

目标是 Linux/WSL 单机、每个副本独占一个可见 CUDA 设备，模型仍限定 Idefics3/SmolVLM。
建议沿用已验证的 Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.51.3：

```bash
python3.10 -m venv .venv-vpd
.venv-vpd/bin/python -m pip install -r requirements-vpd-cuda.txt
```

容器还须检查共享内存：本轮 10GiB `/dev/shm` 下通过，平台重启后 64MiB 下 NCCL 创建共享段失败。
开关机后重新核对共享内存及物理 GPU/NUMA 拓扑，不能把不同拓扑的并发结果合并计算加速比。

GPU 数量指同一台服务器的可见卡数。代码支持阶段副本池和有界并发路由，暂不支持多机、TP、连续批处理和动态扩缩容。
P/D 分别持有语言模型权重，V 只把视觉模型和 Connector 放到自己的 GPU。
每个工作进程先在 CPU 加载完整模型，再裁剪阶段模块并放到指定设备。

## 四卡设备池

本项目四卡默认推荐 1V+1P+2D，配置直接决定工作进程数量和阶段分工：

```bash
PYTHONPATH=src python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct --transport nccl \
  --vision-devices cuda:0 --prefill-devices cuda:1 --decode-devices cuda:2 cuda:3
```

使用同一套配置验收，脚本名称保持兼容：

```bash
PYTHONPATH=src python utils/validate_vpd_multigpu.py \
  --model /path/to/SmolVLM-256M-Instruct \
  --vision-devices cuda:0 --prefill-devices cuda:1 --decode-devices cuda:2 cuda:3 \
  --output-dir metrics/four-gpu-check-001
```

也支持 2V+1P+1D、1V+2P+1D，以及更多副本：三个池都必须非空，NCCL 下设备不能重复。
这三组新参数需一起使用，且不能与旧 `--stage-devices` 混用。
每请求在 V/P/D 池中各轮询一个空闲副本，原子预留整条路由；纯文本跳过 V。`route` 报告实际 global rank。
V/P 在交接完成后释放，D 在生成结束后释放，支持不同请求的阶段重叠和多 D 并行。
默认 `--max-inflight 4` 限制已接纳请求；资源不足时在主进程 FIFO 等待，超额返回 429。
D 提前预留，活动路由数最多等于 D 副本数；单 D 配置仍只有一个活动路由。
每个 V 拥有独立 CPU LRU，预算按每个 V 计算。
多卡验收探针遍历所有 V×P、P×D 候选边，模型验收检查所有副本均参与。
以下保留旧三卡的单副本示例和历史验证记录。

```bash
PYTHONPATH=src .venv-vpd/bin/python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct \
  --transport nccl --stage-devices cuda:0 cuda:1 cuda:2 \
  --vision-cache-mib 0 --port 8099
```

启动器按设备池展开工作进程，直接运行即可，无需额外套 `torchrun`。
默认 NCCL 映射也是 `cuda:0 cuda:1 cuda:2`，允许调换顺序。
`CUDA_VISIBLE_DEVICES` 生效后参数使用逻辑编号。例如物理卡 2/4/6：

```bash
CUDA_VISIBLE_DEVICES=2,4,6 PYTHONPATH=src .venv-vpd/bin/python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct --transport nccl \
  --stage-devices cuda:0 cuda:1 cuda:2
```

`--device` 保留给共享设备 Gloo 模式以及工具的单模型参考设备；工作进程以
`--stage-devices` 为准。NCCL 拒绝 CPU、重复占卡及不可见设备，不自动降级 Gloo。
启动信息和 backend_metrics 返回实际 worker 设备、transport、校验模式。
HTTP/SSE 请求方法、图片和生成预算与 [本机服务](vpd_local.md) 一致。

## 数据通道与生命周期

```text
rank 0 / GPU V: Vision + Connector
      |  Gloo: 有界 JSON 元数据；NCCL: 连续视觉特征；Gloo: ACK
rank 1 / GPU P: 多模态 Prefill + 首 Token
      |  Gloo: 有界 JSON 元数据；NCCL: 打包 KV；Gloo: ACK
rank 2 / GPU D: 恢复 DynamicCache，逐 Token Decode
```

- 默认进程组是 Gloo，专门处理 CPU 元数据和确认消息。
- 所有配置中的 rank 按相同顺序创建 NCCL 数据组，执行一次全员 all-reduce 预热再进入 P2P。
- 每个进程在加载阶段权重前执行 `torch.cuda.set_device`，一进程使用一张卡。
- KV 按 K0/V0/K1/V1 顺序打包成一个连续张量；接收形状由本地模型和请求计算。
- 请求 UUID、序号、模型签名、prompt/context/image 顺序、dtype 和内存预算均继续校验。
- 每次通信等待 Work 完成并同步该设备；接收端完成搬运和验证后再回 ACK。
  发送端保留源 KV、打包缓冲区或视觉特征直至 ACK，随后才能释放/复用。
- 异常、超时或流式消费者断开后关闭整个运行时，清理所有工作进程。需要重启服务后继续。

该实现支持有界的跨请求阶段重叠，每个 worker 内顺序执行，不声称同 worker 的计算通信 stream 重叠。
NCCL 内部选择 PCIe/P2P/SHM 等路径；仅凭使用 NCCL 不能宣称 NVLink、RDMA 或零拷贝。
实现依据 [PyTorch 2.6 分布式文档](https://docs.pytorch.org/docs/2.6/distributed.html)
中的设备独占、异步通信缓冲区生命周期及 CUDA Work 完成语义。

## 校验与 CPU 视觉缓存

默认开启完整载荷 SHA256 校验。CUDA 载荷为计算哈希会产生 CPU 镜像，适合正确性验收。
性能实验可加 `--no-handoff-checksum`：仅关闭载荷哈希，仍验证元数据、顺序、形状和 ACK。
指标中 `verify_payload=false` 会明确标注这个条件，验收输出应先通过开启校验的运行。

视觉 LRU 仍为 CPU 缓存：启用后，视觉特征存入 CPU，命中后重新上传到 V 的 GPU 再发 NCCL。
缓存按请求图数隔离条目；部分命中和重复图片去重时，以重复的未命中图片补齐 encoder batch，
保持与完整模型相同的数值执行形状。`vision_encoder_batch_images` 和 `vision_padding_images`
记录实际计算槽位与补齐成本，`vision_images_encoded` 只表示不同缺失图片数。
因此，只有 **关闭视觉缓存且关闭完整载荷校验** 时，视觉特征交接才没有应用层整张量 CPU 中转；
KV 在 NCCL 模式始终以 CUDA 张量传输，开启哈希时仍有校验用 CPU 镜像。
图像预处理、控制信息本身在 CPU 上。不能把上述条件省略后声称整个服务“无 CPU 拷贝”。

## 三卡一键验收

先上传代码和已有模型，再运行：

```bash
PYTHONPATH=src .venv-vpd/bin/python utils/validate_vpd_multigpu.py \
  --model /path/to/SmolVLM-256M-Instruct \
  --stage-devices cuda:0 cuda:1 cuda:2 \
  --output-dir metrics/three-gpu-acceptance-001
```

输出目录必须不存在。验收顺序：

1. 检查三张不同的 CUDA 卡、Gloo/NCCL 和模型类型；不足三卡直接失败并输出原因。
2. 保存 GPU/驱动、CUDA/NCCL 版本、CUDA_VISIBLE_DEVICES、GPU 拓扑和 peer-access 矩阵，
   以及源码、模型文件 SHA256 和依赖版本。
3. 启用真实三卡用例：缓存/非缓存路径、开启/关闭校验、文本旁路、多图换序、重复图片、
   单 Token 生成及工作进程故障回收。没有足够硬件时这些用例报错，不静默跳过。
4. 分别运行三卡 Gloo/NCCL 传输探针：72/144KiB 视觉量级和 1/8MiB KV 量级；
   float32/bfloat16，两条边逐元素验证，多次采样延迟。
5. 分别运行三卡 Gloo/NCCL 预训练模型验收：六类输入对齐单模型生成结果，检查 HTTP 非流式和 SSE。

`report.json` 汇总是否通过；每步保留日志和独立 JSON。当前代码没有替你租卡、付费或连接服务器。

探针测量的是包含发送端 staging、数据传输、接收端 staging/逐元素检查和 ACK 的完整交接延迟，
不是原始 PCIe/NVLink 带宽。peer-access=false 不直接判定失败，NCCL 可能选择其他可用路径。

## 性能实验

在验收命令加 `--benchmark`，会另外为 Gloo 和 NCCL 分别运行既有 HTTP 消融：
单模型、VPD 关闭缓存、VPD 开启缓存；热复用/冷图片/混合复用；默认三轮、每组 20 个请求。
保留输出 token 一致性检查，关闭整张量 SHA256 以排除校验镜像开销。

```bash
PYTHONPATH=src .venv-vpd/bin/python utils/validate_vpd_multigpu.py \
  --model /path/to/SmolVLM-256M-Instruct --benchmark \
  --requests 20 --rounds 3 --output-dir metrics/three-gpu-benchmark-001
```

验证四卡 1V+1P+2D 的并发时，将上述设备参数换为三组设备池，并显式增加
`--concurrency 2 --max-inflight 4`。不加 `--concurrency` 时客户端仍按并发 1 测量。
可分别用并发 1/2/4 测量排队与吞吐，保持布局、负载、输出长度和缓存开关一致。
`queue_wait_ms` 记录后端路由等待，`decode_started_s/decode_finished_s` 可检查同机解码区间是否相交；
时间区间重叠只证明调度并发，吞吐收益按整组请求墙钟时间另行计算。
单体基线使用相同客户端并发与接纳上限，模型执行仍串行。

单模型只占一张 GPU，三阶段占三张，报告会注明资源差别。它们的吞吐比较不能直接解释为
等 GPU 预算下的扩容收益。Gloo/NCCL 对比使用相同的阶段映射；两个大块实验尚未交错随机化，
有温度/频率漂移风险，不能把微小差异直接视为确定收益。

首 Token 在 Prefill 完成时即可发送，因此 KV 通信的影响还需要看首/第二 Token 间隔、后续 TBT
和 E2E，不能只看 TTFT。小模型三卡可能更慢，实测后据实记录。

## 本机回归与简历边界

```bash
PYTHONPATH=src .venv-pd-cuda/bin/python utils/run_regressions.py \
  --suites vpd_router vpd_parallel vpd_transport vpd vpd_pools vpd_tools vision_cache pd_local pd_service stream_benchmark launcher \
  --timeout 180 --output-dir metrics/local-regression-new
```

本机单 GPU 可以验证 NCCL 组初始化、CUDA 张量准备、校验和同步 API，不能验证跨卡 P2P。
三卡用例默认跳过，报告区分发现数量、执行数量和跳过数量。
真实三卡验收以前，可表述“实现三阶段设备映射、NCCL 张量交接和三卡验收工具”；
不能写“已完成三卡验证”或编写未经实测的多卡性能数字。

上一轮三卡版本的本机验证记录：67 项单元/集成测试与 41 项启动器脚本检查通过；
其中新增通信套件实际执行 12 项，包含单卡真实 NCCL API 测试，另有 2 项三卡测试跳过。
预训练 SmolVLM 的六类输入、HTTP/SSE 在开启/关闭载荷校验的路径中均通过；
CPU 与单卡 Gloo 传输探针通过，基准脚本冒烟的 18 条测量请求输出对齐。
三卡验收入口在单卡硬件上按预期拒绝启动，没有把它计为三卡验收成功。
见 [上一轮验证汇总](validation/three-gpu-implementation-20261031.json)。
冒烟运行只有一轮、每场景两请求，用于验证脚本，不能替换此前正式简历性能实验。

本轮架构整理与修复另见 [代码复查](code_review.md)，新验证记录保存在
`metrics/vpd-architecture-review-20261031/`。配置/在线签名采用启动快照，
HTTP 只准备一次输入；探针在销毁通信组并确认正常退出后才报告通过。
公开测试摘要见 [本轮验证汇总](validation/architecture-review-20261031.json)，完整日志保存在运行机器的 metrics 目录。
