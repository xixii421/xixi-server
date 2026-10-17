# 原版 ModServe：可靠传输与资源背压

本轮修改直接进入原版 `modserve` 请求链路、Image 节点、接收进程、Text 代理及 vLLM 补丁。没有新增调度评分算法。CPU 学习模式仍独立存在。

## 问题与改动

| 原实现的触发条件 | 风险 | 当前处理 |
|---|---|---|
| 多个请求使用同一图片，或文件名第一个点之前相同 | 按图片名缓存可能覆盖、错取 | 每次请求生成内部 request ID，每张图使用 `request_id:index`；文件名仅用于读图片 |
| 不同 API 前端各自触发 send 和 recv | 两端队列顺序可能不一致 | 接收服务统一安排每对 rank 的 FIFO、epoch 和序号，再触发发送 |
| 同一 GPU 上不同 pair communicator 交叉执行 | 多通信组执行顺序难以保证 | 先取得 Image 通信许可，再取得 Text 通信许可；同一传输进程的 GPU 同时只执行一个通信组 |
| HTTP 超时或取消后 GPU 操作仍在执行 | 过早释放缓冲区，或错误复用通道 | 区分排队取消、设备操作中、IPC 消费中；不确定操作保留资源并隔离节点 |
| 编码后才检查 Text 的 token 容量 | 上游已产生大量待发送特征 | 在编码前预留 Text token/字节/请求数，以及 Image 特征字节/数量 |
| 超大请求按预算上限计账 | 实际使用量突破限制 | 超过单节点上限返回 413，暂时无空余容量返回 429 |

## 正常路径

```text
Cluster 选择 Text 节点（沿用配置的 round_robin / least_loaded）
  → 检查接收服务健康
  → Text /reserve_request：预留图像 token、2×特征字节、请求槽位
  → 为每张图选择 Image 节点并 /reserve_feature
  → 全部预留成功后，使用有界线程池启动编码
  → Image 根据实际 tensor 大小校正本地字节预算并发布
  → Cluster /resize_request：按实际 token 和字节数校正 Text 预算
  → Text /generate 核对实际请求与预留额度，然后提交 vLLM
  → vLLM 仅调用本机接收服务，不直接独立触发源端
  → 接收服务准备源端许可，再取得本地许可
  → 源端确认入队 → NCCL 身份头 → NCCL BF16 特征 → 双端确认完成
  → 接收服务发放一次性 IPC lease
  → vLLM 映射、clone、同步完成后，用 lease_token 释放源缓冲区
  → 生成完成后 Text 代理归还预算；Cluster 做幂等取消清理
```

身份头是 `spec + channel_epoch + sequence` 的 SHA-256，作为独立的 32 字节张量通过同一通信组发送。它验证传输归属和元数据，不是整块特征内容校验和，也不是 NCCL tag。

源端接受严格递增的通道序号，并记住首个 epoch。新的接收进程不能用新 epoch 接管旧通道，必须整组重启。源端重复控制消息复用原任务结果，不再次发送数据。IPC lease 只允许一个消费者领取；重复领取返回 409，防止两个读者复制时一个提前确认释放。释放确认可以重复调用。

不同 Image→Text pair 可以使用独立队列，但通信许可覆盖同一传输进程中的全部 communicator。所有节点按 Image→Text 的相同方向取得两个许可，避免交叉等待。协议对 NCCL 的约束依据参见 [NVIDIA 多通信组说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)。真实 GPU 上与 vLLM TP 通信并行运行的情况仍需实测。

## 超时、取消与显存生命周期

| 状态 | 超时或取消行为 |
|---|---|
| `reserved` / `ready` | 可立即回收预算和未使用缓冲区 |
| 排队但尚未执行的传输 | 撤销；worker 不再启动迟到的设备操作 |
| `encoding` | 标记取消；编码线程取得锁后先检查取消；若设备工作已开始，确认完成后才能回收 |
| `sending` / `receiving` | 不直接释放；传输有歧义或失败时节点隔离，拒绝新任务 |
| `leased` | TTL 和 Cluster 取消都不能释放；只能凭正确 lease_token 确认已完成复制 |
| Text `running` | Cluster HTTP 超时不能退还它仍在使用的 token/字节预算 |
| Text `quarantined` | 后端超时、连接失败或服务端错误无法证明执行结束，保留额度并拒绝新请求 |

NCCL 异常时，`TorchTransport` 显式保留可能仍被设备使用的张量和身份头。缓存 TTL 只自动清理可安全回收的状态；不用于强制释放正在使用的 GPU 内存。对于失联的 IPC 消费者、卡死的 GPU 操作或结果不明的后端请求，本实现选择隔离并整组重启，没有实现在线 communicator 修复。

Text 多模态调用关闭 SDK 自动重试，避免在后端结果不明时重放已消费的 transfer ID。纯文本仍使用原有代理路径。

跨主机传递剩余 `timeout_s`，各服务转换为本机单调时钟截止时间；不直接比较不同机器的 `perf_counter` / monotonic 时间戳。Cluster 清理使用额外的有界时间窗口。完成、取消的 ID 在请求超时加 60 秒的窗口内保留；历史达到上限时返回 429，不提前驱逐仍有效的取消记录。

## 预算边界和取舍

- Image 预算限制特征 payload，不覆盖模型权重、预处理输入或编码器中间激活。
- Text 字节预算按两份 payload 计账，保守覆盖接收缓冲区与 IPC clone 重叠；不代表对 vLLM 内部所有副本、KV cache 或激活作精确显存建模。
- 字节数和 token 数是独立维度，额度原子预留。实际大小增加时不持有旧额度无限等待，而是立即尝试扩容；失败则取消整个请求。
- Text 节点选择前移到编码前，并按预估负载记账，直到请求结束；牺牲编码后的最新负载信息，换取早期资源预留。调度器评分代码没有改变。
- 容量不足时采用明确拒绝，没有自动换节点、动态优先级或新的负载均衡算法。客户端可以稍后使用新请求重试。
- 同一传输进程的通信串行化、握手和身份头会增加开销；本轮目标是正确性与资源边界，没有声称吞吐或尾延迟一定提升。

## 配置与部署

所有配置在 `src/lmm_serving/config/config.yaml`，可通过环境变量覆盖。

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `MODSERVE_REQUEST_TIMEOUT_S` | 600 | 整个图像请求的时间预算 |
| `NCCL_INIT_TIMEOUT_S` | 1800 | 等待模型加载、全体 rank 加入和初始化的上限 |
| `TENSOR_TRANSFER_TIMEOUT_S` | 60 | pair 通信组和单个传输任务的时间预算 |
| `TRANSFER_QUEUE_SIZE` | 64 | 每对节点的排队上限 |
| `TRANSFER_MAX_FEATURES` | 128 | 单个 Image / 接收服务的特征槽位上限 |
| `TRANSFER_HISTORY_LIMIT` | 65536 | 单服务的活动和去重历史上限 |
| `IMAGE_FEATURE_BYTE_BUDGET` | 1 GiB | 每个 Image 特征缓冲区上限 |
| `RECV_FEATURE_BYTE_BUDGET` | 1 GiB | 每个接收服务缓冲区上限 |
| `TEXT_FEATURE_BYTE_BUDGET` | 2 GiB | 每个 Text 实例的特征预算，按 2×payload 计账 |
| `TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET` | 90000 | 每个 Text 实例的在途图像 token 上限 |
| `TEXT_MAX_IMAGE_REQUESTS` | 128 | 每个 Text 实例接纳的图像请求数 |
| `MODSERVE_MAX_IMAGES` | 128 | 每请求的图像数上限，同时受模型上限约束（Llama 为 8） |
| `MODSERVE_ENCODE_WORKERS` | 8 | 每个请求的最大编码 HTTP 并发数 |
| `TENSOR_TRANSFER_RECV_BASE_PORT` | 5051 | 接收服务起始端口，后续 Text 实例递增 |

协议版本已变更为 v2，必须更新所有节点的项目代码并重建 vLLM 补丁，再整组重启。仅更新 Python 节点但保留旧后端会被拒绝。已有 vLLM 安装时应显式执行重建脚本；`pixi run setup` 的 import 检查可能跳过已有安装：

```bash
# 仅取固定源码并应用补丁，不安装模型或 vLLM：
SKIP_INSTALL=1 bash llm-backend/build_vllm.sh

# 真正更新原版运行环境中的 vLLM：
bash llm-backend/build_vllm.sh
```

内部控制接口沿用原项目受信任集群网络的部署假设，不应直接暴露到公网。

## 观测和验证

```bash
# 源端/接收端健康、排队数、容量使用量、各状态特征数量：
curl -s http://127.0.0.1:8086/transfer_status
curl -s http://127.0.0.1:5051/transfer_status

# Text 预算及 reserved/running/quarantined 请求数：
curl -s http://127.0.0.1:8080/admission_status

PYTHONPATH=src .venv/bin/python tests/test_modserve_reliability.py
```

测试直接复用生产 `Capacity`、`FeatureRegistry`、`TextAdmission`、`TransferService`、`TorchTransport`、`ImagePipeline`。Gloo 测试替换模型编码、生成和 CUDA IPC handle，但真的跨进程传 BF16 张量，检查并发同图、多图、提前拒绝、实际大小校正、部分失败和迟到编码。还提取真实 vLLM 补丁中的客户端方法，检查 `clone → synchronize → ack` 顺序和同步失败时不释放；Text 代理测试检查 SDK 重试关闭及后端超时后预算保留。

2026-10-15 验证结果：本文件对应的生产改造测试 **24 项通过**；此前学习模式 **17 项通过**；原有 launcher / scheduling / autoscaler / profiler 共 **70 项检查通过**。`SKIP_INSTALL=1` 已在固定提交 `c53dc466b1b802d45d0b61cce36908334ea7a23e` 上实际应用完整 vLLM 补丁，并通过反向应用检查及 `chat_utils.py` 编译检查。环境为 WSL2、Python 3.10.12、PyTorch 2.5.1+cpu；没有安装或启动真实 vLLM 模型。

真实 GPU 验收仍须执行：多 Image/Text pair 交叉并发、实际 CUDA IPC 复制与确认、GPU 峰值显存、NCCL 故障，以及与 vLLM TP 同时运行。当前没有这些实测数据，也没有声称 OOM 已全面消除。

## 面试中的说明范围

可以基于代码和测试讲清：为什么图片名不能作传输键、如何统一收发顺序、为什么同一设备的多个通信组需要协调、HTTP 取消与设备完成的区别、IPC 单消费者 lease、预留/校正/回收，以及为什么异常时选择隔离。

CPU/Gloo 结果可以证明共享协议模块的行为；NCCL 性能、真实显存改善和生产模型吞吐需要补充 GPU 实验后再写到简历中。
