# xixi-server

**多模态大模型推理服务：ModServe 论文复现与改进。**

本项目基于 [Azure/ModServe](https://github.com/Azure/ModServe)，围绕论文 [ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving](https://dl.acm.org/doi/10.1145/3698038.3698540)（SoCC 2025）开展复现与工程改进。保留视觉编码与语言推理解耦的服务架构、模态感知路由、vLLM 多模态接入和请求轨迹回放，在此基础上完善跨阶段特征传输、请求准入、资源背压与异常回收。

上游提供了模型适配、分阶段服务、基础路由及 NCCL / CUDA IPC 数据通路；本仓库的主要改动集中在这些通路的并发正确性和资源生命周期管理。论文中的吞吐与成本结果属于上游实验，不作为本项目的实测成果。

## 服务架构

多模态请求拆分为视觉阶段和语言阶段，两类实例可以分别配置资源；纯文本请求直接进入语言阶段。

```text
客户端 / 请求轨迹回放
          │ HTTP
          ▼
   Cluster Manager
     │          │
     │          └───────────────────────────┐
     ▼                                      ▼
Image Instances                       Text Instances
图像预处理 / 视觉编码                  Prefill / Decode
     │                                      ▲
     └── NCCL → 特征接收服务 → CUDA IPC ──────┘
```

- **视觉侧**：处理图像预处理、视觉编码和特征缓存；同一请求的多张图片可分发到不同实例并行编码。
- **语言侧**：通过补丁接入的 vLLM 消费外部视觉特征，完成语言模型推理。
- **路由侧**：保留 `round_robin` 和 `least_loaded`；视觉侧按图像 Token 估计负载，文本侧根据模型架构使用文本或图文 Token 估计负载。
- **对照路径**：保留 `monolith`，由同一 vLLM 实例执行完整视觉语言模型，便于对照分阶段部署。

上游支持的模型包括 Llama 3.2 11B Vision Instruct（Cross-Attention）与 InternVL2.5-26B（Decoder-Only）；对应配置和模型适配代码保留在仓库中。

## 主要改进

### 1. 请求级特征标识与收发排序

使用内部请求 ID 和图像序号组成 `request_id:index`，避免重复图片或相似文件名导致缓存覆盖、错取。接收服务统一安排每对通信节点的 FIFO、通道 epoch 与序号，再触发源端发送；传输身份头用于校验特征归属和元数据。

同一传输进程内的多个通信组共享设备通信许可，并按 Image → Text 的固定顺序取得许可，约束不同通信组的执行顺序。

### 2. 编码前准入与资源背压

在视觉编码开始前，预留 Text 侧的图像 Token、特征字节和请求槽位，以及 Image 侧的特征字节和缓存槽位。编码完成后按实际张量形状原子校正预算，避免先产生大量特征再等待下游容量。

单请求超过节点容量时返回 `413`，节点暂时没有可用容量时返回 `429`。编码 HTTP 并发通过有界线程池控制；预算失败时取消请求并执行幂等清理。

### 3. IPC lease 与异常生命周期

接收侧使用一次性 IPC lease，限定单个消费者领取特征。vLLM 完成 `clone → synchronize` 后，凭 `lease_token` 确认释放，避免复制尚未结束时源缓冲区被提前回收。

区分排队、编码、传输和 IPC 消费等状态。HTTP 超时不会直接释放仍被设备使用的张量；结果不明的传输或后端执行保留资源并隔离节点，采用整组重启恢复。

### 4. 生产模块复用与回归验证

将预算、特征状态、请求准入、传输服务和图像流水线拆为可独立验证的模块。测试复用实际模块，通过跨进程 Gloo 传输 BF16 张量，覆盖同图并发、多图、容量拒绝、实际大小校正、部分失败和迟到编码；同时验证后端 IPC 确认顺序与超时预算保留。

另提供无需模型与 GPU 的 CPU/Gloo 单卡学习版，模拟编码与生成、保留真实进程间通信，见[学习指南](docs/single_gpu_learning.md)。

## 环境与安装

原版模型服务使用 Linux、Python 3.10、CUDA 12.4 及匹配驱动，并依赖项目补丁版本的 vLLM。默认 `local.yaml` 是四卡拓扑，运行前需核对模型显存需求和 GPU 分配。

```bash
git clone https://github.com/xixii421/xixi-server.git
cd xixi-server

# 已安装 Pixi 时，创建环境并构建项目依赖与后端。
pixi run setup
```

也可以按 [requirements.txt](requirements.txt)、[pixi.toml](pixi.toml) 和后端脚本配置已有 Conda / Python 环境。项目继续使用 `lmm_serving` 包名和 `modserve` / `monolith` 命令参数。

协议改动需要所有节点同步更新，并重新应用 vLLM 补丁。已有安装不能只依赖 setup 的导入检查：

```bash
# 在原版运行环境中重建补丁后端。
bash llm-backend/build_vllm.sh

# 仅下载固定源码并应用补丁，不执行安装。
SKIP_INSTALL=1 bash llm-backend/build_vllm.sh
```

后端源码固定在 `c53dc466b1b802d45d0b61cce36908334ea7a23e`。完整部署要求和协议说明见[可靠传输与资源背压](docs/modserve_reliability.md)。

## 配置与运行

全局配置位于 `src/lmm_serving/config/config.yaml`，实例拓扑位于 `configs/modserve/` 和 `configs/monolith/`。先配置模型、图片目录、权重缓存、节点地址及 GPU，再准备请求回放数据。

```bash
# 检查拓扑；会核对可检测到的本机 GPU 数量。
pixi run python -m lmm_serving.launcher validate modserve

# 只打印启动计划。
pixi run python -m lmm_serving.launcher run modserve 2.0 --dry-run

# 在模型、图片和轨迹均已准备好后执行回放。
pixi run python -m lmm_serving.launcher run modserve 2.0

# 单体部署对照。
pixi run python -m lmm_serving.launcher run monolith 2.0
```

`2.0` 为轨迹负载缩放参数；具体数据准备、采样与回放方式见[实验指南](docs/experiments.md)。生成指标写入 `metrics/<setup>/<run_id>/`，不纳入 Git。`profiles/` 内的数据是上游参考值，启用容量估计或自动扩缩容前应在目标硬件上重新采样。

常用资源配置包括 `TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET`、`TEXT_FEATURE_BYTE_BUDGET`、`IMAGE_FEATURE_BYTE_BUDGET`、`TEXT_MAX_IMAGE_REQUESTS` 和 `TRANSFER_QUEUE_SIZE`，含义与默认值见[配置说明](docs/configuration.md)及[可靠性文档](docs/modserve_reliability.md)。

## 验证与边界

```bash
# 使用已配置好依赖的 Python 环境。
PYTHONPATH=src python tests/test_modserve_reliability.py
PYTHONPATH=src python tests/test_scheduling.py
PYTHONPATH=src python tests/test_autoscaler.py
PYTHONPATH=src python tests/test_profiler.py
PYTHONPATH=src python tests/test_launcher.py
```

2026-10-17 回归结果：可靠性 24 项、调度 9 项、扩缩容 20 项、profiler 2 项通过。Launcher 在本机单卡环境下有两项默认四卡拓扑断言失败；显式模拟四卡数量后，39 项检查全部通过，并确认单卡环境会拒绝原版四卡拓扑。

CPU/Gloo 测试验证了协议与资源管理行为，未完成真实模型端到端、NCCL 多通信组并发、CUDA IPC、峰值显存和吞吐的 GPU 验收。字节预算覆盖特征 payload 及接收 / clone 的保守计账，不覆盖模型权重、全部中间激活和 KV Cache；异常隔离后需要整组重启，不包含在线通信组修复。

## 代码与文档

| 路径 | 职责 |
|---|---|
| `src/lmm_serving/modserve/pipeline.py` | 图像请求编排、资源预留、并发编码与清理 |
| `src/lmm_serving/modserve/components/` | Cluster、Image、Text 与特征接收服务 |
| `src/lmm_serving/common/` | 路由、预算、特征状态、准入与传输协议 |
| `src/lmm_serving/launcher/` | 拓扑校验、进程编排、本机与 SSH 启动 |
| `src/lmm_serving/monolith/` | 单体部署对照路径 |
| `llm-backend/` | 固定版本 vLLM 构建脚本与多模态补丁 |
| `client/`、`traces/` | 请求轨迹回放与输入样本 |
| `tests/` | 协议、流水线、路由及启动回归测试 |

- [可靠传输与资源背压](docs/modserve_reliability.md)：协议、资源边界、部署与验证。
- [配置说明](docs/configuration.md)：全局参数及环境变量。
- [拓扑说明](docs/topology.md)：单机、多机与实例分配。
- [实验指南](docs/experiments.md)：性能采样、轨迹准备与回放。
- [项目目录说明](docs/project_structure.md)：目录职责与生成物管理。

## 论文与上游

本项目保留上游提交历史及作者归属。使用原始架构、模型适配或论文方法时，请引用原论文；工程改动的实现与验证范围以本仓库代码、测试及文档为准。

- 论文：[ModServe，SoCC 2025](https://dl.acm.org/doi/10.1145/3698038.3698540)
- 上游实现：[Azure/ModServe](https://github.com/Azure/ModServe)
- 请求轨迹：[Azure LMM Inference Trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md)
