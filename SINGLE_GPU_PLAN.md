# ModServe 单卡学习版：改造计划

> 状态：规划完成，尚未开始单卡实现。以 [Azure/ModServe](https://github.com/Azure/ModServe) 的 `c7a3e9ac116b735b14bfaeaf575ee048b4c02c23` 为起点，保留原始 Git 历史与 MIT 许可证。本文件是后续改造的执行清单；完成一个关口后，更新复选框、实测数据和决策记录。

## 目标与验收口径

在 WSL Ubuntu 22.04、单张 RTX 4060 Laptop GPU（8 GiB 显存）上，实现可运行的图文推理教学原型。保留 ModServe 的 **Cluster Manager → Image Instance → Text Instance** 分阶段请求路径、模态感知路由以及各阶段耗时观测。先让纯文本与单图请求稳定完成，再评估多图和并发；模型回答应来自真实视觉特征，不能用占位图像张量充当“真实图文推理”。

项目分为两类可验收结果：

- **模拟链路**：无需模型权重或 GPU，可复现路由、排队、阶段交接、错误处理和指标；用于学习系统结构，不宣称模型正确性。
- **真实链路**：选定适配 8 GiB 的模型后，纯文本和单图均能生成合理结果，峰值显存不越界，进程能重复启动与退出；记录 TTFT、TBT、峰值显存和吞吐量。

论文中的多 GPU 并行图像编码、独立 GPU 池扩缩容和吞吐收益不属于单卡验收目标。单卡上“逻辑分离”不等于“物理资源分离”，性能结果必须按本机实测报告。

## 源码现状与改造落点

| 位置 | 当前实现 | 单卡需要处理的点 |
| --- | --- | --- |
| `configs/modserve/local.yaml`、`src/lmm_serving/launcher/topology.py` | 默认 2 个 Image GPU + TP2 Text GPU；校验禁止同机 GPU 重叠 | 新增独立单卡配置/校验模式，不改变原多 GPU 默认语义 |
| `src/lmm_serving/launcher/launch.py` | 为每个 Text Instance 另起 transfer-recv 进程，并为 Image Instance 配置 NCCL、numactl | 单卡模式使用 CPU Image Instance 与本地特征传输；只让 Text 后端持有 GPU |
| `src/lmm_serving/modserve/components/cluster_manager.py` | 维护图文路由、图像状态、token 负载与调度 | 尽量复用，抽出模型/传输相关假设；保留纯文本直达 Text 路径 |
| `src/lmm_serving/modserve/components/image_node_manager.py` | 启动时按 Llama/InternVL 加载 GPU 编码器，输出张量并用 NCCL 发送 | 拆出编码器适配层；先 mock，后 CPU 或经验证的单卡串行方案 |
| `src/lmm_serving/modserve/components/text_node_manager.py`、`src/lmm_serving/common/text_node.py` | 启动修改版 vLLM，dtype 固定 BF16，图像请求使用 `decoupled-multimodal-images` 协议 | 提供可配置的模型启动参数与适配接口；确认特征可被真实后端消费 |
| `src/lmm_serving/modserve/components/tensor_transfer_text_node.py`、`llm-backend/vllm-lmm-serving.patch` | NCCL 接收后以 CUDA IPC handle 供 vLLM 使用；补丁中的视觉协议针对 MLLAMA/INTERNVL | 定义本地传输与特征消费契约；不能只改传输脚本而不改后端 |
| `src/lmm_serving/config/config.yaml`、`profiles/` | 默认 11B、长上下文、较大 batch 与生产负载配置 | 单卡专用配置，显式限制上下文、图像数、并发与显存 |
| `tests/` | 主要覆盖原有拓扑、调度、autoscaling、profiler | 补充单卡拓扑、协议、mock 端到端和失败清理测试 |

原始 `configs/modserve/local.yaml` 需要 4 张 GPU；11B 默认模型不能直接用于 8 GiB 单卡。上游 `llm-backend/build_vllm.sh` 锁定特定 vLLM 提交和补丁，因此不能假定任意小型 VLM 只改 `MODEL` 就能接入。

## 实施阶段与完成条件

### M0：冻结基线与环境探测

- [ ] 记录 WSL、NVIDIA 驱动/CUDA、Python、PyTorch、显存与空闲磁盘；保存命令和结果到 `docs/single_gpu_progress.md`。不将模型权重、缓存和私有数据提交到 Git。
- [ ] 跑现有轻量测试（拓扑、调度、autoscaling）；记录通过/失败及依赖问题。暂不安装或启动 11B 默认栈。
- [ ] 固定原项目 `main` 行为：新路径用独立配置开关和文件实现，原多 GPU 配置及测试继续有效。

**完成条件**：有可复现的环境记录和基线测试结果，知道单卡启动前需要的依赖。

### M1：无 GPU 的端到端模拟链路

- [ ] 定义 `single_gpu_mock` 模式与单机拓扑：一个 Cluster Manager、一个 mock Image Instance、一个 mock Text Instance；保留原有多 GPU 拓扑校验。
- [ ] 让 mock Image Instance 返回带请求 ID、图像 ID、形状与耗时的**可识别模拟特征**；mock Text Instance 接收同一请求的特征元数据并返回确定性文本。纯文本请求绕过 Image Instance。
- [ ] 对两条路径进行端到端验证：纯文本直接到 Text，单图走 Image→Text；同时覆盖多图、找不到图片、超时、重复 ID、实例失败与完成后清理。
- [ ] 输出每次请求的排队、预处理、编码、传输、prefill、decode 耗时字段；mock 值须明确标记，避免和 GPU 实测混淆。
- [ ] 增加一条最小启动命令、一个纯文本样例和一个单图样例，均无需下载模型即可运行。

**完成条件**：从客户端到两个逻辑池的请求链路可一键复现，自动化测试验证路由、特征归属、错误传播和资源释放。此阶段不声称完成真实图文推理。

### M2：真实模型与特征接口可行性关口

- [ ] 列出候选小模型，逐项检查：权重与视觉编码器常驻内存、量化支持、单图视觉 token 数、最短可用上下文、vLLM/Transformers 版本、许可与下载权限。
- [ ] 先用**单体**推理在本机验证候选模型：短文本与单图各一例，记录峰值显存、加载时间、TTFT 和输出正确性。为 KV cache、视觉激活及框架开销留余量；量化能否工作以实测为准。
- [ ] 检查候选模型能否将外部图像特征注入 Text 后端。核对预处理、视觉投影层、token 位置/数量、dtype、shape、设备、请求生命周期与补丁接口；编写最小接口探针。
- [ ] 在此关口选择一条实现路线并写明理由：
  - **优先**：CPU 视觉编码 + GPU Text 后端 + 本机特征交接，前提是模型与后端协议兼容且 CPU 时延可接受；
  - **备选**：同卡顺序执行编码与生成，通过卸载/调度避免模型同时常驻导致 OOM；承认额外切换开销；
  - 若现有 vLLM 补丁无法支持该小模型，先做最小后端适配，或使用可注入特征的教学 Text 后端，不把普通 monolith 图文调用伪装成分离实现。

**完成条件**：有一份写明模型、版本、精度、显存、特征 shape/dtype、后端接入方式与风险的决策记录；单体基线已跑通。未通过时回到候选模型/后端选择，不盲目进入 M3。

### M3：单卡真实分阶段链路

- [ ] 增加可区分的传输接口：mock/NCCL/本机特征传输各自独立。单卡模式不启动 NCCL world 或独立 GPU 接收进程；跨进程特征可用有界 CPU 共享内存/本机 IPC，具体实现由 M2 验证结果决定。
- [ ] 用请求作用域的特征 ID/元数据完成交接。校验 shape、dtype、token 数、容量上限和目标请求；失败、取消、超时或生成完成后释放特征，防止跨请求错配与内存泄漏。
- [ ] 将编码器和模型架构判断从节点启动/路由代码中抽出；Text Instance 通过明确适配器接收特征。需要修改 vLLM 补丁时，锁定版本并对接口增加回归检查。
- [ ] 新增单卡配置：TP1、单图起步、小 batch、短上下文、保守 GPU 内存比例；OOM 时优先收紧上下文/并发/图像 token，再评估量化与卸载。
- [ ] 跑纯文本、单图、连续请求、错误恢复和重复启动/关闭；确认图像确实影响输出（同一问题替换两张明显不同图片作对照），并记录正确性限制。

**完成条件**：一张 8 GiB GPU 上真实图文请求完整经过 Image→Text 两阶段，峰值显存可控，不依赖第二张 GPU 或远程服务；原多 GPU 路径的既有测试未回退。

### M4：测量、对照与项目交付

- [ ] 固定模型、图片、提示词、输出长度与环境，分别测单体基线和单卡分阶段方案；先预热，再记录 TTFT、TBT、吞吐量、峰值显存和各阶段耗时，说明样本数及 P50/P95。
- [ ] 分别测纯文本、单图，以及在资源允许时的多图与轻并发；记录 OOM、超时和模型输出差异。不要把 mock 延迟放进真实模型对比。
- [ ] 用请求日志展示模态感知路由和排队决策。单卡仅有一个 Image/Text 实例时无法比较负载均衡收益；如需展示路由算法，用 M1 的多个 mock 实例做功能验证。
- [ ] 完善 README：明确单卡启动命令、模型准备、已知限制、故障排查与原多 GPU 用法的区别；补齐复现脚本和最小结果表。

**完成条件**：他人可依据 README 在相近单卡机器复现最小链路，并能区分系统功能演示和论文原始性能主张。

## 首轮任务顺序

1. M0 环境与基线测试；保留输出。
2. M1 先打通 mock 路由与请求生命周期，暂不下载大模型。
3. M2 做小模型/外部视觉特征接口的可行性试验；在选型通过前不大改 vLLM。
4. M3 接入真实编码、传输与 Text 后端。
5. M4 做对照测量与文档。

## 决策和风险记录

| 关口 | 当前判断 | 验证方式 / 失败后动作 |
| --- | --- | --- |
| 8 GiB 显存 | 默认 11B+BF16+长上下文不可作为单卡起点 | M2 实测小模型与量化；不足则缩上下文/换模型 |
| 特征协议 | 当前补丁主要绑定 MLLAMA/INTERNVL，任意小 VLM 未必兼容 | 先做外部特征探针；失败则适配后端或选教学后端 |
| 编码放置 | CPU 编码可让 GPU 留给 Text，但可能很慢 | 测加载、编码时间与内存；必要时评估顺序同卡方案 |
| 本机传输 | 简单传字节可运行，但易复制过多且需生命周期管理 | 在单图上先验证正确性，再按耗时决定是否优化 IPC |
| 自动扩缩容 | 单 GPU 没有可扩的 GPU 实例池 | 第一版只做逻辑调度/背压，不承诺资源扩缩收益 |
| 上游兼容 | 过早改写原路径会损伤多 GPU 行为 | 单卡模式隔离配置与适配层，持续运行原有测试 |

## 进度记录模板

每完成一阶段，在 `docs/single_gpu_progress.md` 追加：日期、提交、环境、执行命令、测试结果、显存/时延数据、失败原因、下一步。模型与依赖版本必须精确记录。未经实测的数字不得写成结果。
