# ModServe 单卡学习版：改造计划

本仓库以 Azure/ModServe 的 c7a3e9ac116b735b14bfaeaf575ee048b4c02c23 提交为起点，保留原项目历史与 MIT 许可证。原项目地址：https://github.com/Azure/ModServe 。当前尚未完成单卡适配。

## 目标与边界

目标是在本机 WSL 的单张 RTX 4060 Laptop GPU（8 GiB 显存）上，做一个可运行、可测量的教学原型：保留图文请求分阶段处理、模态感知路由与 Text/Image Instance 接口。单卡版本不声称复现论文的多 GPU 并行编码、独立 GPU 资源扩缩容或吞吐量提升。

## 已知约束

- 原始 local.yaml 需要四张 GPU，默认模型为 Llama 3.2 Vision 11B。
- Text Instance 启动参数固定 BF16；修改版 vLLM 的图像特征协议主要针对 MLLAMA 和 INTERNVL。
- 图像 token 传输依赖 NCCL 与独立的 GPU 接收进程。
- 当前 autoscaling 只调整调度器中的活跃实例数量，并不会按负载启停 vLLM 进程。

## 分阶段工作

1. 建立无 GPU 的模拟 Image/Text Instance，跑通纯文本与图文请求，记录路由与阶段耗时。
2. 选择适合 8 GiB 显存且能接收外部视觉特征的小型多模态模型，确认权重、KV cache 与上下文长度预算。
3. 让图像编码在 CPU 上运行或采用串行单卡流程，以本机进程间传输替代跨 GPU NCCL；保留清晰的传输接口。
4. 将 Text Instance 的模型特定逻辑从路由逻辑中抽离，完成真实模型的端到端请求。
5. 对照单体基线测量正确性、峰值显存、TTFT、TBT 和吞吐量。明确报告单卡方案的性能限制。

原项目的安装与多 GPU 运行方法见 README.md。单卡流程完成前，不应直接运行默认拓扑。
