# ModServe 来源与保留路径

xixi-server 从 [Azure/ModServe](https://github.com/Azure/ModServe) 派生，保留原实现、vLLM 补丁和提交历史，遵循原项目的 [MIT 许可证](../../LICENSE)。
上游架构、原模型适配及 [论文](https://dl.acm.org/doi/10.1145/3698038.3698540) 结果属于原作者。
本仓库首页的性能数字来自新增 SmolVLM 后端的独立实验。

| 路径 | 来源与作用 |
|---|---|
| `src/lmm_serving/modserve/`、`monolith/`、`profiler/` | 保留的原服务、单体部署和容量测量代码；派生开发中增加了可靠性与预算管理 |
| `src/lmm_serving/common/`、`launcher/` | 包含保留逻辑与派生修改的公共模块、拓扑和启动器 |
| `llm-backend/` | 固定版本 vLLM 构建脚本与补丁；未接入新 SmolVLM 三阶段后端 |
| `profiles/` | 上游容量参考值，使用前须在目标硬件重新测量 |
| `traces/` | Azure 请求轨迹的回放样本，数据来源见 [Azure 公共数据集](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md) |

新增主线为 `vpd_local.py`、`vpd_router.py`、`vpd_transport.py`、`vpd_service.py`、`vision_cache.py`、
单体对照、真实模型验收和基准工具。早期 Qwen3 PD 与 CPU/Gloo 模拟链路也保留作参考，
均与上游 vLLM 运行路径分开。

## 原版服务环境

以下用于原版 vLLM 路径；SmolVLM 主线依赖和启动见 [项目首页](../../README.md)。
原环境固定 PyTorch 2.5.1、Transformers 4.48.x 相关依赖，不与主线 PyTorch 2.6.0 / Transformers 4.51.3 混装。

```bash
pixi install --locked
pixi run setup
pixi run python -m lmm_serving.launcher --help
```

`pixi run setup` 构建打补丁的 vLLM 与 flash-attn，需要匹配的 CUDA 构建环境。
本轮公开整理没有重新验收 11B/26B 模型或上游多机部署。

- [配置参数](configuration.md)
- [拓扑部署](topology.md)
- [可靠性、传输预算与背压](modserve_reliability.md)
- [性能采样与请求回放](experiments.md)
