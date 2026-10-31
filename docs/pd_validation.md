# PD 扩展：当前实现状态与验收

实际 WSL 工作分支：`codex/pd-local`。基线提交：
`9c78f7bf3df65ebe4f47d8cb3819228c5b601ecc`。

本机新增的文本 PD 参考后端已完成并验证，使用与结果见 [本机 PD 说明](pd_local.md)。
真实 SmolVLM 三阶段服务也已完成并在单卡验证，见 [V/P/D 服务](vpd_local.md)。
下面的待完成项专指原 ModServe 多模态 vLLM 生产路径。

## 2026-10-30 本机验证记录

WSL Ubuntu、Python 3.10.12、PyTorch 2.5.1+cpu；新增流式工具也在 Windows Python 上通过。

| 测试套件 | 结果 |
|---|---|
| 新增流式评测（真实本机 HTTP/SSE fixture） | 11 项通过 |
| 生产协议与 CPU/Gloo 传输 | 24 项通过 |
| CPU/Gloo 学习链路 | 17 项通过 |
| 调度 | 9 项检查通过 |
| 扩缩容 | 20 项检查通过 |
| 性能估计模型 | 2 项检查通过 |
| 启动器 | 41 项检查通过 |

启动器原有测试出现 3 项失败：路径含空格时的断言未按 shell 解析，以及两项结构测试
依赖本机必须有四卡。已修正测试：结构检查显式使用四卡 fixture，另验证单卡拒绝四卡拓扑；
生产环境的硬件检查没有关闭。

真实日志位于 `metrics/pd-regression-20261030/` 与
`metrics/pd-regression-fixed-20261030/`，后者记录修复后的启动器、流式工具与生产协议结果。
首轮报告中的失败保留，未覆盖。

可复现运行器（指定新的结果目录）：

```bash
python utils/run_regressions.py --output-dir metrics/new-regression
```

运行器为每个测试保存日志及 JSON 结果，并在任一测试失败或超时时返回非零。
这些测试验证协议和工具，不是 PD 后端或 GPU 性能验收。

## 已完成的评测工具

新增 `lmm_serving.stream_benchmark`，向 OpenAI 兼容的
`/v1/chat/completions` 端点发送真实 HTTP 流式请求，保存逐请求 JSONL。
工具不依赖 PyTorch、模型权重或 OpenAI SDK。

准备 `request.json`，填写实际运行的模型名：

```json
{
  "model": "YOUR_RUNNING_MODEL",
  "messages": [{"role": "user", "content": "Explain KV cache briefly."}],
  "temperature": 0,
  "max_tokens": 64
}
```

在需要比较的实际端点运行：

```bash
PYTHONPATH=src python -m lmm_serving.stream_benchmark \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --payload request.json --requests 20 --concurrency 4 \
  --ttft-slo-ms 1000 --output baseline-stream.jsonl

PYTHONPATH=src python tests/test_stream_benchmark.py
```

约束与统计口径：

- 固定并发的闭环负载，计时从各请求实际发送前开始；不模拟生产到达轨迹。
- TTFT 是客户端从开始请求到首个非空 `delta.content` 的时间。
  角色声明、空内容和心跳不计为首内容。推理模型的 reasoning 字段不计入。
- SSE 事件可能包含多个 Token，因此仅记录内容事件间隔，`tbt_ms` 保持 null。
  这些间隔不能作为逐 Token TBT 的精确值。
- 输出 Token 数来自服务端 usage。未返回 usage 时保持 null，不按事件数推测。
- TTFT 达标率以所有请求为分母，失败和超时请求不能从分母中删除。
  当前 goodput 仅按 TTFT 判断，不代表 TTFT/TBT 联合 SLO goodput。
- 保留 HTTP、流内错误和不完整输出；要求 finish reason 和 `[DONE]`。
- `--timeout` 是 socket 空闲超时，不是总请求 deadline。没有自动重试。
- 每个完成的请求立即落盘，输出文件存在时拒绝覆盖。
- 工具不自动预热；冷启动、编译、预热与正式测量需由实验者明确区分。
- 本机 HTTP fixture 的测试结果只验证评测工具，不代表 GPU 性能数据。

现有 ModServe `/generate` 只返回完整响应。该接口必须先补齐端到端流式透传，
才能用同一客户端测量整个视觉/P/D链路。直接测 vLLM 端点时，作用域仅是该后端。

## PD 接入前发现的兼容性问题

原有 modserve/monolith 生产启动器尚未增加独立 P/D 拓扑或 KV Connector 配置。
它固定的 vLLM 提交为
`c53dc466b1b802d45d0b61cce36908334ea7a23e`，不能直接套用最新版 V1 Connector API。

该版本的 `SimpleConnector` 使用 `model_executable.model`、
`model.layers[i].self_attn` 等结构访问 KV。为 InternVL、MLLAMA 接入时，需要检查并适配
模型包装层和注意力布局，不能只增加 `--kv-transfer-config`。

还需要核查以下两点：

1. 缓存查找不能只依赖文本 Token。相同文本、不同图像的请求不能复用错误 KV。
   需要让请求身份或完整多模态身份贯穿 P/D 的缓存交接。
2. MLLAMA 后续 Decode 所需的视觉/Cross-Attention 状态必须完整交接。
   传输文本 Self-Attention KV 不能被预先认定为足够。

## 尚未完成的验收

- 选定实际目标模型、后端和 GPU 拓扑，完成 PD 模型适配。
- 集成 P/D 路由、KV 交接、资源准入、异常回收和流式透传。
- 在真实模型上对照单体输出，覆盖纯文本、单图、多图和相同文本不同图片。
- 对请求取消、传输超时、接收失败进行故障测试，确认无错误缓存复用和提前回收。
- 在相同总 GPU 预算下对比原版 ModServe、PD 静态分配及调度改进。
- 使用真实逐 Token 测量补齐 TBT，报告 P99、SLO 达标率、goodput 和 KV 传输开销。

本机仅检测到一张 RTX 4060 Laptop 8GB。CPU/Gloo 可以验证部分协议行为，
无法替代不同 GPU 间 NCCL、真实 CUDA IPC 和论文规模多模态模型的验收。
以上未完成项不能写成已实现功能或实测收益。
