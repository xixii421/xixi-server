# 本机 Qwen3 PD：使用与验证

这是早期文本 PD 参考路径；当前项目主线见 [SmolVLM V/P/D](../vpd_local.md)。

这是新增的 Transformers 文本 PD 参考后端：Prefill、Decode 在两个独立进程中执行，
通过真实 Gloo 传递 KV。已使用本地 Qwen3-0.6B 权重验证 CPU 与 RTX 4060 Laptop 单卡回环。
它尚未接入原 ModServe 的多模态 vLLM 服务，与当前视觉/P/D 三阶段后端分别验证。

## 数据路径

```text
HTTP / SSE 入口（最多接纳两个请求，生成串行执行）
        │
        ├── Prefill 进程：完整 prompt -> 首 Token + KV
        │                      │
        │             请求身份 / 模型身份 / KV SHA-256
        │                      │ Gloo
        └── Decode 进程：接收并验证 KV -> ACK -> 单 Token 迭代
```

CPU 路径使用 BF16 或 FP32 CPU 张量。
单卡 CUDA 路径为 **GPU P -> CPU -> Gloo -> CPU -> GPU D**，两个模型进程共享同一张物理卡。
没有跨 GPU NCCL、CUDA IPC、NVLink 或 RDMA，因此不能用来宣称 PD 的生产性能收益。

- 交接校验请求 UUID、通道序号、prompt 摘要、模型配置/来源、dtype 和各层 KV 摘要。
- D 使用新分配的张量恢复 DynamicCache，收到缓存后每步只输入一个 Token。
- P 保留源缓存直到 D 完整接收、校验并确认；CUDA 路径先同步接收侧拷贝，再确认。
- 逻辑 KV payload 和生成长度有上限；预算不覆盖权重、激活、临时拷贝和 CUDA context。
- 同一通道一次运行一个请求，并发调用方串行排队；没有 Continuous Batching。
- 通道状态不明、worker 退出或输出消费者失败时停止并回收两个 worker，拒绝复用该通道。
- 当前固定长度 greedy，生成到指定 Token 数，暂不按 EOS 提前结束。
- 服务只接受文本消息；图片、多模态内容和未支持的参数明确拒绝。

## 环境

CPU 环境 `.venv-pd`：Python 3.10、PyTorch 2.5.1+cpu、Transformers 4.51.3。
CUDA 环境 `.venv-pd-cuda`：Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.51.3。
模型目录 `/path/to/Qwen3-0.6B`，加载设置 `local_files_only=True`。

重新创建 CPU 环境：

```bash
uv venv .venv-pd --python /usr/bin/python3
uv pip install --python .venv-pd/bin/python -r requirements-pd-local.txt
```

重新创建 CUDA 环境时，单独安装 GPU PyTorch，不安装 CPU 依赖文件：

```bash
uv venv .venv-pd-cuda --python /usr/bin/python3
uv pip install --python .venv-pd-cuda/bin/python torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-pd-cuda/bin/python \
  transformers==4.51.3 safetensors==0.5.3 numpy==2.2.6
```

## 运行模型与输出对齐

从实际 WSL 项目目录运行。输出文件存在时会拒绝覆盖，请为新实验换文件名。

```bash
cd xixi-server

# 无模型下载，真实随机微型 Qwen3 的双进程验证。
PYTHONPATH=src .venv-pd/bin/python -m lmm_serving.pd_local \
  --verify --output metrics/my-tiny-pd.json

# 真实 Qwen3 权重与未拆分模型对照；改为 cpu 可用 CPU 环境运行。
PYTHONPATH=src .venv-pd-cuda/bin/python -m lmm_serving.pd_local \
  --model /path/to/Qwen3-0.6B --device cuda:0 \
  --verify --new-tokens 8 --output metrics/my-qwen3-pd.json

# 自动启动临时 HTTP 服务，验证聊天模板、非流式输出和 SSE 文本均与参考结果一致。
PYTHONPATH=src .venv-pd-cuda/bin/python utils/validate_pd_model.py \
  --model /path/to/Qwen3-0.6B --device cuda:0 \
  --new-tokens 8 --output metrics/my-qwen3-http.json
```

`backend_ttft_ms` 与 `backend_tbt_ms` 来自独立 worker 的逐 Token 就绪时间。
首、次 Token 之间的间隔包含缓存交接及 D 的首次执行开销。
这些是后端时间，不能替代客户端 TTFT/TBT。`--verify` 另行执行参考模型；不是性能基准。

## 启动流式服务

```bash
PYTHONPATH=src .venv-pd-cuda/bin/python -m lmm_serving.pd_service \
  --model /path/to/Qwen3-0.6B --device cuda:0 --port 8099
```

服务只监听 `127.0.0.1`。`GET /health` 返回 worker 状态。
`POST /v1/chat/completions` 接受 `model`、文本 `messages`、`max_tokens`、
`temperature=0`、`n=1`、`stream` 与 `stream_options.include_usage`。
不支持完整 OpenAI API、鉴权、工具调用或多模态协议。

使用 [流式评测工具](pd_validation.md) 测量该端点时，payload 的 model 写 `Qwen3-0.6B`。
客户端测首个非空文本片段；TextStreamer 可能缓冲词片段，SSE 分块不等于逐 Token。
不要同时运行多组真实模型服务，以免重复加载模型占满 8GB 显存。

## 2026-10-30 验证结果

- PD 核心 14 项通过，包含真实微型模型、FP32/BF16、并发归属、消费者失败与进程回收。
- HTTP/SSE 5 项通过，运行真实微型 Qwen3 P/D，不替换模型计算。
- 流式评测 11 项通过。
- 真实 Qwen3-0.6B：CPU、CUDA 单卡回环生成的 8 个 Token 均与各自未拆分参考完全一致。
- 真实 Qwen3 HTTP/CUDA：非流式 Token、SSE 文本均与独立参考结果一致。

原始证据：

- `metrics/pd-local-regression-20261030/report.json`
- `metrics/pd-qwen3-0.6b-cpu-20261030.json`
- `metrics/pd-qwen3-0.6b-cuda-loopback-20261030.json`
- `metrics/pd-qwen3-0.6b-http-cuda-20261030.json`

当前实验只证明所测输入的功能正确性，未形成系统性的负载曲线、性能提升或成本结论。

## 已有的多模态三阶段入口

现已新增真实 SmolVLM/Idefics3 的 V/P/D 三阶段服务，见 [三阶段使用说明](../vpd_local.md)。
其中视觉特征与 KV 均由真实模型生成并在独立进程间交接，纯文本跳过视觉。

原 vLLM 的 Paged KV 与 Cross-Attention 状态尚未迁移；
原版 11B/26B 模型与多 GPU 通信仍需要额外适配及硬件验收。

适合当前阶段的简历表述：

> 在 ModServe 派生项目中新增文本 PD 参考后端，使用独立进程与 Gloo 完成 BF16 KV
> 及请求状态交接，支持流式接口、容量准入和故障后通道回收；在 CPU 与 RTX 4060
> 单卡回环环境中，验证 Qwen3-0.6B 输出与未拆分推理一致。

不宜写成已完成多模态 PD、跨 GPU 高性能 KV 传输，或已取得某个吞吐提升倍数。
