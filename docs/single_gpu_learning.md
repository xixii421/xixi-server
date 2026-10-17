# 单卡 / CPU 通信学习模式

本模式名为 `single_gpu_mock`，在没有 GPU 或只有一张 GPU 的 Linux / WSL 机器上运行。它使用三个操作系统进程：主进程运行 Cluster Manager，两个 `spawn` 子进程分别运行 Image、Text 和 Gloo rank 0、1。模型编码、prefill、decode 是模拟阶段，特征张量通过 Gloo 真实传输。无需 Hugging Face 账号、图片数据集、模型权重、CUDA、flash-attn 或修改版 vLLM。

## 安装和启动

从仓库根目录执行：

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements-learning.txt
PYTHONPATH=src .venv/bin/python -m lmm_serving.launcher learn --demo
```

依赖文件固定 CPU 版 PyTorch 2.5.1。不要为本模式执行 `pip install -e .` 或 `pixi run setup`，它们安装的是原始大模型环境。如果不用 uv，可以在新的、具有 pip 的 Python 虚拟环境中执行 `python -m pip install -r requirements-learning.txt`，后续同样设置 `PYTHONPATH=src`。

`--demo` 在临时目录生成一张红色图片，自动发起纯文本和 `red.ppm` 单图请求，打印 JSON 响应后关闭进程并清理图片。需要常驻时，指定自己的图片目录：

```bash
PYTHONPATH=src .venv/bin/python -m lmm_serving.launcher learn --image-dir /path/to/images
```

也可使用独立入口 `python -m lmm_serving.learning`，参数相同。Ctrl+C / SIGTERM 会关闭服务并回收子进程。默认占用四个回环端口：Cluster HTTP 7070、Image HTTP 7081、Text HTTP 7082、Gloo rendezvous 29510；不会关闭其他实验的进程。

## 请求样例

另开一个终端。省略 `req_id` 时服务自动分配唯一 ID。下文的 `red.ppm`、`blue.ppm` 需要换成自己图片目录中实际存在的文件名；仓库不再附带图片。

```bash
# 纯文本：Cluster → Text，Image 编码计数不增加。
curl -s http://127.0.0.1:7070/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello"}'

# 单图：Cluster → Image → Gloo → Text。
curl -s http://127.0.0.1:7070/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"描述图片","image_ids":["red.ppm"]}'

# 多图：重复文件也获得不同的 transfer_id，保持输入顺序。
curl -s http://127.0.0.1:7070/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"比较图片","image_ids":["red.ppm","red.ppm","blue.ppm"]}'

# 查看 Gloo rank、进程 PID、收发计数、未释放特征数和字节数。
curl -s http://127.0.0.1:7070/health
```

`features[*]` 包含 `image_id`、唯一 `transfer_id`、`shape=[4,8]`、`dtype=float32`、校验和以及 `verified=true`。红、蓝图片产生不同特征，同一图片产生相同特征内容；请求身份独立于图片内容。返回文本以 `[MOCK]` 开头，只组合提示词、图片名和特征校验和，不具有图片理解能力。

接口只接受 `req_id`、`prompt`、`image_ids`。不支持的推理参数会返回 400，不会静默忽略。自定义图片放在 `image_dir` 内，支持 Pillow 能读取的格式；绝对路径和符号链接也必须解析到该目录内部。

## 数据与控制路径

```text
客户端 ─HTTP→ Cluster Manager（复用 common.routing 调度器）
                    │
                    ├─ 纯文本 ─HTTP─────────────────────────→ Text
                    │
                    └─ 图片 ─HTTP→ Image 编码并缓存 CPU 特征
                                      │
                             rank 0 ─Gloo→ rank 1
                                               │
                                        Text 校验并消费
```

1. Cluster 在接收请求时生成内部 request key；每张图片使用 `request_key:image_index` 作为传输 ID，文件名不参与缓存寻址。
2. Image 读取图片 RGB 字节和尺寸，将 SHA-256 摘要转换成 4×8 float32 模拟特征，保存到有容量和有效期限制的 FeatureStore。
3. Cluster 用全局图片流水线锁统一安排全部图像请求；每次并发触发 Image 的 `/transfer` 与 Text 的 `/consume`。两个控制请求的顺序由同一个协调者管理。
4. Gloo 先传 32 字节的传输身份摘要，再传特征。Text 检查身份和特征校验和后才保存和消费。
5. 成功或失败均发送幂等释放请求。取消标记与 deadline 检查防止较晚完成的编码、接收重新留下特征；空闲时也会回收过期项。

第一版一次执行一条图片流水线，以保证顺序并便于学习；纯文本可并发。没有实现 TP、CUDA IPC、NCCL、NVLink、RDMA 或跨机器部署。原版 `modserve` 的拓扑校验、NCCL 启动和大模型配置保持原有语义。

## 配置和故障行为

默认文件为 `configs/single_gpu_mock/local.json`。可复制后修改：

```bash
PYTHONPATH=src .venv/bin/python -m lmm_serving.launcher learn \
  --config /path/to/learning.json \
  --image-dir /path/to/images \
  --metrics-dir /tmp/modserve-learning-metrics
```

配置中的相对图片、指标路径以仓库根目录为基准；传入的相对配置文件路径以当前目录为基准。`--port` 仅覆盖 Cluster HTTP 端口；同时开多套服务时，应为配置中的全部四个端口选择不同值。

| 参数 | 作用 |
|---|---|
| `max_inflight` | 同时接纳的请求数，包含排队请求；超额返回 429 |
| `max_images` | 每个请求的最大图片数 |
| `max_image_bytes` / `max_image_pixels` | 单张图片文件大小 / 解码像素上限，超额返回 413 |
| `max_pending_bytes` | 每个 worker 的特征缓冲区上限，超额返回 429 |
| `request_timeout_s` | 整条请求链路的截止时间，超时返回 504 |
| `transfer_timeout_s` | Gloo 进程组超时，必须小于请求超时 |
| `startup_timeout_s` | 等待两个 worker 就绪的上限 |
| `feature_ttl_s` | 特征最长保留时间，实际也受请求 deadline 限制 |
| `mock_encode_ms` | 每张图片模拟编码等待时间，可增大以观察排队和超时 |

找不到图片返回 404，非法请求或越界路径返回 400，活动中或最近完成的重复 `req_id` 返回 409。重复 ID 记录保留最近 1024 个已完成请求，重发时应使用新 ID。

开始传输前，剩余预算必须大于 `transfer_timeout_s + 0.5` 秒，否则返回 504。HTTP 排队/编码超时可清理后继续服务；如果 Gloo 收发出错、身份不匹配或任一 worker 退出，监督进程会停止整套服务并以非零状态退出，之后需重新启动。不能在双边通信超时后直接复用原 communicator。

## 指标

完成的请求逐条追加到 `metrics/single_gpu_mock/requests.jsonl`，包含失败请求和多张图片的独立记录；无需正常退出才保存。多次启动默认追加同一文件，可用 `--metrics-dir` 区分实验。

- `queue_ms`：在 Cluster 等待图片流水线锁的实测时间。
- `preprocessing_ms`：读取、解码、摘要计算的实测 CPU 时间。
- `encoding_ms`：模拟等待和构造特征的实测时间，不是模型编码时延。
- `transfer_ms`：HTTP 协调及 Gloo 传输的总实测时间；每张图片另有 `gloo_send_ms` / `gloo_recv_ms`，其中接收计时包含等待发送者的时间。
- `mock_generation_ms`：模拟结果构造和校验的时间。
- `prefill_ms`、`decode_ms`：占位值 0；没有执行模型计算，也没有真实 TTFT / TBT。
- `e2e_ms`：包括清理在内的服务端端到端耗时。

所有记录都带 `mode=single_gpu_mock`、`transport=gloo` 和 `simulated_stages`。这些指标不能用作论文性能复现或 GPU 间通信带宽。

## 验证与源码阅读顺序

```bash
PYTHONPATH=src .venv/bin/python tests/test_learning.py
```

测试使用本机动态端口和真实 Gloo 子进程，覆盖文本绕过、单/多图、重复图片、并发归属、非法请求、重复 ID、容量限制、超时清理与恢复、真实 Gloo 接收超时、worker 故障、关闭后同端口重启。需要允许本机监听 socket；受限执行沙箱中可能需要额外执行权限。

建议依次阅读：

1. `learning/cluster.py`：路由、准入、统一排序和双端触发。
2. `learning/workers.py` 的 `GlooTransport`：rank、`send/recv`、身份头与特征校验。
3. `learning/protocol.py`：有界缓存、归属、截止时间和取消后的清理。
4. `learning/runtime.py`：`spawn`、就绪检查和故障后回收。
5. 对照原版 `image_node_manager.py`、`tensor_transfer_text_node.py`、`vllm-lmm-serving.patch`，理解 CPU/Gloo 与 GPU/NCCL/CUDA IPC 的差异。
