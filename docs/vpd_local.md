# 本机视觉 / Prefill / Decode 三阶段服务

多卡设备池和 NCCL 数据通道见 [多卡部署与验收](vpd_multigpu.md)。
本文介绍单卡/Gloo 用法；当前已完成的真实四卡结果见 [四卡报告](validation/four-gpu-20261101.md)。

本机实际使用 [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
预训练视觉语言模型，三个独立进程完成完整图像到文本生成，不以模拟视觉特征替代编码。

## 启动

按 [安装说明](../README.md#快速开始) 创建 `.venv-vpd`，准备本地模型目录。
已验证的 CUDA 环境为 Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.51.3、Pillow 11.1.0。

```bash
cd xixi-server
PYTHONPATH=src .venv-vpd/bin/python -m lmm_serving.launcher vpd \
  --model /path/to/SmolVLM-256M-Instruct \
  --device cuda:0 --port 8099
```

也可以使用 `python -m lmm_serving.vpd_service`，参数相同。
端点是 `http://127.0.0.1:8099/v1/chat/completions`；`GET /health` 查看状态。
输出启动信息包含 V/P/D 三个 worker PID。

## 发送本地图片

启动服务后另开终端，在项目目录运行下面的 Python 示例。
将 `photo.png` 改为实际 PNG 或 JPEG 路径。请求只发往本机，不自动抓取远程图片。

```python
import base64, json
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

image = Path("photo.png")
mime = "image/jpeg" if image.suffix.lower() in (".jpg", ".jpeg") else "image/png"
data_url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()
payload = {
    "model": "SmolVLM-256M-Instruct",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": "Describe this image."}
    ]}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True}
}
request = Request("http://127.0.0.1:8099/v1/chat/completions",
                  data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with build_opener(ProxyHandler({})).open(request, timeout=60) as response:
    for line in response:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError(chunk["error"])
        for choice in chunk.get("choices", []):
            print(choice.get("delta", {}).get("content", ""), end="", flush=True)
```

在 content 中按顺序加入多个 image_url 即为多图请求；纯文本可用字符串 content。
当前最多 2 张图、每图 2MiB、每图 1,048,576 像素，HTTP body 最多 8MiB。
超额请求返回 413；默认已接纳请求达到 4 个时返回 429，可用 `--max-inflight` 调整。
只支持 greedy、n=1、最多 32 个新 Token。固定长度生成，尚未按 EOS 提前结束。

默认启用 16MiB CPU 视觉特征 LRU，可用 `--vision-cache-mib 0` 关闭。
缓存按图片内容和模型/预处理命名空间寻址，支持同图不同问题、部分命中和多图换序。
命中跳过视觉编码，仍执行图像预处理、P/D 和必要的传输。
`stream_options.include_backend_metrics=true` 可在最终 SSE 元数据中返回命中、编码图数和阶段耗时。
当前实验见 [四卡报告](validation/four-gpu-20261101.md)，旧版本结果见 [历史单卡消融](vpd_benchmark.md)。

## 三阶段如何交接

```text
入口：验证图片格式、大小、顺序及预算，为请求分配 UUID 和通道序号
  |
  +-- 有图 -> V 进程：预处理 -> 真实视觉 Transformer -> Connector 投影
  |               |
  |               +-- Gloo 发送视觉特征、请求身份、图像顺序、SHA-256 -> P 进程
  |                                                                   |
  +-- 纯文本 ---------------------------------------------------------+
                                                                      |
                P：替换图像占位符 -> 完整 Prefill -> 首 Token + KV Cache
                                                                      |
                D：分配本地 KV -> 校验并 ACK -> 每步输入一个 Token -> SSE
```

### 模型权重按职责拆开

- V 只保留视觉编码器与投影 Connector。
- P、D 只保留语言 Transformer 和 LM Head，移除视觉编码器及 Connector。
- 单体参考模型仍直接接受 pixel_values，使用完整原模型前向，没有经过拆分特征注入路径。
- 验收基线与三阶段采用同一图像预处理：关闭切图，长边限制为 512，保持模型图像 Token 对齐。

该架构在 Prefill 将视觉特征替换进语言序列，视觉信息随后包含在 Self-Attention KV 中。
因此 D 只需接收完整 KV 和首 Token，无需重复编码图片；这不等同于 Cross-Attention 模型的状态交接。

### 身份与生命周期

- V -> P 校验 UUID、序号、模型签名、原始图片摘要与顺序、prompt 摘要、shape/dtype 和特征内容摘要。
- P -> D 复用 KV 协议，额外绑定有序图片上下文摘要，避免相同文字、不同图片之间的缓存错配。
- 接收端确认实际数据完整，CUDA 模式确认拷贝完成后才 ACK，发送端随后释放源特征或缓存。
- 视觉预处理拒绝发生在数据传输前，后续合法请求仍可使用通道。
- worker 退出、状态不明或输出消费者断开时，停止并回收所有 worker，唤醒全部在途请求。
- 并发调用方按空闲 V/P/D 槽位路由，V/P 交接后可复用，D 保留到生成结束；默认接纳上限为 4。
- 事件泵按请求 ID 和 rank 分发事件，各 SSE 流分别按索引重排，不依赖不同进程消息到达顺序。

## 验证

真实权重验收脚本会先独立执行单体模型，再启动 V/P/D，最后测真实 HTTP/SSE。
它自动创建纯色图片作为可复现输入，不需要用户提供照片。

```bash
PYTHONPATH=src .venv-vpd/bin/python utils/validate_vpd_model.py \
  --model /path/to/SmolVLM-256M-Instruct --device cuda:0 \
  --new-tokens 8 --output metrics/my-vpd-verification.json
```

输出文件已存在会拒绝覆盖。运行验收前关闭其他真实模型服务，避免重复占用显存。

无需下载权重的测试使用真实随机微型 Idefics3（真实视觉与语言计算，并非 mock）：

```bash
PYTHONPATH=src .venv-cpu/bin/python tests/test_vpd.py
PYTHONPATH=src .venv-cpu/bin/python -m lmm_serving.launcher vpd \
  --dtype float32 --port 8099
```

微型测试模型没有训练，model 名为 `tiny-idefics3`，只用于协议和正确性回归。
重新创建 CPU 环境可使用 `requirements-pd-local.txt`。

### 早期单卡版本的实测记录

- 三阶段 14 项测试通过，覆盖模型权重分工、真实前向、图像限额、同文不同图、多图顺序与重复、
  纯文本绕过视觉、并发归属、预处理拒绝后的恢复、进程故障、消费者取消、SSE 与统一 CLI 启动。
- 原文本 PD 14 项、HTTP 5 项、流式评测 11 项测试通过。
- RTX 4060 Laptop / BF16 / 真实 SmolVLM：纯文本、红图、蓝图、红蓝、蓝红、红红，
  6 组输入各生成 8 个 Token，全部与对应单体参考完全一致。
- 真实权重 HTTP 非流式 Token、SSE 文本与单体参考一致。
- 单图传输视觉特征 73,728 字节，双图 147,456 字节；不同图与换序后的特征摘要不同。
- 红图输出片段包含 `red`，蓝图输出片段包含 `blue`。这是冒烟验证，不是视觉能力评测。

原始结果位于：

- `metrics/vpd-smolvlm-cuda-20261031.json`
- `metrics/vpd-regression-20261031/report.json`
- `metrics/vpd-final-regression-20261031/report.json`
- `metrics/vpd-cpu-regression-20261031/report.json`

## 边界

这是可运行的本机三阶段服务，当前后端是 Transformers + Idefics3/SmolVLM。
本页的单卡/Gloo 配置让三个进程共享一张 GPU；特征和 KV 经 CPU/Gloo 交接。
当前另有独立设备映射和 NCCL 路径，见 [架构设计](architecture.md) 与 [三卡验收](vpd_multigpu.md)。
真实四卡功能验收与固定拓扑并发基准已经完成；NCCL 使用 SHM，不宣称 CUDA IPC、零拷贝或等预算扩容收益。

当前支持图像特征插入语言序列的架构；不包括 Llama Vision 的 Cross-Attention 状态适配。
原 ModServe 的 11B/26B 多卡 vLLM 路径保留，尚未迁移到这个三阶段后端。
支持有界并发路由，单 worker 内顺序执行；路由提前预留 D，单 D 时只有一个活动路由。
没有 Continuous Batching、动态扩缩容或跨机调度。
字节上限只约束逻辑 payload，不代表模型、激活、CUDA context 和临时拷贝的总显存上限。

本页单卡用法用于功能验证，当前四卡性能按 [报告](validation/four-gpu-20261101.md) 的条件单独引用。
实现修正与回归范围见 [实现记录](validation/implementation-notes.md)。
