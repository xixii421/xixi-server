# 单卡学习模式进度

## 2026-10-15：CPU/Gloo 模拟链路

本轮按用户授权落地通信学习模式，暂不实现真实模型推理。原始多卡路径保留；新增 `python -m lmm_serving.launcher learn` 和独立 `lmm_serving.learning` 入口。

### 环境

- WSL2 Ubuntu 22.04，内核 `6.18.33.2-microsoft-standard-WSL2`。
- Python `3.10.12`，CPU PyTorch `2.5.1+cpu`，Gloo 可用。
- Flask `3.1.0`、Pillow `11.1.0`，项目内独立 `.venv`。
- 环境探测时磁盘可用约 859 GiB。
- 此执行环境的 NVML 返回 `GPU access blocked by the operating system`；没有测得显卡、显存、驱动和 CUDA 版本，也没有测 GPU 性能。
- 默认沙箱不允许创建监听 socket；真实 HTTP/Gloo 演示与集成测试在获准的本机回环通信执行环境中运行。

### 已完成

- Cluster 主进程 + Image/Text 两个 `spawn` 子进程；两个 Gloo rank。
- 真实图片字节生成确定性模拟特征，通过 Gloo 传输和校验；无模型下载。
- 复用原版调度器实现，加入请求准入和统一图片流水线排序。
- 唯一传输 ID、有界缓存、deadline、取消标记、幂等释放、TTL 回收。
- 明确的 mock 输出和 JSONL 阶段指标，纯文本绕过 Image。
- 本地双图片样例、配置、运行指南及自动化测试。

### 已执行验证

```bash
PYTHONPATH=src .venv/bin/python -m lmm_serving.learning --demo
PYTHONPATH=src .venv/bin/python tests/test_learning.py
PYTHONPATH=src .venv/bin/python tests/test_launcher.py
PYTHONPATH=src .venv/bin/python tests/test_scheduling.py
PYTHONPATH=src .venv/bin/python tests/test_autoscaler.py
PYTHONPATH=src .venv/bin/python tests/test_profiler.py
```

首次端到端 demo 的纯文本和单图均返回 200，单图特征在 Text 进程校验通过，演示结束后子进程退出。耗时仅为 CPU 教学链路的一次运行，不作为性能结果。

对外入口 `python -m lmm_serving.launcher learn --demo --metrics-dir /tmp/modserve-launcher-demo` 也已实跑通过：两条请求均为 200、`cleanup_errors=[]`，进程退出码为 0。

原有测试结果：launcher 39、scheduling 9、autoscaler 20、profiler 2 项检查通过，共 70 项。

学习模式最终 17 项测试全部通过，实测测试耗时 31.2 秒，包含真实 CPU/Gloo 端到端通信、活动重复 ID、Gloo 接收超时和故障回收。原版 70 项检查也全部通过。

### 限制与下一步

真实编码和语言模型生成未实现，不对应 M2/M3/M4 完成。图片流水线按请求串行；没有实现多池实例扩缩容、TP、CUDA IPC 或 NCCL 性能验证。后续可在保持协议和测试的前提下增加更多 rank / 池实例或接入独立真实模型后端。
