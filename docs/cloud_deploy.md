# xixi-server 三卡租用与部署指南

本次目标：在一台服务器的三张独立 GPU 上完成 V/P/D 正确性验收，再测通信与缓存收益。
本文只准备部署材料，不代表已经租卡或完成三卡验收。

## 1. 平台怎么选

| 平台 | 本次建议 | 入口 |
|---|---|---|
| AutoDL | 首选，按下面的流程操作；官方有中文 SSH、上传、VSCode 和计费教程 | https://www.autodl.com/ |
| 恒源云 | AutoDL 没有合适的同机多卡库存时比较配置和总价，步骤以其控制台为准 | https://www.gpushare.com/docs/getting-started/quickstart/ |
| Runpod | 海外 GPU Pods 备选；选可 SSH 的 Pod，按实际 GPU 数与存储看总价 | https://docs.runpod.io/pods/choose-a-pod |

这里需要能运行多进程、SSH 和 NCCL 的 GPU 实例，不使用按 API 调用计费的 Serverless 产品。
平台推荐是针对本次小规模实验的选择，不代表对实时库存、P2P 拓扑或最低价的保证。

## 2. AutoDL 下单配置

登录后进入容器实例的租用/创建入口。页面名称可能调整，按这些字段选择：

- 计费：按量计费，先预留约 100 元，不急着买包月。
- GPU：优先 RTX 3090 24GB，数量 3；4090 24GB 也可。
- 必须在同一个实例中选择多卡。若只有四卡规格，就租四卡、程序使用前三张，费用按四张算。
- CPU 建议至少 8 vCPU，主机内存建议 32GB，磁盘建议 50GB。
- 镜像：优先 Python 3.10 / PyTorch 2.6 / CUDA 12.4 的基础镜像。没有完全相同的镜像时，选有 Python 3.10、驱动兼容 CUDA 12.4 的镜像，随后新建本文的环境。
- 看清最终实例总价及磁盘费用。不要把“元/卡时”误当整个三卡实例的价格。

GPU 型号不保证具体 P2P/NVLink 路径，启动后记录拓扑。普通功能验收不要求有 NVLink。
如果准备时使用无卡模式，重新启用三卡取决于当时库存；无卡模式也可能有费用。

## 3. 连接实例

实例运行后，从控制台复制真实的 SSH 指令。在本机 Windows PowerShell 执行：

```powershell
ssh -p PORT root@HOST
```

`PORT` 和 `HOST` 替换为控制台给出的端口、主机名。若控制台用户名不是 root，使用实际用户名。
首次连接核对控制台提供的主机信息；密码在终端中输入时不显示字符，属于正常现象。

连接后的终端运行在租用的 Linux 服务器上；断开 SSH 不等于关闭服务器或停止计费。
也可以在 VSCode 安装 Remote - SSH，通过同一条 SSH 指令连接后打开服务器文件夹。

## 4. 克隆最新代码并上传模型

本机目录 `D:\重要文档\AI Infra\output\gpu-deploy` 中已经准备：

- `xixi-server-deploy.tar.gz`：本轮代码的离线备份；在线部署优先克隆下面的分支。
- `SmolVLM-256M-Instruct.tar`：本机已经下载的预训练模型，约 494MiB，无需服务器重新下载。
- `MODEL_SHA256SUMS`：模型包的摘要，用来检查上传是否完整。

先在**远程 SSH 终端**确认数据盘目录，再创建上传目录。以下按 AutoDL 的 `/root/autodl-tmp` 举例：

```bash
mkdir -p /root/autodl-tmp/vpd-upload
cd /root/autodl-tmp
git clone --branch codex/pd-local --single-branch https://github.com/xixii421/xixi-server.git
```

另外打开一个**本机 Windows PowerShell**，运行以下上传命令。`scp` 的端口参数是大写 `-P`：

```powershell
Set-Location 'D:\重要文档\AI Infra\output\gpu-deploy'
scp -P PORT SmolVLM-256M-Instruct.tar MODEL_SHA256SUMS root@HOST:/root/autodl-tmp/vpd-upload/
```

在**远程终端**校验、解压。使用新目录，避免覆盖之前的实验：

```bash
cd /root/autodl-tmp/vpd-upload
sha256sum -c MODEL_SHA256SUMS
mkdir -p /root/autodl-tmp/models
tar -xf SmolVLM-256M-Instruct.tar -C /root/autodl-tmp/models
```

模型摘要显示 OK 后再继续。代码位于 `/root/autodl-tmp/xixi-server`；不要只克隆默认 main 分支，它尚未合入本轮修改。
若服务器 GitHub 连接不通，可以上传代码离线包、创建一个新代码目录并解压，作为备用方案。

## 5. 创建运行环境

以下全部在**远程终端**运行：

```bash
cd /root/autodl-tmp/xixi-server
python --version
nvidia-smi
```

优先使用 Python 3.10。若镜像带 Conda，可以新建：

```bash
conda create -n vpd python=3.10 -y
conda activate vpd
python -m pip install -r requirements-vpd-cuda.txt
```

如果镜像自带 Python 3.10，也可以用 venv（二选一，不必同时创建）：

```bash
python3.10 -m venv .venv-vpd
source .venv-vpd/bin/activate
python -m pip install -r requirements-vpd-cuda.txt
```

依赖固定为 PyTorch 2.6.0+cu124、Transformers 4.51.3 等。若镜像已经完全匹配，pip 会复用满足版本的依赖。
`nvidia-smi` 顶部的 CUDA Version 表示驱动支持能力，不等于 Python 包所带的 CUDA runtime 版本。

```bash
python -c "import torch; print('torch:', torch.__version__); print('runtime:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('NCCL:', torch.distributed.is_nccl_available())"
nvidia-smi topo -m
```

至少看到 3 张 GPU，并且 NCCL 为 True。看不到三张时先检查实例卡数和 `CUDA_VISIBLE_DEVICES`，不要先跑正式基准。

## 6. 第一次先做完整正确性验收

耗时操作建议在 tmux 中运行，SSH 断开后任务仍能继续。若系统已有 tmux，直接使用；没有时通过镜像包管理器安装。

```bash
tmux new -s vpd
```

进入新会话后确认已激活刚才的 Python 环境，再执行：

```bash
cd /root/autodl-tmp/xixi-server
PYTHONPATH=src python utils/validate_vpd_multigpu.py \
  --model /root/autodl-tmp/models/SmolVLM-256M-Instruct \
  --stage-devices cuda:0 cuda:1 cuda:2 \
  --output-dir metrics/three-gpu-check-001
```

如果租的是四张卡，这个命令仍只使用前三张。程序自己启动三个 worker，无需另加 torchrun。

验收会完成微型模型三卡测试、Gloo/NCCL 双通道探针、预训练模型对照及 HTTP/SSE 检查，保存拓扑、源码/权重摘要和逐步骤日志。

```bash
python -m json.tool metrics/three-gpu-check-001/report.json
```

最终 `passed` 应为 true。失败时查看对应步骤的 `.log`，不要据失败或跳过用例填写简历指标。
输出目录必须不存在，重复运行用 `002` 等新名字。

tmux 暂时退出：依次按 `Ctrl+B`、松开后按 `D`；重新进入：`tmux attach -t vpd`。

## 7. 验收通过后再做正式实验

```bash
PYTHONPATH=src python utils/validate_vpd_multigpu.py \
  --model /root/autodl-tmp/models/SmolVLM-256M-Instruct \
  --stage-devices cuda:0 cuda:1 cuda:2 \
  --benchmark --rounds 3 --requests 20 \
  --output-dir metrics/three-gpu-benchmark-001
```

这会先验收，再测 Gloo 和 NCCL 下的单体/无缓存/有缓存配置，覆盖热图、冷图和混合负载。
不要同时启动另一个模型服务抢占 GPU。性能对比的指标口径与资源差异见 `docs/resume_measurements.md`。

## 8. 想手动体验 HTTP 服务

完成实验后，单独启动以下命令：

```bash
PYTHONPATH=src python -m lmm_serving.launcher vpd \
  --model /root/autodl-tmp/models/SmolVLM-256M-Instruct \
  --transport nccl --stage-devices cuda:0 cuda:1 cuda:2 --port 8099
```

另一个远程终端访问：

```bash
curl http://127.0.0.1:8099/health
curl -N http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolVLM-256M-Instruct","messages":[{"role":"user","content":"Explain KV cache briefly."}],"max_tokens":8,"stream":true}'
```

要在本机访问，用本机 PowerShell 新开 SSH 隧道：

```powershell
ssh -p PORT -N -L 8099:127.0.0.1:8099 root@HOST
```

隧道保持运行时，本机 `http://127.0.0.1:8099/health` 对应远程服务。服务仅绑定环回地址，不必额外开放公网业务端口。

## 9. 下载结果并停止计费

**本机 PowerShell**：

```powershell
Set-Location 'D:\重要文档\AI Infra\output\gpu-deploy'
scp -P PORT -r root@HOST:/root/autodl-tmp/xixi-server/metrics ./cloud-metrics
```

确认日志和 JSON/JSONL 已下载后，在平台控制台关闭 GPU 实例并确认状态。
退出 Python、关闭 VSCode、断开 SSH 或退出 tmux 都不等于平台关机。
AutoDL 按开机到关机计费；额外存储的计费/保留规则另看控制台。准备彻底删除实例前先备份结果。

## 官方参考

- [AutoDL 创建实例](https://www.autodl.com/docs/quick_start/)
- [AutoDL SSH](https://www.autodl.com/docs/ssh/)
- [AutoDL 上传/下载命令](https://www.autodl.com/docs/scp/)
- [AutoDL 计费](https://api.autodl.com/docs/price/)
- [AutoDL 无卡模式与升降配置](https://backup.autodl.com/docs/save_money/)
- [恒源云快速开始](https://www.gpushare.com/docs/getting-started/quickstart/)
- [Runpod GPU Pods](https://www.runpod.io/product/cloud-gpus)

平台入口、库存与价格可能调整。本文给的是项目部署路径，实际购买以当前控制台显示为准。
