# 项目目录说明

项目有两条运行路径：`modserve` 是视觉编码与文本生成分离的原版主线；`learning` 是无需模型和 GPU 的 CPU/Gloo 学习模式。`monolith` 提供完整模型一起运行的对照基线。

## 根目录

| 目录 | 用途 |
|---|---|
| `src/lmm_serving/` | 项目 Python 源码，见下表 |
| `configs/` | 实例部署拓扑；`modserve/`、`monolith/` 是多 GPU YAML，`single_gpu_mock/` 是学习模式 JSON |
| `client/` | 按 Azure 请求 trace 的时间顺序向服务发送请求，产生实验负载 |
| `llm-backend/` | 固定版本 vLLM 的构建脚本和源码补丁；实现外部视觉特征接入与模型内部测量 |
| `profiles/` | 各模型的编码、后端容量参考数据；自动扩缩容会读取。当前为上游参考值，不能当成本机实测 |
| `traces/` | 小型请求回放样本；`experiment_traces/` 保存调整请求速率后的输入 |
| `tests/` | 拓扑、调度、扩缩容、profiler、生产传输/背压和学习模式测试 |
| `utils/` | 环境安装、远程部署、图片准备、trace 下载/转换、指标合并、实验进程清理等脚本 |
| `docs/` | 配置、部署、实验、可靠传输改造、学习模式及验证记录 |
| `.venv/` | 当前机器的 Python 环境，包含 CPU PyTorch 和测试依赖；不是源码，不提交 Git |
| `.git/` | Git 提交历史、分支及工作区索引，保留以管理和恢复项目 |

当前执行环境还提供空的只读 `.agents/`、`.codex/`、`.aws/` 目录；它们不是业务代码，也不属于项目跟踪文件。

## 源码子目录

| `src/lmm_serving/` 下的目录 | 用途 |
|---|---|
| `modserve/` | 分离式多模态服务主线；`pipeline.py` 负责编码前预留、并发编码、实际大小校正及清理；`components/` 运行 Cluster、Image、Text 和 NCCL 接收服务 |
| `monolith/` | 单体对照架构；`components/` 启动完整视觉语言模型实例及前端路由 |
| `common/` | 共用基础模块：路由、指标、视觉 token 计算、自动扩缩容、Text 代理；新增的 `resources.py`、`features.py`、`admission.py`、`tensor_transfer.py` 实现生产预算和传输协议 |
| `launcher/` | 命令行、拓扑校验、GPU/rank 分配、本机/SSH 进程启动和退出 |
| `config/` | 全局模型、目录、端口、显存预算等运行参数，以及 Python / shell 加载器；与根目录的部署拓扑 `configs/` 分开 |
| `profiler/` | 测量编码和后端耗时、拟合容量模型，生成 `profiles/` 中的容量文件 |
| `learning/` | 独立 CPU/Gloo 学习链路：模拟编码/生成，真实进程间张量传输 |

## 根目录文件

- `README.md`：项目入口、安装和运行命令。
- `pyproject.toml`：Python 包的元数据、源码布局和打包配置。
- `requirements.txt`：原版服务的 Python 依赖；vLLM 由补丁构建脚本另行安装。
- `requirements-learning.txt`：CPU 学习模式的轻量依赖。
- `pixi.toml` / `pixi.lock`：原版 CUDA/Python 环境、任务定义和依赖锁定。
- `SINGLE_GPU_PLAN.md`：单卡适配计划及阶段状态，真实单卡推理尚未完成。
- `.gitignore`：排除虚拟环境、模型下载、生成结果和缓存。

## 可以重新生成的目录

以下内容不是需要保留的业务源码，清理后相应命令会重新创建：

- `src/**/__pycache__/`、`tests/__pycache__/`：Python 字节码缓存。
- `metrics/`、`logs/`：实验/演示输出；清理前按需保留自己的实验记录。
- `llm-backend/vllm/`：构建脚本下载并应用补丁的 vLLM 源码。若它已经被 editable 安装，删除会破坏该安装，不能当作普通缓存直接删。

本次清理删除了空的 `online_serving/`、Python 缓存及仅用于补丁验证、尚未安装的 vLLM 下载副本，保留了 `llm-backend/build_vllm.sh` 和 `vllm-lmm-serving.patch`。

论文消融/评分代码 `src/lmm_serving/experiments/`、绘图目录 `notebooks/` 和静态演示目录 `examples/` 已删除。学习模式的 `--demo` 和测试改为在临时目录生成图片，结束后自动清理；常驻学习服务通过 `--image-dir` 指定自己的图片目录（默认为仓库根目录下的 `images/`）。保留请求回放、性能采样及测试，便于验证服务行为。

阅读原版主线可按 `launcher/ → modserve/pipeline.py → modserve/components/ → common/ → vLLM 补丁` 的顺序进行。
