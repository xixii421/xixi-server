"""Audit raw experiment records, then derive a report and honest resume wording."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

MODES=("monolithic","vpd_uncached","vpd_cached")
SCENARIOS=("hot_reuse","cold_unique","mixed_reuse")
LABELS={"monolithic":"单体模型","vpd_uncached":"三阶段／无缓存","vpd_cached":"三阶段／视觉缓存"}
SCENE_LABELS={"hot_reuse":"重复图片","cold_unique":"全新图片","mixed_reuse":"混合复用"}


def audit(folder):
    report=json.loads((folder/"report.json").read_text())
    records=[json.loads(line) for line in (folder/"requests.jsonl").read_text().splitlines()]
    config=report["configuration"]
    if not report["passed"] or config["rounds"]<3:
        raise ValueError("requires a passed experiment with at least three rounds")
    if "loopback_transport" not in config:
        raise ValueError("use the corrected direct-loopback experiment, not preliminary proxy-affected data")
    expected=9*config["rounds"]*config["requests_per_scenario_per_round"]
    if len(records)!=expected:
        raise ValueError("raw request count does not match experiment configuration")
    groups=defaultdict(list)
    references={}
    for item in records:
        record=item["record"]
        if record.get("connection_route")!="direct_loopback":
            raise ValueError("loopback experiment used a proxy or unverified route")
        if record["status"]!="ok" or record["completion_tokens"]!=config["new_tokens"]:
            raise ValueError("failure or unexpected generation count in raw records")
        if record["completion_tokens"]!=len(record["backend_metrics"]["token_ids"]):
            raise ValueError("usage does not match actual generated tokens")
        groups[item["scenario"],item["mode"]].append(item)
        if item["mode"]=="monolithic":
            key=item["scenario"],item["index"]
            tokens=record["backend_metrics"]["token_ids"]
            if key in references and references[key]!=tokens:
                raise ValueError("unstable monolithic reference")
            references[key]=tokens
    for item in records:
        if item["record"]["backend_metrics"]["token_ids"]!=references[item["scenario"],item["index"]]:
            raise ValueError("output mismatch with the full-model reference")
    for (scenario,mode),items in groups.items():
        measured=statistics.median(i["record"]["ttft_ms"] for i in items)
        if not math.isclose(measured,report["summary"][scenario][mode]["client_ttft_p50_ms"],rel_tol=1e-9):
            raise ValueError("reported median differs from raw samples")
        if len(items)!=config["rounds"]*config["requests_per_scenario_per_round"]:
            raise ValueError("unequal samples across layouts")
    return report,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--docs",type=Path,required=True)
    args=parser.parse_args()
    report,records=audit(args.results)
    config=report["configuration"]
    args.docs.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scenario in SCENARIOS:
        for mode in MODES:
            row=report["summary"][scenario][mode]
            rows.append(f"| {SCENE_LABELS[scenario]} | {LABELS[mode]} | {row['requests']} | "
                f"{row['client_ttft_p50_ms']:.1f} | {row['client_ttft_p95_ms']:.1f} | "
                f"{row['client_e2e_p50_ms']:.1f} | {row['successful_output_tokens_per_s']:.1f} | "
                f"{row['image_hit_rate']*100:.0f}% |")
    hot=report["summary"]["hot_reuse"]
    before,after=hot["vpd_uncached"],hot["vpd_cached"]
    hot_gain=report["comparisons"]["hot_reuse"]["client_ttft_median_reduction_pct"]
    mixed=report["comparisons"]["mixed_reuse"]
    cold=report["comparisons"]["cold_unique"]
    rounds=report["comparisons"]["hot_reuse"]["paired_round_ttft_reduction_pct"]
    text=f"""# 三阶段视觉特征复用：正式实验与边界

原始实验目录：`{args.results}`。本文由原始 JSONL 校验后生成，没有过滤异常慢请求。

## 方法

- RTX 4060 Laptop 8GB；SmolVLM-256M BF16；Transformers/PyTorch eager。
- 单体、三阶段无缓存、三阶段缓存；相同权重、图像预处理、greedy、正常 KV Cache。
- 图片为可复现的 64×64 随机纹理，模型预处理长边 512、关闭切图；每请求两图、{config['new_tokens']} 个输出 Token。
- 闭环并发 1；每配置每场景 {config['rounds']} 轮，每轮 {config['requests_per_scenario_per_round']} 请求。
- 每场景先执行相同的 5 个预热请求，覆盖全未命中、部分命中和命中；预热不计入结果。
- 每轮轮换三种配置的先后顺序；模型常驻，加载与启动不计时。
- 图片 fixture 在计时前构建；吞吐使用完整 cohort 的实际墙钟时间。
- TTFT 从客户端请求开始到首个非空 SSE 内容；不是模型内部计时。
- SSE 分块不等于 Token，因此不从分块间隔推导客户端 TBT。
- GPU 未锁频，保留长尾与每轮结果；合成复用负载不等同于生产流量。
- 环回 HTTP 显式直连。先前受环境代理影响的实验保留在 `vpd-benchmark-final-20261031`，不用于本文结论；没有从该数据中挑选或剔除请求。

## 结果

| 场景 | 配置 | 请求数 | TTFT P50 ms | TTFT P95 ms | E2E P50 ms | 输出 token/s | 图片命中率 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

重复图片：三阶段缓存相对同一三阶段无缓存，客户端 TTFT P50 从 **{before['client_ttft_p50_ms']:.1f} ms**
降至 **{after['client_ttft_p50_ms']:.1f} ms**，降低 **{hot_gain:.1f}%**。
三轮各自的中位数改善分别为 `{[round(v,1) for v in rounds]}`%。
缓存命中率 {after['image_hit_rate']*100:.0f}%，实测视觉编码图数从 {before['encoded_images']} 降至 {after['encoded_images']}。

混合复用：TTFT P50 相对变化为 {-mixed['client_ttft_median_reduction_pct']:+.1f}%（负值为延迟降低），
吞吐变化为 {mixed['output_throughput_increase_pct']:+.1f}%。
全新图片：TTFT P50 相对变化为 {-cold['client_ttft_median_reduction_pct']:+.1f}%，
吞吐变化为 {cold['output_throughput_increase_pct']:+.1f}%。正的延迟变化表示退化。

## 正确性与归因

原始记录共 {len(records)} 条，全部成功；其中 {len(records)*2//3} 条三阶段请求逐 Token 对齐单体参考。
无缓存与有缓存采用同样的分阶段、Gloo、KV、采样和流式协议，唯一消融开关是视觉特征缓存。
命中只跳过视觉编码，图像预处理、P、D 和传输继续执行；不复用其他请求的文本 KV。
容量为 16MiB CPU resident payload，按字节 LRU 淘汰；不代表全部进程和显存占用。

## 可说明与不可外推的结论

这项结果证明重复图片负载下视觉特征复用有效；不证明 PD 自身或跨卡通信带来同样收益。
单体模型行保留了单卡部署的对照，没有用无 KV、反复重算的弱基线。
冷图与混合负载同时报告，不能只展示命中收益。P95 为小样本描述，不作为生产 SLO 保证。
这不是 11B/26B 原 vLLM 后端、分布式集群、动态批处理或成本节省的验收。

复现：

```bash
PYTHONPATH=src .venv-pd-cuda/bin/python utils/benchmark_vpd.py \\
  --model /home/xixii/huggingface/SmolVLM-256M-Instruct --device cuda:0 \\
  --requests 20 --rounds 3 --new-tokens 8 --output-dir metrics/new-vpd-benchmark
```
"""
    (args.docs/"vpd_benchmark.md").write_text(text,encoding="utf-8")
    stable=hot_gain>0 and all(value>0 for value in rounds)
    performance=(f"在 RTX 4060 Laptop 8GB、SmolVLM-256M BF16、并发 1、双图/8 输出 Token 的重复图片合成负载下，"
        f"客户端 TTFT 中位数由 {before['client_ttft_p50_ms']:.1f} 降至 {after['client_ttft_p50_ms']:.1f} ms"
        f"（降低 {hot_gain:.1f}%，视觉缓存命中率 {after['image_hit_rate']*100:.0f}%）；"
        f"每配置 3 轮、每轮 20 请求，轮换配置顺序，独立报告冷图与混合负载。"
        if stable else "完成单体、三阶段无缓存与缓存模式的三轮对照实验，记录命中、未命中和混合负载的收益与开销；未将不稳定指标写成确定加速结论。")
    resume=f"""# 项目 2：可替换简历文案

**xixi-server｜多模态三阶段推理服务与视觉特征复用优化**　2026.06—至今

基于 ModServe 开源仓库的工程扩展；本机后端采用 PyTorch / Transformers / SmolVLM / Gloo / HTTP-SSE。

- **三阶段服务：**将视觉编码与投影、Prefill、Decode 拆为独立进程，按阶段裁剪模型权重；支持单图、多图和纯文本绕过视觉，通过 Gloo 交接 BF16 视觉特征与 KV Cache。
- **缓存与资源管理：**实现内容寻址、模型/预处理隔离的视觉特征复用与字节级 LRU，处理部分命中、多图重排和重复图片；以请求 ID、序号、摘要校验及接收确认管理在途状态，支持容量拒绝和故障后的进程组回收。
- **性能验证：**{performance}
- **正确性与工程化：**构建真实模型、三进程传输、HTTP/SSE、取消与故障回归；正式实验中 {len(records)*2//3} 条三阶段请求逐 Token 对齐持久单体基线，保存逐请求数据、运行配置及复现脚本。

## 一页简历空间紧张时

- 在 ModServe 派生项目中实现视觉/P/D 三进程服务，按阶段裁剪真实多模态模型权重，支持单图、多图与流式生成；以 UUID、上下文摘要和 ACK 管理特征/KV 交接及异常回收。
- 实现内容寻址、16MiB 字节级 LRU 的视觉特征复用，支持部分命中与多图重排；在 4060 8GB / SmolVLM-256M BF16 / 并发 1 / 双图 8 输出 Token 的重复图片合成负载下，100% 命中时 TTFT 中位数由 {before['client_ttft_p50_ms']:.1f} 降至 {after['client_ttft_p50_ms']:.1f} ms（降低 {hot_gain:.1f}%）。
- 进行单体/三阶段无缓存/有缓存的三轮对照，每配置每场景 60 请求；{len(records)*2//3} 条三阶段请求逐 Token 对齐单体，保留冷图代价、原始日志与复现脚本。

详细测试数量与冷热图对照放仓库 README。不要删除指标的实验条件和“重复图片合成负载”。

## 事实边界

- ModServe 的原始架构和论文结果归原作者；这里写的是派生仓库的新后端、缓存与验证。
- 当前是单卡主机内存/Gloo 回环，不写成跨 GPU NCCL、CUDA IPC 或零拷贝优化。
- 当前模型采用视觉特征插入语言序列，不宣称已适配 Cross-Attention 状态交接。
- 不写超过 vLLM、生产 P99 SLO、动态批处理或服务成本降低。
- “2026.06”沿用原简历项目起始月；正式投递时请与实际开始时间核对。
"""
    (args.docs/"resume_project2.md").write_text(resume,encoding="utf-8")
    print(json.dumps({"audited_requests":len(records),"resume_metric_stable":stable,
                      "hot_ttft_reduction_pct":hot_gain,"output_docs":str(args.docs)},indent=2))


if __name__=="__main__":
    main()
