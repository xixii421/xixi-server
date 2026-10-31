"""Paired, warmed HTTP ablation: full model / VPD / VPD with bounded feature reuse.

Synthetic image-reuse workloads; closed-loop concurrency one. This does not
claim production-trace coverage, cross-GPU scaling or causal gains from PD alone.
"""
import argparse
import base64
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time

import numpy as np
from PIL import Image
import torch

from lmm_serving.monolithic_local import MonolithicVLM
from lmm_serving.vpd_local import LocalVPD,VModelSpec
from lmm_serving.vpd_service import VPDHTTPServer
from lmm_serving.stream_benchmark import measure
from lmm_serving.vpd_options import add_runtime_arguments,placement_kwargs

MODES=("monolithic","vpd_uncached","vpd_cached")
SCENARIOS=("hot_reuse","cold_unique","mixed_reuse")
QUESTIONS=("Describe the colors in these images.","What patterns are visible here?",
           "Give a short description of the images.","Summarize the visual content briefly.")


def image_url(seed):
    pixels=np.random.default_rng(seed).integers(0,256,(64,64,3),dtype=np.uint8)
    buffer=io.BytesIO()
    Image.fromarray(pixels).save(buffer,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()


def payload(model,seeds,index,new_tokens):
    parts=[{"type":"image_url","image_url":{"url":image_url(seed)}} for seed in seeds]
    parts.append({"type":"text","text":QUESTIONS[index%len(QUESTIONS)]})
    return {"model":model,"messages":[{"role":"user","content":parts}],"max_tokens":new_tokens,
            "temperature":0,"stream_options":{"include_backend_metrics":True}}


def seeds_for(scenario,index):
    if scenario=="hot_reuse":
        return [0,1] if index%2==0 else [1,0]
    if scenario=="cold_unique":
        return [1000+2*index,1001+2*index]
    return [0,2000+index] if index%5==4 else [0,1]


def percentile(values,p):
    values=sorted(values)
    return values[max(0,math.ceil(p*len(values))-1)] if values else None


def aggregate(records):
    successful=[r for r in records if r["record"]["status"]=="ok"]
    measures=[r["record"] for r in successful]
    ttft=[r["ttft_ms"] for r in measures]
    e2e=[r["e2e_ms"] for r in measures]
    backend=[r["backend_metrics"] for r in measures]
    hits=sum(r.get("cache_hits",0) for r in backend)
    images=sum(r["image_count"] for r in backend)
    cohorts={r["cohort_id"]:r["cohort_elapsed_s"] for r in records if "cohort_id" in r}
    elapsed=sum(cohorts.values()) if cohorts else sum(e2e)/1000
    return {"requests":len(records),"succeeded":len(successful),
            "client_ttft_p50_ms":statistics.median(ttft) if ttft else None,
            "client_ttft_p95_ms":percentile(ttft,.95),
            "client_ttft_iqr_ms":[percentile(ttft,.25),percentile(ttft,.75)],
            "client_e2e_p50_ms":statistics.median(e2e) if e2e else None,
            "successful_output_tokens_per_s":sum(r["completion_tokens"] for r in measures)/elapsed if elapsed else None,
            "vision_stage_p50_ms":statistics.median([r["vision_ms"] for r in backend]) if backend and all("vision_ms" in r for r in backend) else None,
            "image_hit_rate":hits/images if images else None,"cache_hits":hits,"images":images,
            "encoded_images":sum(r.get("vision_images_encoded",0) for r in backend),
            "max_cache_resident_bytes":max([r.get("cache_resident_bytes",0) for r in backend],default=0)}


def environment():
    command=["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm",
             "--format=csv,noheader"]
    try:
        return subprocess.run(command,capture_output=True,text=True,timeout=5).stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return "unavailable"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True,help="new directory; no overwrite")
    parser.add_argument("--requests",type=int,default=20,help="per scenario/mode/round")
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--new-tokens",type=int,default=8)
    add_runtime_arguments(parser,default_device="cuda:0")
    args=parser.parse_args()
    if args.requests<1 or args.rounds<1 or not 1<=args.new_tokens<=32:
        parser.error("invalid sample or generation count")
    placement=VModelSpec(device=args.device,transport=args.transport,**placement_kwargs(args))
    placement.check_hardware()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    all_records=[]
    warmups=[]
    blocks=[]
    # Identical warmups in every layout; reserved images prime full misses,
    # partial misses and hits before each scenario without touching cold data.
    with (args.output_dir/"requests.jsonl").open("x",encoding="utf-8") as raw:
        for round_index in range(args.rounds):
            order=MODES[round_index%3:]+MODES[:round_index%3]
            for mode in order:
                spec=VModelSpec(model_path=str(args.model.resolve()),
                               device=placement.devices[0] if mode=="monolithic" else args.device,dtype="bfloat16",
                               transport="gloo" if mode=="monolithic" else args.transport,
                               stage_devices=None if mode=="monolithic" else placement.stage_devices,
                               stage_pools=None if mode=="monolithic" else placement.stage_pools,
                               verify_payload=not args.no_handoff_checksum,
                               vision_cache_bytes=16*1024*1024 if mode=="vpd_cached" else 0)
                runtime=(MonolithicVLM(spec) if mode=="monolithic" else
                         LocalVPD(spec,startup_timeout_s=180,request_timeout_s=60))
                hardware_before=environment()
                with runtime:
                    server=VPDHTTPServer(("127.0.0.1",0),runtime)
                    thread=threading.Thread(target=server.serve_forever,daemon=True)
                    thread.start()
                    endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                    try:
                        for scenario_index,scenario in enumerate(SCENARIOS):
                            reserved=10000+scenario_index*10
                            warm_seeds=([reserved,reserved+1],[0,1],[0,reserved+2],[0,1],[0,reserved+3])
                            for index,seeds in enumerate(warm_seeds):
                                record=measure(endpoint,payload(server.model_name,seeds,index,args.new_tokens),timeout_s=90)
                                if record["status"]!="ok" or record["backend_metrics"] is None:
                                    raise RuntimeError(f"warmup failed: {record}")
                                warmups.append({"round":round_index,"mode":mode,"scenario":scenario,"record":record})
                            batch=[]
                            bodies=[payload(server.model_name,seeds_for(scenario,i),i,args.new_tokens)
                                    for i in range(args.requests)]
                            cohort_started=time.perf_counter()
                            for index in range(args.requests):
                                seeds=seeds_for(scenario,index)
                                record=measure(endpoint,bodies[index],timeout_s=90)
                                item={"round":round_index,"mode":mode,"scenario":scenario,"index":index,
                                      "image_seeds":seeds,"prompt":QUESTIONS[index%len(QUESTIONS)],"record":record}
                                batch.append(item)
                                if record["status"]!="ok" or record["backend_metrics"] is None:
                                    raw.write(json.dumps(item)+"\n")
                                    raw.flush()
                                    raise RuntimeError(f"measured request failed: {record}")
                            cohort_elapsed=time.perf_counter()-cohort_started
                            for item in batch:
                                item["cohort_id"]=f"{round_index}:{mode}:{scenario}"
                                item["cohort_elapsed_s"]=cohort_elapsed
                            all_records.extend(batch)
                            for item in batch:
                                raw.write(json.dumps(item)+"\n")
                            raw.flush()
                            block={"round":round_index,"mode":mode,"scenario":scenario,
                                   "cohort_elapsed_s":cohort_elapsed,"summary":aggregate(batch)}
                            blocks.append(block)
                            print(json.dumps(block),flush=True)
                    finally:
                        server.shutdown()
                        thread.join()
                        server.server_close()
                blocks[-1]["hardware_before_mode"]=hardware_before
    references={}
    for item in all_records:
        if item["mode"]=="monolithic":
            key=(item["scenario"],item["index"])
            tokens=item["record"]["backend_metrics"]["token_ids"]
            if key in references and references[key]!=tokens:
                raise RuntimeError("monolithic reference changed across rounds")
            references[key]=tokens
    mismatches=[{"round":r["round"],"mode":r["mode"],"scenario":r["scenario"],"index":r["index"]}
                for r in all_records if r["record"]["backend_metrics"]["token_ids"]!=references[(r["scenario"],r["index"])]]
    summary={scenario:{mode:aggregate([r for r in all_records if r["scenario"]==scenario and r["mode"]==mode])
                       for mode in MODES} for scenario in SCENARIOS}
    comparisons={}
    for scenario in SCENARIOS:
        off,on=summary[scenario]["vpd_uncached"],summary[scenario]["vpd_cached"]
        paired=[]
        for round_index in range(args.rounds):
            pair={b["mode"]:b["summary"] for b in blocks if b["round"]==round_index and b["scenario"]==scenario}
            paired.append(100*(1-pair["vpd_cached"]["client_ttft_p50_ms"]/pair["vpd_uncached"]["client_ttft_p50_ms"]))
        comparisons[scenario]={"client_ttft_median_reduction_pct":100*(1-on["client_ttft_p50_ms"]/off["client_ttft_p50_ms"]),
            "client_e2e_median_reduction_pct":100*(1-on["client_e2e_p50_ms"]/off["client_e2e_p50_ms"]),
            "output_throughput_increase_pct":100*(on["successful_output_tokens_per_s"]/off["successful_output_tokens_per_s"]-1),
            "paired_round_ttft_reduction_pct":paired}
    report={"passed":not mismatches,"correctness_mismatches":mismatches,"summary":summary,
            "comparisons":comparisons,"blocks":blocks,"warmups":warmups,
            "configuration":{"model":str(args.model.resolve()),"device":args.device,
                "vpd_stage_devices":list(placement.devices),"vpd_transport":args.transport,
                "vpd_stage_pools":[list(pool) for pool in placement.pools],"vpd_worker_count":placement.worker_count,
                "handoff_checksum":not args.no_handoff_checksum,
                "resource_comparison":"monolithic uses one GPU; VPD uses the listed stage devices; no equal-GPU-budget scaling claim",
                "dtype":"bfloat16","rounds":args.rounds,"requests_per_scenario_per_round":args.requests,
                "new_tokens":args.new_tokens,"warmups_per_scenario":len(warm_seeds),
                "concurrency":1,"mode_order":"rotating Latin order by round",
                "load":"synthetic image reuse; two 64x64 seeded noise images per request",
                "preprocessing":"same no-split 512-longest-edge processor in all layouts",
                "clock_control":"not locked; no production-trace or linear-scaling claim",
                "warmup_policy":"same five reserved-image full-miss/partial-miss/hit requests per scenario",
                "throughput_clock":"whole cohort wall time; fixture construction and log writes excluded",
                "loopback_transport":"explicit direct HTTP; environment proxies bypassed for loopback only",
                "torch":torch.__version__},
            "scope":"warmed HTTP/SSE; persistent full-model baseline; identical KV/greedy policy"}
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"comparisons":comparisons},indent=2))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
