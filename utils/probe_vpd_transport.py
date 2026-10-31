"""Validate and time V->P / P->D payload handoffs without loading model weights."""
import argparse
from datetime import timedelta
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import subprocess
import tempfile
import time
import traceback

import torch
import torch.distributed as dist

from lmm_serving.vpd_local import VModelSpec
from lmm_serving.vpd_options import add_runtime_arguments,placement_kwargs
from lmm_serving.vpd_transport import TensorChannel


def hardware_snapshot(devices):
    def command(args):
        try:
            result=subprocess.run(args,capture_output=True,text=True,timeout=10)
            return {"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
        except (OSError,subprocess.SubprocessError) as error:
            return {"error":str(error)}
    cuda=[d for d in devices if d.startswith("cuda:")]
    return {"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
            "nccl":list(torch.cuda.nccl.version()) if dist.is_nccl_available() else None,
            "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices":[{"logical":d,"name":torch.cuda.get_device_name(d)} for d in dict.fromkeys(cuda)],
            "peer_access":{f"{a}->{b}":torch.cuda.can_device_access_peer(int(a.split(":")[1]),int(b.split(":")[1]))
                           for a in cuda for b in cuda if a!=b},
            "nvidia_smi":command(["nvidia-smi"]),"topology":command(["nvidia-smi","topo","-m"])}


def probe_worker(rank,spec,store,events,sizes,repeats,warmups,timeout):
    channel=None
    result={"rank":rank,"passed":False}
    try:
        spec=spec.for_rank(rank)
        if spec.device.startswith("cuda:"):
            torch.cuda.set_device(spec.device)
        dist.init_process_group("gloo",init_method="file://"+store,rank=rank,world_size=spec.worker_count,
                                timeout=timedelta(seconds=timeout))
        channel=TensorChannel.start(spec,timeout)
        results=[]
        sequence=0
        # 验证相邻阶段的所有候选连接，包括未被一次轮询请求选到的副本组合。
        edges=[(name,source,destination) for stage,name in ((0,"V->P"),(1,"P->D"))
               for source in spec.ranks_for(stage) for destination in spec.ranks_for(stage+1)]
        for dtype in (torch.float32,torch.bfloat16):
            element_bytes=torch.empty((),dtype=dtype).element_size()
            for size in sizes:
                for edge,source,destination in edges:
                    for iteration in range(warmups+repeats):
                        sequence+=1
                        expected=((torch.arange(size//element_bytes,device=spec.device)%97)+sequence%13).to(dtype)
                        if spec.device.startswith("cuda:"):
                            torch.cuda.synchronize(spec.device)
                        dist.barrier()
                        if rank==destination:
                            received=channel.stage(channel.receive(expected.shape,dtype,source))
                            if not torch.equal(received,expected):
                                raise ValueError(f"rank {rank}: payload corruption at sequence {sequence}")
                            dist.send(torch.tensor([sequence],dtype=torch.int64),dst=source)
                        elif rank==source:
                            started=time.perf_counter()
                            payload=channel.prepare(expected)
                            channel.send(payload,destination)
                            ack=torch.zeros(1,dtype=torch.int64)
                            dist.recv(ack,src=destination)
                            elapsed_ms=(time.perf_counter()-started)*1000
                            if ack.item()!=sequence:
                                raise ValueError("probe ACK mismatch")
                            if iteration>=warmups:
                                results.append({"edge":edge,"source_rank":source,"destination_rank":destination,
                                                "dtype":str(dtype),"bytes":expected.numel()*element_bytes,
                                                "iteration":iteration-warmups,"handoff_ms":elapsed_ms})
                        dist.barrier()  # 其余副本也参与屏障，防止不同连接间操作交错。
        result.update(passed=True,samples=results)
    except BaseException:
        result["error"]=traceback.format_exc()
    finally:
        try:
            if dist.is_initialized():
                try:
                    if channel is not None:
                        channel.close()
                finally:
                    dist.destroy_process_group()
        except BaseException:
            result.update(passed=False,error=result.get("error","")+traceback.format_exc())
    # 组销毁也属于验收；清理失败或卡住时不能提前发送成功消息。
    events.put(result)


def wait_for_workers(processes,timeout=5):
    """成功消息不代替进程退出：确认 worker 正常结束后才能判定探针通过。"""
    for process in processes:
        process.join(timeout=timeout)
    exitcodes=[process.exitcode for process in processes]
    if any(code != 0 for code in exitcodes):
        raise RuntimeError(f"transport probe workers did not exit cleanly: {exitcodes}")
    return exitcodes


def run_probe(spec,*,sizes,repeats,warmups,timeout):
    context=mp.get_context("spawn")
    events=context.Queue()
    processes=[]
    results=[]
    with tempfile.TemporaryDirectory(prefix="vpd-probe-") as directory:
        try:
            for rank in range(spec.worker_count):
                process=context.Process(target=probe_worker,args=(rank,spec,str(Path(directory)/"store"),
                    events,sizes,repeats,warmups,timeout))
                processes.append(process)
                process.start()
            deadline=time.monotonic()+timeout
            while len(results)<spec.worker_count:
                if time.monotonic()>=deadline:
                    raise TimeoutError("transport probe deadline exceeded")
                if any(p.exitcode not in (None,0) for p in processes):
                    raise RuntimeError("transport probe worker crashed")
                try:
                    result=events.get(timeout=.1)
                except queue.Empty:
                    continue
                results.append(result)
                if not result["passed"]:
                    raise RuntimeError(result["error"])
            exitcodes=wait_for_workers(processes)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join()
            events.close()
            events.cancel_join_thread()
    samples=[sample for result in results for sample in result["samples"]]
    groups={}
    for sample in samples:
        key=(sample["edge"],sample["source_rank"],sample["destination_rank"],sample["dtype"],sample["bytes"])
        groups.setdefault(key,[]).append(sample["handoff_ms"])
    summary=[{"edge":edge,"source_rank":source,"destination_rank":destination,
              "dtype":dtype,"bytes":size,"samples":len(values),
              "handoff_p50_ms":statistics.median(values),
              "handoff_p95_ms":sorted(values)[max(0,math.ceil(.95*len(values))-1)]}
             for (edge,source,destination,dtype,size),values in groups.items()]
    return {"passed":True,"samples":samples,"summary":summary,"worker_exitcodes":exitcodes,
            "worker_results":[{"rank":r["rank"],"passed":r["passed"],"exitcode":exitcodes[r["rank"]]}
                              for r in sorted(results,key=lambda result:result["rank"])]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--sizes-kib",nargs="+",type=int,default=[72,144,1024,8192])
    parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--warmups",type=int,default=3)
    parser.add_argument("--timeout",type=float,default=180)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if (any(n<=0 or n>65536 for n in args.sizes_kib) or args.repeats<1 or args.warmups<0
            or not math.isfinite(args.timeout) or args.timeout<=0):
        parser.error("invalid probe size, count or timeout")
    spec=VModelSpec(device=args.device,transport=args.transport,**placement_kwargs(args))
    report={"passed":False,"transport":spec.transport,"stage_devices":list(spec.devices),
            "stage_pools":[list(pool) for pool in spec.pools],"worker_count":spec.worker_count,
            "scope":"validated single-host handoff: sender staging + payload + receiver staging/equality + Gloo ACK; not raw link bandwidth"}
    try:
        spec.check_hardware()
        report["hardware"]=hardware_snapshot(spec.devices)
        report.update(run_probe(spec,sizes=[n*1024 for n in args.sizes_kib],repeats=args.repeats,
                                warmups=args.warmups,timeout=args.timeout))
    except Exception as error:
        report["error"]=str(error)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"output":str(args.output),"error":report.get("error")}))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
