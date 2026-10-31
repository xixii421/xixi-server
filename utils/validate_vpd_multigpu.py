"""Device-pool GPU acceptance bundle; writes logs, hashes and JSON results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from lmm_serving.vpd_local import VModelSpec
from lmm_serving.vpd_options import add_placement_arguments,placement_kwargs
from probe_vpd_transport import hardware_snapshot

ROOT=Path(__file__).resolve().parents[1]


def file_sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_step(name,arguments,output_dir,env,timeout):
    started=time.monotonic()
    process=None
    timed_out=False
    with (output_dir/f"{name}.log").open("wb") as log:
        try:
            process=subprocess.Popen([sys.executable,*arguments],cwd=ROOT,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            code=process.returncode
        except BaseException:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    return {"name":name,"arguments":arguments,"returncode":code,"timed_out":timed_out,
            "elapsed_s":time.monotonic()-started,"passed":code==0 and not timed_out}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True,help="existing local Idefics3/SmolVLM weights")
    add_placement_arguments(parser)
    parser.add_argument("--output-dir",type=Path,required=True,help="new directory; no overwrite")
    parser.add_argument("--benchmark",action="store_true",help="also run paired Gloo/NCCL HTTP ablations")
    parser.add_argument("--requests",type=int,default=20)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--concurrency",type=int,default=1,help="HTTP concurrency for optional benchmark")
    parser.add_argument("--max-inflight",type=int,default=4,help="HTTP/runtime admission limit")
    args=parser.parse_args()
    if args.requests<1 or args.rounds<1:
        parser.error("requests and rounds must be positive")
    if not 1<=args.concurrency<=args.max_inflight:
        parser.error("concurrency must be positive and no greater than max-inflight")
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={"passed":False,"hardware_validated":False,"created_utc":datetime.now(timezone.utc).isoformat(),
            "steps":[],"is_performance_benchmark":args.benchmark,
            "benchmark_concurrency":args.concurrency,"max_inflight":args.max_inflight,
            "scope":"single-host V/P/D device-pool acceptance; bounded request pipeline; no multi-node/RDMA or same-device asynchronous stream-overlap claim"}
    try:
        spec=VModelSpec(model_path=str(args.model.resolve()),transport="nccl",**placement_kwargs(args))
        spec.full_config()  # Reject unsupported model families before GPU work.
        spec.check_hardware()
        report["hardware_validated"]=True
        report.update(stage_devices=list(spec.devices),stage_pools=[list(pool) for pool in spec.pools],
                      worker_count=spec.worker_count)
        report["hardware"]=hardware_snapshot(spec.devices)
        manifest={"source_sha256":{str(p.relative_to(ROOT)):file_sha(p)
                                  for folder in ("src/lmm_serving","tests","utils")
                                  for p in sorted((ROOT/folder).rglob("*.py"))},
                  "model_files_sha256":{p.name:file_sha(p) for p in sorted(args.model.iterdir())
                                        if p.is_file() and p.suffix in (".json",".safetensors")},
                  "base_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
                  "python":sys.version,
                  "pip_freeze":subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True).stdout.splitlines()}
        (args.output_dir/"source_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"VPD_RUN_THREE_GPU_TESTS":"0",
             "VPD_RUN_FOUR_GPU_TESTS":"0","PYTHONUNBUFFERED":"1"}
        placement=[argument for flag,pool in zip(("--vision-devices","--prefill-devices","--decode-devices"),spec.pools)
                   for argument in (flag,*pool)]
        steps=[]
        if all(len(pool)==1 for pool in spec.pools):
            env.update(VPD_RUN_THREE_GPU_TESTS="1",VPD_TEST_STAGE_DEVICES=",".join(spec.devices))
            steps.append(("three_gpu_tests",["tests/test_vpd_transport.py"],360))
        elif spec.worker_count==4:
            env.update(VPD_RUN_FOUR_GPU_TESTS="1",VPD_TEST_STAGE_POOLS=json.dumps(spec.pools),
                       VPD_TEST_FOUR_GPU_DEVICES=",".join(spec.devices))
            steps.append(("four_gpu_tests",["tests/test_vpd_pools.py"],360))
        for backend in ("gloo","nccl"):
            steps.append((f"probe_{backend}",["utils/probe_vpd_transport.py","--transport",backend,
                          *placement,"--output",str(args.output_dir/f"probe_{backend}.json")],240))
            steps.append((f"model_{backend}",["utils/validate_vpd_model.py","--model",str(args.model.resolve()),
                          "--transport",backend,*placement,"--max-inflight",str(args.max_inflight),
                          "--output",str(args.output_dir/f"model_{backend}.json")],600))
        if args.benchmark:
            for backend in ("gloo","nccl"):
                steps.append((f"benchmark_{backend}",["utils/benchmark_vpd.py","--model",str(args.model.resolve()),
                              "--transport",backend,*placement,"--no-handoff-checksum",
                              "--rounds",str(args.rounds),"--requests",str(args.requests),
                              "--concurrency",str(args.concurrency),"--max-inflight",str(args.max_inflight),
                              "--output-dir",str(args.output_dir/f"benchmark_{backend}")],3600))
        for name,arguments,timeout in steps:
            result=run_step(name,arguments,args.output_dir,env,timeout)
            report["steps"].append(result)
            (args.output_dir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(result),flush=True)
            if not result["passed"]:
                raise RuntimeError(f"{name} failed; see {name}.log")
        report["passed"]=True
    except Exception:
        report["error"]=traceback.format_exc()
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"hardware_validated":report["hardware_validated"],
                      "output_dir":str(args.output_dir),"error":report.get("error")}),flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
