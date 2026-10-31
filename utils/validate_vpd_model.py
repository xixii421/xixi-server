"""Real pretrained SmolVLM: unsplit pixel input versus three independent stages."""
import argparse
import base64
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import threading
from urllib.request import Request

from PIL import Image
import torch

from lmm_serving.vpd_local import LocalVPD,VModelSpec,reference_generate
from lmm_serving.vpd_service import VPDHTTPServer
from lmm_serving.stream_benchmark import measure,open_endpoint
from lmm_serving.vpd_options import add_runtime_arguments,placement_kwargs


def make_messages(colors=()):
    content=[]
    for color in colors:
        output=io.BytesIO()
        Image.new("RGB",(64,64),color).save(output,format="PNG")
        content.append({"type":"image_url","image_url":{"url":
            "data:image/png;base64,"+base64.b64encode(output.getvalue()).decode()}})
    content.append({"type":"text","text":"Describe the colors in order."})
    return [{"role":"user","content":content}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,help="local pretrained model; omit for deterministic tiny Idefics3")
    add_runtime_arguments(parser,default_device="cuda:0")
    parser.add_argument("--vision-cache-mib",type=int,default=16)
    parser.add_argument("--new-tokens",type=int,default=8)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    spec=VModelSpec(model_path=str(args.model.resolve()) if args.model else None,
                   device=args.device,dtype="bfloat16",transport=args.transport,
                   **placement_kwargs(args),
                   verify_payload=not args.no_handoff_checksum,
                   vision_cache_bytes=args.vision_cache_mib*1024*1024)
    try:
        spec.check_hardware()
    except RuntimeError as error:
        report={"passed":False,"hardware_validated":False,"tests_executed":0,
                "error":str(error),"model_spec":asdict(spec)}
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report),flush=True)
        return 2
    reference_spec=replace(spec,device=spec.devices[0])
    colors=[[],["red"],["blue"],["red","blue"],["blue","red"],["red","red"]]
    while (sum(bool(case) for case in colors)<len(spec.pools[0]) or
           len(colors)<max(len(spec.pools[1]),len(spec.pools[2]))):
        colors.append(["red","blue"])
    references=[]
    for case in colors:
        references.append(reference_generate(reference_spec,make_messages(case),args.new_tokens))
        print(json.dumps({"reference_ready":case}),flush=True)
    cases=[]
    with LocalVPD(spec,startup_timeout_s=180,request_timeout_s=60,max_inflight=args.max_inflight) as runtime:
        for case,reference in zip(colors,references):
            result=runtime.run(make_messages(case),args.new_tokens)
            passed=result["token_ids"]==reference
            cases.append({"colors":case,"matched":passed,"reference_token_ids":reference,
                          "generated_text":runtime.processor.tokenizer.decode(result["token_ids"],skip_special_tokens=True) if runtime.processor else None,
                          "result":result})
            print(json.dumps({"three_stage_checked":case,"matched":passed}),flush=True)
        server=VPDHTTPServer(("127.0.0.1",0),runtime)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        payload={"model":server.model_name,"messages":make_messages(["red","blue"]),
                 "max_tokens":args.new_tokens,"temperature":0}
        try:
            with open_endpoint(Request(endpoint,data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"}),timeout=60) as response:
                nonstream=json.load(response)
            stream=measure(endpoint,payload,timeout_s=60)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        expected_text=(runtime.processor.tokenizer.decode(references[3],skip_special_tokens=True)
                       if runtime.processor else "".join(f"{t} " for t in references[3]))
    invariants={
        "all_cases_match_unsplit":all(case["matched"] for case in cases),
        "all_independent_processes":all(len(set(c["result"]["worker_pids"]))==spec.worker_count for c in cases),
        "text_bypasses_vision":cases[0]["result"]["vision_bypassed"],
        "requested_placement_reported":all(c["result"]["stage_devices"]==list(spec.devices) for c in cases),
        "requested_transport_reported":all(c["result"]["transport"]==spec.transport for c in cases),
        "http_nonstream_matches":nonstream["backend_metrics"]["token_ids"]==references[3],
        "http_stream_ok":stream["status"]=="ok",
        "http_stream_matches":stream["content"]==expected_text,
    }
    for stage,name in enumerate(("vision","prefill","decode")):
        used={case["result"]["route"][name] for case in cases}
        invariants[f"all_{name}_workers_used"]=set(spec.ranks_for(stage))<=used
    checksum_checks=("different_images_different_features","swapping_images_changes_ordered_features")
    if spec.verify_payload:
        invariants[checksum_checks[0]]=cases[1]["result"]["prefill_feature_digest"]!=cases[2]["result"]["prefill_feature_digest"]
        invariants[checksum_checks[1]]=cases[3]["result"]["prefill_feature_digest"]!=cases[4]["result"]["prefill_feature_digest"]
    report={"passed":all(invariants.values()),"invariants":invariants,"model_spec":asdict(spec),
            "checks_skipped":list(checksum_checks) if not spec.verify_payload else [],
            "hardware_validated":True,"tests_executed":len(cases),
            "checksum_identity_checks_executed":spec.verify_payload,
            "torch_version":torch.__version__,"device_names":[torch.cuda.get_device_name(d) if d.startswith("cuda") else "CPU" for d in spec.devices],
            "cases":cases,"stream_measurement":stream,"http_backend":nonstream["backend_metrics"],
            "scope":spec.scope,
            "is_performance_benchmark":False}
    with args.output.open("x",encoding="utf-8") as output:
        json.dump(report,output,ensure_ascii=False,indent=2)
    print(json.dumps({"passed":report["passed"],"invariants":invariants,"output":str(args.output)},indent=2))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
