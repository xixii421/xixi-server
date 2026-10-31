"""Runnable vision/prefill/decode service with a real Idefics3/SmolVLM model."""
import argparse
import json
from pathlib import Path

from .pd_service import PDHTTPServer
from .vpd_local import LocalVPD, VModelSpec
from .vpd_options import add_runtime_arguments, placement_kwargs


class VPDHTTPServer(PDHTTPServer):
    def __init__(self,address,runtime,max_inflight=None):
        if max_inflight is None:
            max_inflight=getattr(runtime,"max_inflight",2)
        tokenizer = runtime.processor.tokenizer if runtime.processor else None
        super().__init__(address,runtime,tokenizer,max_inflight)
        self.model_name = Path(runtime.spec.model_path).name if runtime.spec.model_path else "tiny-idefics3"
        self.max_body_bytes = 8*1024*1024
        self.scope = runtime.scope

    def prepare(self,messages,new_tokens):
        return self.runtime.prepare(messages,new_tokens)

    def execute(self,prepared,new_tokens,*,on_token=None):
        return self.runtime.run_prepared(prepared,on_token=on_token)


def run(args):
    spec = VModelSpec(model_path=str(Path(args.model).resolve()) if args.model else None,
                      device=args.device,dtype=args.dtype,vision_cache_bytes=args.vision_cache_mib*1024*1024,
                      **placement_kwargs(args),
                      transport=args.transport,verify_payload=not args.no_handoff_checksum)
    with LocalVPD(spec,max_inflight=args.max_inflight,
                  startup_timeout_s=args.startup_timeout,request_timeout_s=args.request_timeout) as runtime:
        server = VPDHTTPServer(("127.0.0.1",args.port),runtime)
        print(json.dumps({"url":f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                          "model":server.model_name,"device":spec.devices[0],"worker_pids":
                          [p.pid for p in runtime.processes],"stages":["vision","prefill","decode"],
                          "stage_devices":list(spec.devices),"transport":spec.transport,
                          "stage_pools":[list(pool) for pool in spec.pools],"worker_info":runtime.worker_info,
                          "control_transport":"gloo","verify_payload":spec.verify_payload,
                          "scope":runtime.scope}),flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,help="local SmolVLM/Idefics3 directory; omit for tiny model")
    add_runtime_arguments(parser)
    parser.add_argument("--dtype",choices=("float32","bfloat16"),default="bfloat16")
    parser.add_argument("--port",type=int,default=8099)
    parser.add_argument("--vision-cache-mib",type=int,default=16,help="CPU visual feature LRU budget; 0 disables")
    parser.add_argument("--startup-timeout",type=float,default=180)
    parser.add_argument("--request-timeout",type=float,default=60)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
