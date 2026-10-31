"""Offline real-model correctness + real loopback HTTP/SSE validation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import threading
from urllib.request import Request

import torch
from transformers import AutoTokenizer

from lmm_serving.pd_local import LocalPD, ModelSpec, reference_generate
from lmm_serving.pd_service import PDHTTPServer
from lmm_serving.stream_benchmark import measure,open_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = ModelSpec(model_path=str(args.model.resolve()), dtype="bfloat16", device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True)
    messages = [{"role": "user", "content": "Explain KV cache in one sentence."}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    expected = reference_generate(spec, tokens, args.new_tokens)
    with LocalPD(spec) as runtime:
        server = PDHTTPServer(("127.0.0.1", 0), runtime, tokenizer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        payload = {"model": server.model_name, "messages": messages,
                   "max_tokens": args.new_tokens, "temperature": 0}
        try:
            with open_endpoint(Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}), timeout=60) as response:
                nonstream = json.load(response)
            stream = measure(endpoint, payload, timeout_s=60)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
    observed = nonstream["backend_metrics"]["token_ids"]
    expected_text = tokenizer.decode(expected, skip_special_tokens=True)
    report = {"model_spec": asdict(spec), "torch_version": torch.__version__,
              "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU",
              "prompt_token_count": len(tokens), "reference_token_ids": expected,
              "observed_token_ids": observed, "greedy_matches_reference": expected == observed,
              "expected_text": expected_text, "streamed_text": stream["content"],
              "streamed_text_matches_reference": stream["content"] == expected_text,
              "stream_request": stream, "backend_metrics": nonstream["backend_metrics"],
              "scope": "single-host correctness and HTTP/SSE measurement; not cross-GPU performance"}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    passed = (report["greedy_matches_reference"] and report["streamed_text_matches_reference"]
              and stream["status"] == "ok")
    print(json.dumps({"passed": passed, "output": str(args.output),
                      "generated_text": expected_text, "stream_status": stream["status"],
                      "greedy_matches_reference": report["greedy_matches_reference"],
                      "streamed_text_matches_reference": report["streamed_text_matches_reference"]},
                     ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
