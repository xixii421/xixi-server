"""Real Qwen3 prefill/decode in separate processes, joined by Gloo KV transfer.

This is a bounded correctness reference, not the production vLLM connector.
Requests execute serially; there is no batching, prefix reuse or multimodal input.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import cached_property
import gc
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
import uuid

import torch
import torch.distributed as dist


def validate_device(device):
    if not isinstance(device, str) or (device != "cpu" and
            re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None):
        raise ValueError("device must be cpu or an indexed CUDA device, e.g. cuda:0")


@dataclass(frozen=True)
class ModelSpec:
    model_path: str | None = None  # None selects a deterministic tiny test model.
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42
    threads: int = 2
    max_prompt_tokens: int = 256
    max_new_tokens: int = 32
    max_cache_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("supported dtypes: float32, bfloat16")
        validate_device(self.device)
        for name in ("threads", "max_prompt_tokens", "max_new_tokens", "max_cache_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_path is not None and not Path(self.model_path).is_dir():
            raise ValueError("model_path must be an existing local directory")

    @cached_property
    def _config(self):
        if self.model_path:
            cfg = json.loads((Path(self.model_path)/"config.json").read_text())
            if cfg.get("model_type") != "qwen3" or cfg.get("use_sliding_window"):
                raise ValueError("this reference supports full-attention Qwen3 only")
            return cfg
        return {"model_type": "qwen3", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 8, "max_position_embeddings": 512,
                "bos_token_id": 1, "eos_token_id": 2,
                "attention_dropout": 0.0, "use_cache": True,
                "use_sliding_window": False, "sliding_window": None}

    def config(self):
        return self._config

    @cached_property
    def _signature(self):
        # Both workers load the same immutable local source. This is an identity
        # check, not a cryptographic validation of the weight file contents.
        source = str(Path(self.model_path).resolve()) if self.model_path else "tiny"
        return hashlib.sha256(json.dumps([source, self.config(), self.dtype, self.seed],
                                        sort_keys=True).encode()).hexdigest()

    def signature(self):
        return self._signature

    def validate_generation(self, new_tokens):
        if type(new_tokens) is not int or not 1 <= new_tokens <= self.max_new_tokens:
            raise ValueError("invalid generation length")

    def validate(self, tokens, new_tokens):
        self.validate_generation(new_tokens)
        cfg = self.config()
        if (not isinstance(tokens, list) or not tokens or len(tokens) > self.max_prompt_tokens
                or any(type(t) is not int or not 0 <= t < cfg["vocab_size"] for t in tokens)):
            raise ValueError("invalid token IDs or prompt exceeds configured capacity")
        if len(tokens)+new_tokens > cfg["max_position_embeddings"]:
            raise ValueError("request exceeds model context capacity")
        size = (2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"]
                * cfg["head_dim"] * len(tokens) * (4 if self.dtype == "float32" else 2))
        # Reserve for the complete bounded generation, not only incoming KV.
        peak = size * (len(tokens)+new_tokens) // len(tokens)
        if peak > self.max_cache_bytes:
            raise ValueError("request exceeds KV payload budget")
        return size


def load_model(spec):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(spec.threads)
    torch.manual_seed(spec.seed)
    if spec.model_path:
        model = Qwen3ForCausalLM.from_pretrained(
            spec.model_path, local_files_only=True, torch_dtype=getattr(torch, spec.dtype),
            attn_implementation="eager")
    else:
        config = Qwen3Config(**spec.config())
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).to(getattr(torch, spec.dtype))
    return model.to(spec.device).eval()


@torch.inference_mode()
def forward(model, tokens, cache=None):
    return model(input_ids=torch.tensor([tokens], dtype=torch.long, device=model.device),
                 past_key_values=cache, use_cache=True, logits_to_keep=1)


def reference_generate(spec, tokens, new_tokens):
    """Independent, non-disaggregated model execution; fixed-length greedy."""
    spec.validate(tokens, new_tokens)
    model = load_model(spec)
    result = forward(model, tokens)
    output = [int(result.logits[0, -1].argmax())]
    for _ in range(new_tokens-1):
        result = forward(model, [output[-1]], result.past_key_values)
        output.append(int(result.logits[0, -1].argmax()))
    del model, result
    gc.collect()
    if spec.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def tensor_digest(tensor):
    """按原始字节计算 SHA256；BF16 先看成 uint8，避免直接转 NumPy 的 dtype 限制。"""
    # 对 CUDA 张量，.cpu() 会产生整张量的 CPU 镜像；摘要校验有传输和计算成本。
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def send_header(header, destination=1):
    """控制消息分两次发送：先发 JSON 字节长度，再发 JSON 字节本身。"""
    # 这些张量都在 CPU，且不指定 group，因此使用 worker 初始化的默认 Gloo 组。
    # 大的视觉特征和 KV 另走张量通道；这里不使用 pickle 传递 Python 对象。
    data = json.dumps(header, allow_nan=False).encode()
    if len(data) > 131072:
        raise ValueError("header exceeds protocol limit")
    dist.send(torch.tensor([len(data)], dtype=torch.int64), dst=destination)
    dist.send(torch.tensor(list(data), dtype=torch.uint8), dst=destination)


def receive_header(source=0):
    """与 send_header 严格配对：长度 -> 检查上限 -> 分配 -> 接收并解析 JSON。"""
    # 不知道消息长度就无法提前分配接收张量；先限制长度也能避免任意大分配。
    size = torch.zeros(1, dtype=torch.int64)
    dist.recv(size, src=source)
    if not 0 < int(size[0]) <= 131072:
        raise ValueError("invalid header size")
    data = torch.empty(int(size[0]), dtype=torch.uint8)
    dist.recv(data, src=source)
    return json.loads(bytes(data.tolist()))


def validate_header(header, spec, request, *, verify_payload=True):
    """在接收 KV 前核对身份、模型与预算，防止串请求、旧消息或配置不一致。"""
    # request_id 区分请求，sequence 区分串行通道上的交接次序。
    # prompt_digest 比对完整 token 序列，context_digest 比对有序多模态上下文。
    expected = {"version": 1, "request_id": request["request_id"],
                "sequence": request["sequence"], "signature": spec.signature(),
                "prompt_length": len(request["tokens"]), "new_tokens": request["new_tokens"],
                "dtype": spec.dtype,
                "prompt_digest": hashlib.sha256(json.dumps(request["tokens"]).encode()).hexdigest()}
    if any(header.get(k) != value for k, value in expected.items()):
        raise ValueError("KV handoff identity or model metadata mismatch")
    if header.get("context_digest", "") != request.get("context_digest", ""):
        raise ValueError("multimodal context identity mismatch")
    first = header.get("first_token")
    if type(first) is not int or not 0 <= first < spec.config()["vocab_size"]:
        raise ValueError("invalid first token")
    # 每层 K/V 各有一个摘要。关闭载荷校验时约定为 None，不能双方各用各的模式。
    digests = header.get("digests")
    if (not isinstance(digests, list) or len(digests) != 2*spec.config()["num_hidden_layers"]
            or any((not isinstance(d, str) or len(d) != 64) if verify_payload else d is not None
                   for d in digests)):
        raise ValueError("invalid cache digest metadata")
    spec.validate(request["tokens"], request["new_tokens"])


def send_cache(spec, request, result, first_token, ready_at, destination=1):
    """旧的文本 PD/Gloo 路径：逐个发送 CPU K/V；三阶段打包路径见 send_kv。"""
    cache = result.past_key_values.to_legacy_cache()
    tensors = [tensor.detach().cpu().contiguous() for pair in cache for tensor in pair]
    header = {"version": 1, "request_id": request["request_id"],
              "sequence": request["sequence"], "signature": spec.signature(),
              "prompt_length": len(request["tokens"]), "new_tokens": request["new_tokens"],
              "dtype": spec.dtype, "first_token": first_token,
              "prompt_digest": hashlib.sha256(json.dumps(request["tokens"]).encode()).hexdigest(),
              "context_digest": request.get("context_digest", ""),
              "first_token_ready_s": ready_at, "digests": [tensor_digest(t) for t in tensors]}
    send_header(header, destination)
    for tensor in tensors:
        dist.send(tensor, dst=destination)
    # 保留源缓存和 CPU 副本，直到接收端确认所有 K/V 已验证并准备好用于计算。
    ack = torch.zeros(2, dtype=torch.int64)
    dist.recv(ack, src=destination)
    if ack.tolist() != [request["sequence"], len(request["tokens"])]:
        raise ValueError("invalid handoff acknowledgement")


def receive_cache(spec, request, source=0):
    """旧文本 PD 接收端：验头 -> 逐个收 K/V -> 上传设备 -> 恢复缓存 -> ACK。"""
    from transformers import DynamicCache
    header = receive_header(source)
    validate_header(header, spec, request)
    cfg = spec.config()
    shape = (1, cfg["num_key_value_heads"], len(request["tokens"]), cfg["head_dim"])
    tensors = []
    for digest in header["digests"]:
        tensor = torch.empty(shape, dtype=getattr(torch, spec.dtype))
        dist.recv(tensor, src=source)
        if tensor_digest(tensor) != digest:
            raise ValueError("KV payload checksum mismatch")
        tensors.append(tensor)
    # Gloo 接收的是 CPU 张量；模型在 GPU 时必须先上传并确认搬运完成。
    tensors = [t.to(spec.device) for t in tensors]
    if spec.device.startswith("cuda"):
        torch.cuda.synchronize()
    cache = DynamicCache.from_legacy_cache(tuple(zip(tensors[::2], tensors[1::2])))
    if cache.get_seq_length() != len(request["tokens"]):
        raise ValueError("received cache length mismatch")
    dist.send(torch.tensor([request["sequence"], len(request["tokens"])], dtype=torch.int64), dst=source)
    return cache, header


def worker(rank, spec, store_path, control, events, wire_timeout_s):
    try:
        model = load_model(spec)
        dist.init_process_group("gloo", init_method="file://"+store_path,
                                rank=rank, world_size=2,
                                timeout=timedelta(seconds=wire_timeout_s))
        events.put({"type": "ready", "rank": rank})
        while True:
            request = control.get()
            if request is None:
                break
            spec.validate(request["tokens"], request["new_tokens"])
            if rank == 0:
                started = time.perf_counter()
                result = forward(model, request["tokens"])
                first = int(result.logits[0, -1].argmax())
                ready_at = time.perf_counter()
                events.put({"type": "token", "request_id": request["request_id"],
                            "index": 0, "token": first})
                send_cache(spec, request, result, first, ready_at)
                events.put({"type": "handoff", "request_id": request["request_id"],
                            "prefill_ms": (ready_at-started)*1000,
                            "handoff_ms": (time.perf_counter()-ready_at)*1000})
                del result
            else:
                cache, header = receive_cache(spec, request)
                output, ready_times = [header["first_token"]], [header["first_token_ready_s"]]
                lengths = []
                for _ in range(request["new_tokens"]-1):
                    result = forward(model, [output[-1]], cache)
                    cache = result.past_key_values
                    output.append(int(result.logits[0, -1].argmax()))
                    ready_times.append(time.perf_counter())
                    events.put({"type": "token", "request_id": request["request_id"],
                                "index": len(output)-1, "token": output[-1]})
                    lengths.append(1)
                events.put({"type": "result", "request_id": request["request_id"],
                            "token_ids": output, "token_ready_times_s": ready_times,
                            "decode_input_lengths": lengths,
                            "received_cache_tokens": len(request["tokens"])})
                del cache
                if lengths:
                    del result
    except BaseException as error:
        events.put({"type": "error", "rank": rank,
                    "error": f"{type(error).__name__}: {error}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class LocalPD:
    """One serialized channel with bounded request length and fail-stop recovery."""
    def __init__(self, spec=ModelSpec(), *, startup_timeout_s=120, request_timeout_s=30):
        if any(not math.isfinite(t) or t <= 0 for t in (startup_timeout_s, request_timeout_s)):
            raise ValueError("timeouts must be positive and finite")
        self.spec, self.startup_timeout_s = spec, startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.context = mp.get_context("spawn")
        self.events = self.context.Queue()
        self.controls = [self.context.Queue() for _ in range(2)]
        self.processes = []
        self._directory = None
        self._lock = threading.Lock()
        self._closed = False
        self._sequence = 0

    def _event(self, deadline):
        while time.monotonic() < deadline:
            if any(p.exitcode is not None for p in self.processes):
                raise RuntimeError("PD worker exited; channel must be restarted")
            try:
                event = self.events.get(timeout=min(.1, max(.001, deadline-time.monotonic())))
                if event["type"] == "error":
                    raise RuntimeError(event["error"])
                return event
            except queue.Empty:
                continue
        raise TimeoutError("PD operation deadline exceeded")

    def __enter__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="modserve-pd-")
        store = str(Path(self._directory.name)/"gloo-store")
        try:
            for rank in range(2):
                process = self.context.Process(target=worker, args=(rank, self.spec, store,
                    self.controls[rank], self.events, max(self.startup_timeout_s,self.request_timeout_s)))
                self.processes.append(process)
                process.start()
            ready = set()
            deadline = time.monotonic()+self.startup_timeout_s
            while len(ready) < 2:
                event = self._event(deadline)
                if event["type"] != "ready":
                    raise RuntimeError("invalid worker startup event")
                ready.add(event["rank"])
            return self
        except BaseException:
            self.close()
            raise

    def run(self, tokens, new_tokens=8, *, on_token=None):
        cache_bytes = self.spec.validate(tokens, new_tokens)
        with self._lock:
            if self._closed or len(self.processes) != 2:
                raise RuntimeError("PD runtime is not active")
            self._sequence += 1
            request = {"request_id": uuid.uuid4().hex, "sequence": self._sequence,
                       "tokens": tokens.copy(), "new_tokens": new_tokens}
            started = time.perf_counter()
            for control in self.controls:
                control.put(request)
            deadline = time.monotonic()+self.request_timeout_s
            received = self.collect(request, deadline, on_token)
            result = {**received["result"], **received["handoff"]}
            result.pop("type")
            times = result["token_ready_times_s"]
            result.update(mode=("cpu_qwen3_pd_reference" if self.spec.device == "cpu"
                                else "single_gpu_qwen3_pd_loopback"), transport="gloo",
                          worker_pids=[p.pid for p in self.processes],
                          kv_payload_bytes=cache_bytes, e2e_ms=(time.perf_counter()-started)*1000,
                          backend_ttft_ms=(times[0]-started)*1000,
                          backend_tbt_ms=[(b-a)*1000 for a,b in zip(times,times[1:])],
                          scope="local worker token readiness; not client streaming or cross-GPU")
            return result

    def collect(self, request, deadline, on_token=None, expected=("handoff", "result")):
        """主进程收集 worker 事件：完成消息可以交错到达，Token 必须按编号输出。"""
        received, token_events, emitted = {}, {}, 0
        try:
            while len(received) < len(expected) or emitted < request["new_tokens"]:
                event = self._event(deadline)
                if event.get("request_id") != request["request_id"]:
                    raise RuntimeError("unexpected request identity or channel event")
                if event["type"] == "token":
                    index, token = event["index"], event["token"]
                    if type(index) is not int or not 0 <= index < request["new_tokens"] or index in token_events:
                        raise RuntimeError("invalid or duplicate token event")
                    token_events[index] = token
                    # P 和 D 使用不同进程的 Queue feeder，首 Token 事件可能
                    # 比 D 的后续事件晚到。先按 index 缓存，只输出连续的前缀。
                    while emitted in token_events:
                        if on_token is not None:
                            on_token(token_events[emitted])
                        emitted += 1
                    continue
                if event["type"] not in expected or event["type"] in received:
                    raise RuntimeError("unexpected or duplicate channel completion")
                received[event["type"]] = event
            if [token_events[i] for i in range(request["new_tokens"])] != received["result"]["token_ids"]:
                raise RuntimeError("streamed tokens differ from completed generation")
        except BaseException:
            self.close()
            raise
        return received

    def close(self):
        """关闭整个运行时：尝试正常退出，再终止仍卡在通信中的进程。"""
        # 发生半次交接、超时或客户端取消时，无法保证双方消息位置仍然一致。
        # 因此采用整组停止策略，不能继续复用这个对象；后续请求需新建运行时。
        if self._closed:
            return
        self._closed = True
        for control in self.controls:
            control.put(None)
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join()
        for control in self.controls:
            control.close()
            control.cancel_join_thread()
        self.events.close()
        self.events.cancel_join_thread()
        if self._directory:
            self._directory.cleanup()

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="local Qwen3 directory; omit for tiny random model")
    parser.add_argument("--prompt", default="Explain KV cache in one sentence.")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--verify", action="store_true", help="compare with independent unsplit inference")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwrites")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = ModelSpec(model_path=str(args.model.resolve()) if args.model else None,
                     dtype=args.dtype, device=args.device)
    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        tokens = tokenizer.encode(args.prompt)
    else:
        tokenizer = None
        tokens = [1, 5, 9, 12]
    expected = reference_generate(spec, tokens, args.new_tokens) if args.verify else None
    with LocalPD(spec) as runtime:
        result = runtime.run(tokens, args.new_tokens)
    result["model_spec"] = asdict(spec)
    result["reference_token_ids"] = expected
    result["greedy_matches_reference"] = result["token_ids"] == expected if expected is not None else None
    if tokenizer:
        result["generated_text"] = tokenizer.decode(result["token_ids"])
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if expected is None or result["greedy_matches_reference"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
