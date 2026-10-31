"""Loopback HTTP/SSE entry for the local Qwen3 PD correctness reference."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import uuid

import torch

from .pd_local import LocalPD, ModelSpec


class BodyLimitError(ValueError):
    status_code = 413


class PDHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime, tokenizer=None, max_inflight=2):
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        self.runtime, self.tokenizer = runtime, tokenizer
        self.slots = threading.BoundedSemaphore(max_inflight)
        self.model_name = Path(runtime.spec.model_path).name if runtime.spec.model_path else "tiny-qwen3"
        self.max_body_bytes = 65536
        self.scope = "single-host PD reference; text only; serialized generation"
        super().__init__(address, Handler)

    def prepare(self, messages, new_tokens):
        if any(not isinstance(m["content"], str) for m in messages):
            raise ValueError("messages must contain text-only role/content objects")
        if self.tokenizer is None:
            text = "\n".join(m["content"] for m in messages)
            tokens = [1]+[3+(byte % 61) for byte in text.encode()]
        else:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, enable_thinking=False)
        self.runtime.spec.validate(tokens, new_tokens)
        return tokens

    def execute(self, prepared, new_tokens, *, on_token=None):
        return self.runtime.run(prepared, new_tokens, on_token=on_token)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status, body):
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def event(self, body):
        data = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        self.wfile.write(("data: "+data+"\n\n").encode())
        self.wfile.flush()

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "not found"})
            return
        active = (not self.server.runtime._closed
                  and all(p.is_alive() for p in self.server.runtime.processes))
        self.reply(200 if active else 503,
                   {"ready": active, "model": self.server.model_name,
                    "scope": self.server.scope})

    def parse_payload(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("request body must be nonempty")
        if length > self.server.max_body_bytes:
            raise BodyLimitError("request body exceeds configured byte limit")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict) or data.get("model") != self.server.model_name:
            raise ValueError("model must match the running model name")
        allowed = {"model", "messages", "max_tokens", "temperature", "n", "stream", "stream_options"}
        if set(data)-allowed:
            raise ValueError("unsupported request fields")
        if data.get("n", 1) != 1 or data.get("temperature", 0) != 0:
            raise ValueError("only n=1 and greedy temperature=0 are supported")
        if type(data.get("stream", False)) is not bool:
            raise ValueError("stream must be boolean")
        options = data.get("stream_options", {})
        if (not isinstance(options, dict) or set(options)-{"include_usage","include_backend_metrics"}
                or any(type(options.get(key,False)) is not bool for key in ("include_usage","include_backend_metrics"))):
            raise ValueError("invalid stream_options")
        messages = data.get("messages")
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                       or m["role"] not in ("user", "assistant", "system") for m in messages)):
            raise ValueError("messages must contain role/content objects")
        new_tokens = data.get("max_tokens", 8)
        tokens = self.server.prepare(messages, new_tokens)
        return data, tokens, new_tokens

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": "not found"})
            return
        if self.server.runtime._closed:
            self.reply(503, {"error": "PD channel stopped; restart the service"})
            return
        if not self.server.slots.acquire(blocking=False):
            self.reply(429, {"error": "local PD admission capacity exhausted"})
            return
        streaming = False
        try:
            data, tokens, count = self.parse_payload()
            request_id = "chatcmpl-"+uuid.uuid4().hex
            def chunk(delta, finish=None):
                return {"id": request_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": self.server.model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if data.get("stream", False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                streaming = True
                self.event(chunk({"role": "assistant"}))
                if self.server.tokenizer is None:
                    callback = lambda token: self.event(chunk({"content": f"{token} "}))
                    streamer = None
                else:
                    from transformers import TextStreamer
                    handler = self
                    class SSEStreamer(TextStreamer):
                        def on_finalized_text(self, text, stream_end=False):
                            if text:
                                handler.event(chunk({"content": text}))
                    streamer = SSEStreamer(self.server.tokenizer, skip_special_tokens=True)
                    callback = lambda token: streamer.put(torch.tensor([token]))
                result = self.server.execute(tokens, count, on_token=callback)
                if streamer:
                    streamer.end()
                self.event(chunk({}, "length"))
                options=data.get("stream_options",{})
                if options.get("include_usage", False) or options.get("include_backend_metrics",False):
                    usage = chunk({})
                    usage["choices"] = []
                    prompt_count = result["received_cache_tokens"]
                    if options.get("include_usage",False):
                        usage["usage"] = {"prompt_tokens": prompt_count, "completion_tokens": count,
                                          "total_tokens": prompt_count+count}
                    if options.get("include_backend_metrics",False):
                        usage["backend_metrics"]=result
                    self.event(usage)
                self.event("[DONE]")
            else:
                result = self.server.execute(tokens, count)
                content = (self.server.tokenizer.decode(result["token_ids"], skip_special_tokens=True)
                           if self.server.tokenizer else "".join(f"{t} " for t in result["token_ids"]))
                self.reply(200, {"id": request_id, "object": "chat.completion",
                    "model": self.server.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "length"}],
                    "usage": {"prompt_tokens": result["received_cache_tokens"], "completion_tokens": count,
                              "total_tokens": result["received_cache_tokens"]+count}, "backend_metrics": result})
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as error:
            if streaming:
                try:
                    self.event({"error": {"message": str(error)}})
                    self.event("[DONE]")
                except OSError:
                    pass
            else:
                self.reply(getattr(error,"status_code",400 if isinstance(error, (ValueError, TypeError, KeyError)) else 503),
                           {"error": str(error)})
        finally:
            self.server.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    spec = ModelSpec(model_path=str(args.model.resolve()) if args.model else None,
                     device=args.device, dtype=args.dtype)
    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    with LocalPD(spec) as runtime:
        server = PDHTTPServer(("127.0.0.1", args.port), runtime, tokenizer)
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                          "model": server.model_name, "device": spec.device}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
