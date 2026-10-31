"""Measure an OpenAI-compatible streaming endpoint without guessing token times.

Run with ``PYTHONPATH=src python -m lmm_serving.stream_benchmark --help``.
This measures the endpoint supplied by the caller, not the non-streaming
ModServe /generate wrapper. SSE content events may contain multiple tokens;
their spacing is explicitly NOT reported as TBT.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import ipaddress
import math
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.parse import urlsplit
import uuid

_LOOPBACK_OPENER=build_opener(ProxyHandler({}))


def is_loopback(endpoint):
    host=urlsplit(endpoint).hostname
    if host=="localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError,TypeError):
        return False


def open_endpoint(request,timeout=60):
    """Local experiments must not inherit an HTTP proxy lacking NO_PROXY."""
    endpoint=request.full_url if isinstance(request,Request) else request
    if is_loopback(endpoint):
        return _LOOPBACK_OPENER.open(request,timeout=timeout)
    return urlopen(request,timeout=timeout)


def sse_events(lines):
    """Parse SSE data fields, including multi-line data and CRLF framing."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    # SSE dispatch requires the terminating blank line. A partial final event
    # is not a successfully completed stream.


def measure(endpoint, payload, *, timeout_s=60, opener=None, clock=time.perf_counter):
    """One request; timeout_s is the socket inactivity timeout, not a deadline."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    body = dict(payload)
    body.update(stream=True, n=1, stream_options={**body.get("stream_options",{}),"include_usage":True})
    started = clock()
    result = {"request_id": uuid.uuid4().hex, "status": "failed",
              "started_s": started, "ended_s": None, "first_content_s": None,
              "content_event_times_s": [], "content": "", "completion_tokens": None,
              "prompt_tokens": None, "finish_reason": None, "error": None,
              "http_status": None, "measurement_scope": "client_endpoint",
              "backend_metrics": None,
              "connection_route":("custom_opener" if opener is not None else
                                   "direct_loopback" if is_loopback(endpoint) else "default_proxy_policy"),
              "ttft_definition": "first_nonempty_content_delta",
              "tbt_ms": None}
    try:
        request = Request(endpoint, data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Accept": "text/event-stream"})
        pieces = []
        done = False
        with (opener or open_endpoint)(request, timeout=timeout_s) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status {response.status}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("endpoint did not return an SSE stream")
            for event in sse_events(response):
                if event == "[DONE]":
                    done = True
                    break
                chunk = json.loads(event)
                if chunk.get("error"):
                    raise ValueError(f"stream error: {chunk['error']}")
                if chunk.get("backend_metrics") is not None:
                    result["backend_metrics"]=chunk["backend_metrics"]
                choices = chunk.get("choices", [])
                if len(choices) > 1:
                    raise ValueError("multiple choices are not supported")
                if choices:
                    choice = choices[0]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if not isinstance(content, str):
                            raise ValueError("non-text content is not supported")
                        arrival = clock()
                        if result["first_content_s"] is None:
                            result["first_content_s"] = arrival
                        result["content_event_times_s"].append(arrival)
                        pieces.append(content)
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
                usage = chunk.get("usage")
                if usage is not None:
                    for key in ("completion_tokens", "prompt_tokens"):
                        count = usage.get(key)
                        if type(count) is not int or count < 0:
                            raise ValueError(f"invalid usage field: {key}")
                        result[key] = count
        result["content"] = "".join(pieces)
        if not done or result["finish_reason"] is None:
            raise ValueError("stream ended without DONE and a finish reason")
        if result["first_content_s"] is None:
            raise ValueError("stream produced no text content")
        result["status"] = "ok"
    except Exception as error:
        if isinstance(error, HTTPError):
            result["http_status"] = error.code
            error.close()
        result["error"] = f"{type(error).__name__}: {error}"
    result["ended_s"] = clock()
    first = result["first_content_s"]
    result["ttft_ms"] = (first - started) * 1000 if first is not None else None
    arrivals = result["content_event_times_s"]
    result["content_event_gaps_ms"] = [(b-a)*1000 for a, b in zip(arrivals, arrivals[1:])]
    result["e2e_ms"] = (result["ended_s"] - started) * 1000
    return result


def summarize(records, elapsed_s, ttft_slo_ms=None):
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive and finite")
    if ttft_slo_ms is not None and (not math.isfinite(ttft_slo_ms) or ttft_slo_ms <= 0):
        raise ValueError("ttft_slo_ms must be positive and finite")
    successful = [r for r in records if r["status"] == "ok"]
    ttfts = sorted(r["ttft_ms"] for r in successful)
    # Nearest-rank percentile: an explicit definition for small samples, too.
    p99 = ttfts[math.ceil(.99 * len(ttfts))-1] if ttfts else None
    tokens_known = bool(successful) and all(r["completion_tokens"] is not None for r in successful)
    good = (sum(r["ttft_ms"] <= ttft_slo_ms for r in successful)
            if ttft_slo_ms is not None else None)
    return {"requests": len(records), "succeeded": len(successful),
            "failed": len(records)-len(successful), "elapsed_s": elapsed_s,
            "successful_requests_per_s": len(successful)/elapsed_s,
            "successful_output_tokens_per_s": (
                sum(r["completion_tokens"] for r in successful)/elapsed_s if tokens_known else None),
            "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
            "ttft_p99_ms": p99, "ttft_slo_ms": ttft_slo_ms,
            "ttft_only_goodput_requests_per_s": good/elapsed_s if good is not None else None,
            "ttft_only_slo_attainment": good/len(records) if good is not None and records else None,
            "tbt_ms": None, "load_model": "closed_loop_fixed_concurrency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="full /v1/chat/completions URL")
    parser.add_argument("--payload", type=Path, required=True, help="one JSON chat request")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60, help="socket inactivity timeout in seconds")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file; never overwrites")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("requests and concurrency must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if args.ttft_slo_ms is not None and (not math.isfinite(args.ttft_slo_ms) or args.ttft_slo_ms <= 0):
        parser.error("ttft-slo-ms must be positive and finite")
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("model") or not isinstance(payload.get("messages"), list):
        parser.error("payload must be a JSON object with model and messages")
    records = []
    with args.output.open("x", encoding="utf-8") as output:
        started = time.perf_counter()
        def run(index):
            record = measure(args.endpoint, payload, timeout_s=args.timeout)
            record["benchmark_index"] = index
            return record
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending = [pool.submit(run, index) for index in range(args.requests)]
            for future in as_completed(pending):
                record = future.result()
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
        summary = summarize(records, time.perf_counter()-started, args.ttft_slo_ms)
        output.write(json.dumps({"summary": summary}) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
