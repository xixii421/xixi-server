from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from flask import Flask, jsonify

from lmm_serving.common.routing import make_scheduler

from .protocol import RequestError, RequestRegistry, remaining
from .workers import configure_app, json_body


def post(url, payload, deadline):
    try:
        response = requests.post(url, json=payload, timeout=remaining(deadline))
    except requests.Timeout:
        raise RequestError(f"downstream timeout: {url}", 504) from None
    except requests.RequestException:
        raise RequestError(f"downstream unavailable: {url}", 503) from None
    try:
        data = response.json()
    except ValueError:
        raise RequestError(f"invalid downstream response: {url}", 502) from None
    if response.status_code != 200:
        raise RequestError(data.get("error", "downstream error"), response.status_code)
    return data


def create_cluster_app(cfg, fatal):
    app = Flask("learning-cluster")
    configure_app(app)
    registry = RequestRegistry()
    admission = threading.BoundedSemaphore(cfg.max_inflight)
    # One sequencer owns BOTH HTTP triggers for every Gloo transfer. Locks local
    # to individual requests cannot guarantee matching send/receive order.
    image_pipeline = threading.Lock()
    metrics_lock = threading.Lock()
    metrics_path = Path(cfg.metrics_dir) / "requests.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    image_url = f"http://{cfg.host}:{cfg.image_port}"
    text_url = f"http://{cfg.host}:{cfg.text_port}"
    image_scheduler = make_scheduler("least_loaded", [image_url])
    text_scheduler = make_scheduler("least_loaded", [text_url])

    @app.get("/health")
    def health():
        workers = {}
        ready = not fatal.is_set()
        for role, url in (("image", image_url), ("text", text_url)):
            try:
                response = requests.get(url + "/health", timeout=0.5)
                workers[role] = response.json()
                ready = ready and response.status_code == 200
            except (requests.RequestException, ValueError):
                workers[role] = {"ready": False}
                ready = False
        return jsonify(
            mode="single_gpu_mock", transport="gloo", ready=ready,
            active_requests=registry.active_count(), workers=workers,
        ), 200 if ready else 503

    @app.post("/generate")
    def generate():
        arrival = time.monotonic()
        deadline = arrival + cfg.request_timeout_s
        data = json_body()
        unknown = set(data) - {"req_id", "prompt", "image_ids"}
        if unknown:
            raise RequestError(f"unsupported learning-mode fields: {sorted(unknown)}")
        req_id = data.get("req_id", uuid.uuid4().hex)
        prompt = data.get("prompt", "Describe the image.")
        ids = data.get("image_ids", [])
        if not isinstance(req_id, str) or not 1 <= len(req_id) <= 128:
            raise RequestError("req_id must be a nonempty string up to 128 characters")
        if not isinstance(prompt, str) or len(prompt) > 8192:
            raise RequestError("prompt must be a string up to 8192 characters")
        if not isinstance(ids, list) or len(ids) > cfg.max_images:
            raise RequestError(f"image_ids must be a list of at most {cfg.max_images} images")
        if any(not isinstance(i, str) or not i or len(i) > 512 for i in ids):
            raise RequestError("image_ids must contain nonempty path strings")
        if fatal.is_set():
            raise RequestError("Gloo channel failed; restart the learning stack", 503)
        if not admission.acquire(blocking=False):
            raise RequestError("max_inflight reached; retry later with a new req_id", 429)
        try:
            registry.begin(req_id)
        except Exception:
            admission.release()
            raise

        key = uuid.uuid4().hex
        internal = {"request_key": key, "deadline": deadline}
        locked = False
        image_node = text_node = None
        image_load = len(ids) * 32
        text_load = len(prompt.split()) + image_load
        status, result = 200, {}
        metrics = {
            "mode": "single_gpu_mock", "transport": "gloo", "req_id": req_id,
            "request_key": key, "num_images": len(ids),
            "simulated_stages": ["encoding", "prefill", "decode"],
            "queue_ms": 0.0, "preprocessing_ms": 0.0, "encoding_ms": 0.0,
            "transfer_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0,
            "mock_generation_ms": 0.0, "features": [], "cleanup_errors": [],
        }
        try:
            features = []
            if ids:
                locked = image_pipeline.acquire(timeout=remaining(deadline))
                metrics["queue_ms"] = (time.monotonic() - arrival) * 1000
                if not locked:
                    raise RequestError("image pipeline queue deadline exceeded", 504)
                image_node = image_scheduler.acquire(image_load)
                encoded = post(image_node + "/encode", {**internal, "image_ids": ids}, deadline)
                features = encoded["features"]
                metrics["preprocessing_ms"] = sum(f["preprocessing_ms"] for f in features)
                metrics["encoding_ms"] = sum(f["encoding_ms"] for f in features)
            text_node = text_scheduler.acquire(text_load)
            for feature in features:
                # Do not start an uncancellable Gloo operation near an HTTP
                # deadline. Leave enough time for its process-group timeout.
                if remaining(deadline) <= cfg.transfer_timeout_s + 0.5:
                    raise RequestError("insufficient deadline for a safe transfer", 504)
                payload = {**internal, "feature": feature}
                start = time.monotonic()
                try:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        sender = pool.submit(post, image_node + "/transfer", payload, deadline)
                        receiver = pool.submit(post, text_node + "/consume", payload, deadline)
                        sent, received = sender.result(), receiver.result()
                except Exception:
                    # An interrupted two-sided exchange cannot be safely reused.
                    fatal.set()
                    raise
                feature["gloo_send_ms"] = sent["gloo_send_ms"]
                feature["gloo_recv_ms"] = received["gloo_recv_ms"]
                metrics["transfer_ms"] += (time.monotonic() - start) * 1000
            generated = post(text_node + "/generate", {
                **internal, "prompt": prompt, "features": features,
            }, deadline)
            metrics["features"] = generated["features"]
            metrics["mock_generation_ms"] = generated["mock_generation_ms"]
            result = {
                "req_id": req_id, "mode": "single_gpu_mock", "transport": "gloo",
                "content": generated["content"], "features": generated["features"],
                "route": ["cluster", "image", "text"] if ids else ["cluster", "text"],
            }
        except RequestError as error:
            status = error.status
            result = {"req_id": req_id, "mode": "single_gpu_mock", "error": str(error)}
        except Exception:
            app.logger.exception("learning request failed req_id=%s", req_id)
            status = 500
            result = {"req_id": req_id, "mode": "single_gpu_mock", "error": "internal learning-mode error"}
        finally:
            # Cancellation is idempotent and also rejects a late encode/receive
            # completing after this request has already returned an error.
            nodes = [node for node in (image_node, text_node) if node]
            for node in nodes:
                try:
                    post(node + "/release", {"request_key": key}, time.monotonic() + 1)
                except RequestError as error:
                    metrics["cleanup_errors"].append(str(error))
            if image_node:
                image_scheduler.release(image_node, image_load)
            if text_node:
                text_scheduler.release(text_node, text_load)
            if locked:
                image_pipeline.release()
            registry.finish(req_id)
            admission.release()
            metrics["e2e_ms"] = (time.monotonic() - arrival) * 1000
            metrics["http_code"] = status
            # Append each completed request immediately, including failure paths.
            with metrics_lock:
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        result["metrics"] = metrics
        return jsonify(result), status

    return app
