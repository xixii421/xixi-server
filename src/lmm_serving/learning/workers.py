from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from .config import LearningConfig
from .protocol import (
    FEATURE_SHAPE, FeatureStore, RequestError, remaining, tensor_checksum,
)

logger = logging.getLogger(__name__)


def json_body():
    data = request.get_json()
    if not isinstance(data, dict):
        raise RequestError("JSON body must be an object")
    return data


def configure_app(app):
    app.config["MAX_CONTENT_LENGTH"] = 65536

    @app.errorhandler(RequestError)
    def request_error(error):
        return jsonify(error=str(error), mode="single_gpu_mock"), error.status

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description, mode="single_gpu_mock"), error.code


def envelope(data):
    key = data.get("request_key")
    if not isinstance(key, str) or len(key) != 32:
        raise RequestError("invalid internal request_key")
    try:
        int(key, 16)
        deadline = float(data["deadline"])
    except (ValueError, TypeError, KeyError):
        raise RequestError("invalid request deadline or key") from None
    if not 0 < remaining(deadline) < 86400:
        raise RequestError("invalid request deadline")
    return key, deadline


def descriptor(data, request_key):
    if not isinstance(data, dict):
        raise RequestError("invalid feature descriptor")
    transfer_id = data.get("transfer_id", "")
    if not isinstance(transfer_id, str) or not transfer_id.startswith(request_key + ":"):
        raise RequestError("transfer_id does not belong to this request")
    if data.get("shape") != list(FEATURE_SHAPE) or data.get("dtype") != "float32":
        raise RequestError("expected a 4x8 float32 feature")
    checksum = data.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise RequestError("invalid feature checksum")
    return transfer_id


def image_path(cfg, image_id):
    root = Path(cfg.image_dir)
    path = (root / image_id).resolve()
    if not path.is_relative_to(root):
        raise RequestError("image path escapes image_dir")
    if not path.is_file():
        raise RequestError(f"image not found: {image_id}", 404)
    if path.stat().st_size > cfg.max_image_bytes:
        raise RequestError("image exceeds max_image_bytes", 413)
    return path


class GlooTransport:
    """Two-rank data plane. HTTP carries descriptors; Gloo carries identity + data.

    The cluster serializes image pipelines across ALL requests, so both ranks
    execute identical sequences. The header detects accidental request mismatch.
    A failed operation poisons this communicator: the supervisor stops the stack.
    """

    def __init__(self, fatal):
        self.fatal = fatal

    def send(self, transfer_id, tensor):
        try:
            header = torch.tensor(list(hashlib.sha256(transfer_id.encode()).digest()), dtype=torch.uint8)
            dist.send(header, dst=1)
            dist.send(tensor.contiguous(), dst=1)
        except Exception:
            self.fatal.set()
            raise RequestError("Gloo send failed; restart the learning stack", 503) from None

    def recv(self, transfer_id):
        try:
            header = torch.empty(32, dtype=torch.uint8)
            tensor = torch.empty(FEATURE_SHAPE, dtype=torch.float32)
            dist.recv(header, src=0)
            dist.recv(tensor, src=0)
            expected = list(hashlib.sha256(transfer_id.encode()).digest())
            if header.tolist() != expected:
                raise ValueError("transfer identity mismatch")
            return tensor
        except Exception:
            self.fatal.set()
            raise RequestError("Gloo receive failed or mismatched; restart the learning stack", 503) from None


def create_worker_app(role, cfg, stop, fatal):
    app = Flask(f"learning-{role}")
    configure_app(app)
    store = FeatureStore(cfg.max_pending_bytes, cfg.feature_ttl_s)
    transport = GlooTransport(fatal)
    counters = {"encoded_images": 0, "sent_features": 0, "received_features": 0}

    @app.get("/health")
    def health():
        return jsonify(
            role=role, rank=0 if role == "image" else 1, transport="gloo",
            pid=os.getpid(),
            mode="single_gpu_mock", ready=not fatal.is_set(),
            **store.stats(), **counters,
        ), 503 if fatal.is_set() else 200

    @app.post("/release")
    def release():
        key = json_body().get("request_key")
        if not isinstance(key, str) or len(key) != 32:
            raise RequestError("invalid request_key")
        store.release(key)
        return jsonify(**store.stats())

    if role == "image":
        @app.post("/encode")
        def encode():
            data = json_body()
            key, deadline = envelope(data)
            ids = data.get("image_ids")
            if not isinstance(ids, list) or not 1 <= len(ids) <= cfg.max_images:
                raise RequestError("invalid image_ids")
            if any(not isinstance(i, str) or not i for i in ids):
                raise RequestError("image_ids must contain nonempty strings")
            features = []
            try:
                for index, image_id in enumerate(ids):
                    remaining(deadline)
                    start = time.perf_counter()
                    path = image_path(cfg, image_id)
                    try:
                        with Image.open(path) as image:
                            if image.width * image.height > cfg.max_image_pixels:
                                raise RequestError("image exceeds max_image_pixels", 413)
                            pixels = image.convert("RGB")
                            digest = hashlib.sha256(
                                str(pixels.size).encode() + pixels.tobytes()
                            ).digest()
                    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                        raise RequestError(f"cannot decode image: {image_id}") from None
                    preprocessing_ms = (time.perf_counter() - start) * 1000
                    start = time.perf_counter()
                    # Deterministic simulated encoding from real image bytes.
                    stop.wait(min(cfg.mock_encode_ms / 1000, remaining(deadline)))
                    remaining(deadline)
                    tensor = torch.tensor(list(digest), dtype=torch.float32).reshape(FEATURE_SHAPE)
                    transfer_id = f"{key}:{index}"
                    store.put(transfer_id, key, tensor, deadline)
                    counters["encoded_images"] += 1
                    features.append({
                        "transfer_id": transfer_id, "image_id": image_id,
                        "shape": list(FEATURE_SHAPE), "dtype": "float32",
                        "checksum": tensor_checksum(tensor),
                        "preprocessing_ms": preprocessing_ms,
                        "encoding_ms": (time.perf_counter() - start) * 1000,
                    })
            except Exception:
                store.release(key)
                raise
            return jsonify(features=features, simulated_encoding=True)

        @app.post("/transfer")
        def send_feature():
            data = json_body()
            key, deadline = envelope(data)
            feature = data.get("feature")
            transfer_id = descriptor(feature, key)
            tensor = store.pop(transfer_id, key)
            remaining(deadline)
            start = time.perf_counter()
            transport.send(transfer_id, tensor)
            counters["sent_features"] += 1
            return jsonify(transfer_id=transfer_id, gloo_send_ms=(time.perf_counter() - start) * 1000)

    else:
        @app.post("/consume")
        def receive_feature():
            data = json_body()
            key, deadline = envelope(data)
            feature = data.get("feature")
            transfer_id = descriptor(feature, key)
            start = time.perf_counter()
            tensor = transport.recv(transfer_id)
            elapsed = (time.perf_counter() - start) * 1000
            if tensor_checksum(tensor) != feature["checksum"]:
                fatal.set()
                raise RequestError("feature checksum mismatch", 502)
            # put checks cancellation/deadline again AFTER the blocking receive.
            store.put(transfer_id, key, tensor, deadline)
            counters["received_features"] += 1
            return jsonify(transfer_id=transfer_id, gloo_recv_ms=elapsed, verified=True)

        @app.post("/generate")
        def generate():
            data = json_body()
            key, deadline = envelope(data)
            start = time.perf_counter()
            verified = []
            features = data.get("features", [])
            try:
                for feature in features:
                    remaining(deadline)
                    transfer_id = descriptor(feature, key)
                    tensor = store.pop(transfer_id, key)
                    if tensor_checksum(tensor) != feature["checksum"]:
                        raise RequestError("stored feature checksum mismatch", 502)
                    verified.append({**feature, "verified": True, "sum": tensor.sum().item()})
                prompt = data.get("prompt", "")
                image_summary = ",".join(f"{f['image_id']}:{f['checksum'][:12]}" for f in verified)
                content = f"[MOCK] prompt={prompt}; images=[{image_summary}]"
                return jsonify(
                    content=content, features=verified, simulated_generation=True,
                    mock_generation_ms=(time.perf_counter() - start) * 1000,
                    prefill_ms=0.0, decode_ms=0.0,
                )
            finally:
                store.release(key)

    return app, store


def worker_main(role, config_dict, stop, fatal):
    cfg = LearningConfig(**config_dict)
    server = None
    try:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{cfg.gloo_port}",
            rank=0 if role == "image" else 1, world_size=2,
            timeout=timedelta(seconds=cfg.transfer_timeout_s),
        )
        app, store = create_worker_app(role, cfg, stop, fatal)
        port = cfg.image_port if role == "image" else cfg.text_port
        server = make_server(cfg.host, port, app, threaded=True)
        server.timeout = 0.2
        print(f"[single_gpu_mock] {role} ready rank={dist.get_rank()} pid={os.getpid()} port={port}", flush=True)
        while not stop.is_set() and not fatal.is_set():
            server.handle_request()
            store.stats()  # Reap expired features even without incoming requests.
    finally:
        if server:
            server.server_close()
        if dist.is_initialized():
            dist.destroy_process_group()
