"""Production image-request orchestration, independent of model loading.

Keeps the configured routing algorithms; reserves chosen nodes before encoding.
Tests exercise this same pipeline with CPU tensors and injected HTTP adapters.
"""
from __future__ import annotations

import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from lmm_serving.common.features import feature_bytes
from lmm_serving.common.resources import ProtocolError, time_left


@dataclass
class PipelineResult:
    response: dict
    images: list
    request_id: str
    image_ms: float
    backend_ms: float


class ImagePipeline:
    def __init__(self, *, image_scheduler, text_scheduler, image_ranks, recv_addresses,
                 token_estimator, text_load, model_arch, timeout_s, max_images=128,
                 encode_workers=8, post=None, logger=None, image_metric=None):
        self.image_scheduler = image_scheduler
        self.text_scheduler = text_scheduler
        self.image_ranks = image_ranks
        self.recv_addresses = recv_addresses  # text host:port -> (recv host:port, rank)
        self.token_estimator = token_estimator
        self.text_load = text_load
        self.hidden_size = 4096 if model_arch == "MLLAMA" else 6144
        self.timeout_s = float(timeout_s)
        self.max_images = int(max_images)
        self.encode_workers = int(encode_workers)
        self.http_post = post or requests.post
        self.logger = logger
        self.image_metric = image_metric

    def _post(self, address, endpoint, payload, deadline):
        try:
            response = self.http_post(
                f"http://{address}{endpoint}",
                json={**payload, "timeout_s": time_left(deadline)},
                timeout=time_left(deadline),
            )
        except requests.Timeout as error:
            raise ProtocolError(f"downstream timeout: {address}{endpoint}", 504) from error
        except requests.RequestException as error:
            raise ProtocolError(f"downstream unavailable: {address}{endpoint}", 503) from error
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
        except ValueError:
            raise ProtocolError("invalid downstream JSON", 502) from None
        if response.status_code != 200:
            raise ProtocolError(data.get("error", "downstream error"), response.status_code)
        return data

    def run(self, data):
        ids = data.get("image_ids", [])
        if (not isinstance(ids, list) or not 1 <= len(ids) <= self.max_images
                or any(not isinstance(i, str) or not i for i in ids)):
            raise ProtocolError("invalid image_ids or too many images")
        deadline = time.monotonic() + self.timeout_s
        owner = uuid.uuid4().hex
        tokens = [int(self.token_estimator(image_id)) for image_id in ids]
        estimated = sum(tokens)
        text_load = self.text_load(int(data.get("num_text_tokens", 0)), estimated)
        text_node = self.text_scheduler.acquire(text_load)
        recv_node, recv_rank = self.recv_addresses[text_node]
        plans = []
        text_attempted = False
        generated = False
        try:
            # Receiver readiness and authoritative Text budget checked BEFORE
            # any image is encoded. No change to node scoring/routing policy.
            self._post(recv_node, "/check_transfer_ready", {}, deadline)
            text_attempted = True
            self._post(text_node, "/reserve_request", {
                "request_id": owner, "image_tokens": estimated,
                "feature_bytes": estimated * self.hidden_size * 2,
            }, deadline)
            for index, (image_id, count) in enumerate(zip(ids, tokens)):
                node = self.image_scheduler.acquire(count)
                plan = {"image_id": image_id, "image_index": index,
                        "transfer_id": f"{owner}:{index}", "request_id": owner,
                        "req_id": data.get("req_id", ""), "node": node, "tokens": count}
                plans.append(plan)
                self._post(node, "/reserve_feature", {
                    "transfer_id": plan["transfer_id"], "request_id": owner,
                    "feature_bytes": count * self.hidden_size * 2,
                }, deadline)

            def encode(plan):
                start = time.monotonic()
                result = self._post(plan["node"], "/generate", {
                    k: v for k, v in plan.items() if k not in {"node", "tokens"}
                }, deadline)
                if result.get("transfer_id") != plan["transfer_id"] or result.get("request_id") != owner:
                    raise ProtocolError("encoder returned another request's feature", 502)
                result["feature_bytes"] = feature_bytes(result["tensor_shape"], result.get("dtype"))
                if result["tensor_shape"][-1] != self.hidden_size or result["dtype"] != "bfloat16":
                    raise ProtocolError("encoder features are incompatible with the text model", 502)
                result["num_image_tokens"] = math.prod(result["tensor_shape"][:-1])
                result["node"] = plan["node"]
                if self.image_metric:
                    self.image_metric(plan, result, (time.monotonic() - start) * 1000)
                return result

            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=min(len(plans), self.encode_workers)) as pool:
                images = list(pool.map(encode, plans))
            image_ms = (time.monotonic() - start) * 1000
            # Actual values can be larger OR smaller than estimates. Resize is
            # atomic and nonblocking; failed growth rolls the request back.
            actual_tokens = sum(i["num_image_tokens"] for i in images)
            self._post(text_node, "/resize_request", {
                "request_id": owner, "image_tokens": actual_tokens,
                "feature_bytes": sum(i["feature_bytes"] for i in images),
            }, deadline)
            payload = {
                **data, "request_id": owner,
                "images": ids,
                "transfer_ids": [i["transfer_id"] for i in images],
                "transfer_dtypes": [i["dtype"] for i in images],
                "image_id_shapes": [i["tensor_shape"] for i in images],
                "source_node_ips": [i["node"] for i in images],
                "src_ranks": [self.image_ranks[i["node"]] for i in images],
                "dest_ranks": [recv_rank] * len(images),
            }
            start = time.monotonic()
            generated = True
            response = self._post(text_node, "/generate", payload, deadline)
            return PipelineResult(response, images, owner, image_ms, (time.monotonic() - start) * 1000)
        finally:
            # Running engine work and leased IPC buffers reject premature
            # cancellation. Their owning service retains the credits until safe.
            cleanup_deadline = time.monotonic() + 5

            def cancel_plan(plan):
                for node in (plan["node"], recv_node):
                    try:
                        self._post(node, "/cancel_feature", {
                            "transfer_id": plan["transfer_id"], "request_id": owner,
                        }, cleanup_deadline)
                    except ProtocolError as error:
                        if self.logger:
                            self.logger.warning("Feature cleanup deferred: %s", error)
                self.image_scheduler.release(plan["node"], plan["tokens"])
            if plans:
                with ThreadPoolExecutor(max_workers=min(len(plans), self.encode_workers)) as pool:
                    list(pool.map(cancel_plan, plans))
            if text_attempted:
                try:
                    self._post(text_node, "/release_request", {"request_id": owner}, time.monotonic() + 2)
                except ProtocolError as error:
                    if self.logger:
                        self.logger.warning("Text reservation retained (generated=%s): %s", generated, error)
            self.text_scheduler.release(text_node, text_load)
