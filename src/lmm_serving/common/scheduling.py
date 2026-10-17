from __future__ import annotations

import logging
import threading
from typing import Any

import pandas as pd
import requests

# Re-export the original routing API for existing callers.
from .routing import (
    LeastLoadedScheduler,
    Node,
    RoundRobinScheduler,
    Scheduler,
    make_scheduler,
)

# Per-request metrics schema, shared by both cluster managers (keyed by req_id).
REQUEST_METRICS_COLUMNS = [
    "req_id",
    "vllm_req_id",
    "timestamp",
    "num_images",
    "n_context_tokens",
    "n_generated_tokens",
    "num_image_tokens",
    "cluster_manager_waiting_time",
    "total_image_processing_time",
    "total_llm_backend_processing_time",
    "e2e_time",
    "http_code",
]


def request_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUEST_METRICS_COLUMNS).set_index("req_id")


def record_request_metrics(
    df: pd.DataFrame,
    lock: threading.Lock,
    req_id: str,
    vllm_req_id: str,
    timestamp: str | float = "",
    num_images: int = 0,
    n_context_tokens: int = 0,
    n_generated_tokens: int = 0,
    num_image_tokens: int = 0,
    cluster_manager_waiting_time: float = 0,
    total_image_processing_time: float = 0,
    total_llm_backend_processing_time: float = 0,
    e2e_time: float = 0,
    http_code: int = 200,
) -> None:
    row = {
        "vllm_req_id": vllm_req_id,
        "timestamp": timestamp,
        "num_images": num_images,
        "n_context_tokens": n_context_tokens,
        "n_generated_tokens": n_generated_tokens,
        "num_image_tokens": num_image_tokens,
        "cluster_manager_waiting_time": cluster_manager_waiting_time,
        "total_image_processing_time": total_image_processing_time,
        "total_llm_backend_processing_time": total_llm_backend_processing_time,
        "e2e_time": e2e_time,
        "http_code": http_code,
    }
    with lock:
        df.loc[req_id] = row


def num_pending_tokens(
    n_text_tokens: int | str, n_image_tokens: int | str, *, decoder_only: bool
) -> int:
    return int(n_text_tokens) + (int(n_image_tokens) if decoder_only else 0)


def guarded_post(
    logger: logging.Logger, url: str, payload: dict[str, Any], timeout_s: float = 600,
) -> requests.Response:
    try:
        return requests.post(url, json=payload, timeout=timeout_s)
    except requests.Timeout:
        logger.warning("Request to %s timed out", url)
        resp = requests.Response()
        resp.status_code = 504
        resp._content = b'{"error":"downstream timeout"}'
        return resp
    except requests.RequestException:
        logger.exception("Connection to %s failed", url)
        resp = requests.Response()
        resp.status_code = 503
        resp._content = b"{}"
        return resp
