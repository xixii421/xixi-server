from __future__ import annotations

from typing import Any

from flask import Flask, request, jsonify, Response
import requests
import threading
import time
import os
import json
import sys
import yaml
import pandas as pd
import signal

from PIL import Image

from lmm_serving import config as cfg
from lmm_serving.common.logging_config import get_logger
from lmm_serving.common import scheduling
from lmm_serving.common import autoscaling
from lmm_serving.common.vision_tokens import vision_tokens
from lmm_serving.common.resources import ProtocolError
from lmm_serving.modserve.pipeline import ImagePipeline

logger = get_logger("modserve.cluster_manager")

CLUSTER_MANAGER_IP = sys.argv[1]  # 127.0.0.1
CLUSTER_MANAGER_PORT = sys.argv[2]  # 7070
METRICS_FILE = sys.argv[3]
# Optional 4th arg: cluster config filename inside configs/<setup>/.
NODES_CONFIG = (
    sys.argv[4]
    if len(sys.argv) > 4
    else os.environ.get("NODES_CONFIG", "local.yaml")
)
IMAGE_METRICS_FILE = (
    METRICS_FILE[: METRICS_FILE.rfind("/") + 1] + "image_node_metrics.csv"
)

# create an empty file for image encoding metrics, overwrite if exists
metrics_df = scheduling.request_metrics_df()

image_metrics_df = pd.DataFrame(
    columns=[
        "transfer_id",
        "req_id",
        "image_index",
        "image_id",
        "num_image_tokens",
        "waiting_time",
        "preprocessing_time",
        "encoding_time",
        "e2e_time",
    ]
)
image_metrics_df = image_metrics_df.set_index("transfer_id")

# text node : [image_nodes]
modserve_text_nodes = []
modserve_image_nodes = []

image_node_ranks = {}

with open(f"{os.environ['CLUSTER_CONFIG_DIR']}/{NODES_CONFIG}") as f:
    modserve_config = yaml.safe_load(f)
    for modserve_text_node in modserve_config["modserve_text_nodes"]:
        modserve_text_nodes.append(modserve_text_node["text_node_address"])

    for modserve_image_node in modserve_config["modserve_image_nodes"]:
        modserve_image_nodes.append(modserve_image_node["image_node_address"])
        image_node_ranks[modserve_image_node["image_node_address"]] = (
            modserve_image_node["image_node_rank"]
        )

RECV_RANK = len(modserve_image_nodes)
text_recv_rank = {addr: RECV_RANK + j for j, addr in enumerate(modserve_text_nodes)}

logger.info(
    "Modserve cluster_manager starting: listen=%s:%s config=%s text_nodes=%d image_nodes=%d",
    CLUSTER_MANAGER_IP,
    CLUSTER_MANAGER_PORT,
    NODES_CONFIG,
    len(modserve_text_nodes),
    len(modserve_image_nodes),
)
logger.debug(
    "Loaded cluster config: text_nodes=%s image_nodes=%s recv_rank=%s image_node_ranks=%s",
    modserve_text_nodes,
    modserve_image_nodes,
    RECV_RANK,
    image_node_ranks,
)

DATA_LOCK = threading.Lock()
METRICS_FILE_LOCK = threading.Lock()
IMAGE_METRICS_DF_LOCK = threading.Lock()

app = Flask(__name__)


def _post(url: str, payload: dict[str, Any]) -> requests.Response:
    return scheduling.guarded_post(logger, url, payload, float(cfg.MODSERVE_REQUEST_TIMEOUT_S))


MODEL_ARCH = "MLLAMA" if "Llama" in cfg.MODEL else "INTERNVL"
SCHEDULER = getattr(cfg, "SCHEDULER", "round_robin")
TEXT_SCHEDULER = getattr(cfg, "TEXT_SCHEDULER", "round_robin")
INSTANCE_PRIORITY = getattr(cfg, "INSTANCE_SCHEDULING", "fcfs").lower() == "priority"
DECODER_ONLY = MODEL_ARCH == "INTERNVL"  # CroAttn (MLLAMA) vs DecOnly (InternVL)
TEXT_IMG_WEIGHT = float(getattr(cfg, "AUTOSCALE_TEXT_IMG_WEIGHT", 0.0) or 0.0)
if TEXT_IMG_WEIGHT <= 0.0:
    TEXT_IMG_WEIGHT = 1.0 if DECODER_ONLY else 0.35

text_node_scheduler = scheduling.make_scheduler(TEXT_SCHEDULER, modserve_text_nodes)
image_node_scheduler = scheduling.make_scheduler(SCHEDULER, modserve_image_nodes)
logger.info(
    "Routing schedulers: image=%s text=%s instance_priority=%s (model_arch=%s)",
    SCHEDULER,
    TEXT_SCHEDULER,
    INSTANCE_PRIORITY,
    MODEL_ARCH,
)


def _text_load(n_text_tokens: int, n_image_tokens: int) -> int:
    return int(n_text_tokens + TEXT_IMG_WEIGHT * n_image_tokens)


# Autoscaling stuff
AUTOSCALE = getattr(cfg, "AUTOSCALE", False)
AUTOSCALE_INTERVAL_S = getattr(cfg, "AUTOSCALE_INTERVAL_S", 30)
AUTOSCALE_WINDOW_S = getattr(cfg, "AUTOSCALE_WINDOW_S", 60)

image_load_window = autoscaling.LoadWindow(AUTOSCALE_WINDOW_S)
text_load_window = autoscaling.LoadWindow(AUTOSCALE_WINDOW_S)
autoscaler = None  # started in __main__ when AUTOSCALE is on


def _load_profile_mc() -> tuple[float, float]:
    path = f"{cfg.REPO_ROOT}/profiles/{cfg.MODEL.split('/')[-1]}/profile.json"
    enc_mc = bak_mc = 0.0
    try:
        with open(path) as f:
            prof = json.load(f)
        enc_mc = float(prof["encoder"]["m_c_per_s"])
        bak_mc = float(prof["backend"]["m_c_per_s"])
    except (OSError, KeyError, ValueError) as e:
        logger.warning(
            "No usable profile at %s (%s); autoscaler will keep all instances active",
            path,
            e,
        )
    enc_override = float(getattr(cfg, "AUTOSCALE_IMAGE_MC", 0) or 0)
    bak_override = float(getattr(cfg, "AUTOSCALE_TEXT_MC", 0) or 0)
    if enc_override > 0:
        enc_mc = enc_override
    if bak_override > 0:
        bak_mc = bak_override
    logger.info("autoscaler M_C: encoder=%.0f backend=%.0f tok/s", enc_mc, bak_mc)
    return enc_mc, bak_mc


def _image_weight(image_id: str) -> int:

    if MODEL_ARCH == "MLLAMA":
        return vision_tokens(MODEL_ARCH, 0, 0)
    try:
        with Image.open(f"{cfg.IMAGE_DIR}/{image_id}") as im:
            wh = im.size  # (w, h) from header, no decode
    except Exception as e:
        logger.warning(
            "Could not read dims for image_id=%s (%s); using single-tile weight",
            image_id,
            e,
        )
        return vision_tokens(MODEL_ARCH, 1, 1)
    return vision_tokens(MODEL_ARCH, wh[0], wh[1])


def populate_metrics_df(req_id: str, vllm_req_id: str, **kwargs: Any) -> None:
    scheduling.record_request_metrics(
        metrics_df, METRICS_FILE_LOCK, req_id, vllm_req_id, **kwargs
    )


def record_pipeline_image(plan, encoded, e2e_ms):
    with IMAGE_METRICS_DF_LOCK:
        image_metrics_df.loc[plan["transfer_id"]] = [
            plan["req_id"], plan["image_index"], plan["image_id"],
            encoded["num_image_tokens"],
            e2e_ms - encoded.get("preprocessing_time", 0) - encoded.get("image_encoding_time", 0),
            encoded.get("preprocessing_time", 0), encoded.get("image_encoding_time", 0), e2e_ms,
        ]


image_pipeline = ImagePipeline(
    image_scheduler=image_node_scheduler, text_scheduler=text_node_scheduler,
    image_ranks=image_node_ranks,
    recv_addresses={address: (f"{address.split(':')[0]}:{int(cfg.TENSOR_TRANSFER_RECV_BASE_PORT) + j}",
                              text_recv_rank[address])
                    for j, address in enumerate(modserve_text_nodes)},
    token_estimator=_image_weight, text_load=_text_load, model_arch=MODEL_ARCH,
    timeout_s=cfg.MODSERVE_REQUEST_TIMEOUT_S,
    max_images=min(int(cfg.MODSERVE_MAX_IMAGES), 8 if MODEL_ARCH == "MLLAMA" else 400),
    encode_workers=cfg.MODSERVE_ENCODE_WORKERS, logger=logger,
    image_metric=record_pipeline_image,
)


@app.route("/generate", methods=["POST"])
def process_request() -> tuple[Response, int]:
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify(error="request must be a JSON object"), 400
    ids = data.get("image_ids", [])
    if (not isinstance(ids, list) or len(ids) > int(cfg.MODSERVE_MAX_IMAGES)
            or any(not isinstance(i, str) or not i for i in ids)):
        return jsonify(error="invalid image_ids or too many images"), 400
    e2e_start_time = time.perf_counter()
    cluster_manager_waiting_time = (
        time.perf_counter() - data.get("perf_time_start", time.perf_counter())
    ) * 1000

    # request id
    req_id = data.get("req_id", "")
    # text prompt
    prompt = data.get("prompt", "")
    n_text_tokens = data.get("num_text_tokens", 0)
    n_generated_tokens = data.get("num_output_tokens", 1)
    timestamp = data.get("timestamp", 0)

    image_ids = data.get("image_ids", [])
    req_type = "IMAGE" if len(image_ids) > 0 else "TEXT"

    logger.debug(
        "Received %s request req_id=%s images=%s", req_type, req_id, len(image_ids)
    )
    if req_type == "TEXT":
        logger.debug(
            "Request ID: %s %s Number of text tokens: %s Text prompt: %s",
            req_id,
            req_type,
            n_text_tokens,
            prompt,
        )
    else:
        logger.debug(
            "Request ID: %s %s Number of text tokens: %s Text prompt: %s Number of images: %s Image IDs: %s",
            req_id,
            req_type,
            n_text_tokens,
            prompt,
            len(image_ids),
            image_ids,
        )

    populate_metrics_df(
        req_id=req_id,
        vllm_req_id="",
        timestamp=timestamp,
        num_images=len(image_ids),
        n_context_tokens=n_text_tokens,
        n_generated_tokens=n_generated_tokens,
    )

    # Text-only requests go straight to a text node.
    if req_type == "TEXT":
        text_load = _text_load(n_text_tokens, 0)
        text_load_window.add(text_load)
        node_ip_to_route = text_node_scheduler.acquire(text_load)
        node_api_url = "http://" + node_ip_to_route + "/generate"
        if INSTANCE_PRIORITY:
            data["priority"] = text_load
        logger.info(
            "Routing %s request req_id=%s to text node %s (load=%d)",
            req_type,
            req_id,
            node_api_url,
            text_load,
        )
        logger.debug("Request payload: %s", data)
        try:
            response = _post(node_api_url, data)
        finally:
            text_node_scheduler.release(node_ip_to_route, text_load)

        if response.status_code != 200:
            logger.error(
                "Text node returned error: req_id=%s status=%s url=%s",
                req_id,
                response.status_code,
                node_api_url,
            )

        try:
            response_json = response.json()
        except ValueError:
            logger.error(
                "Non-JSON response from text node: req_id=%s url=%s status=%s body=%.200s",
                req_id,
                node_api_url,
                response.status_code,
                response.text,
            )
            response_json = {}
        vllm_req_id = response_json.get("vllm_req_id", "")
        content = response_json.get("content", "")
        vllm_processing_time = response_json.get("request_latency_ms", 0.0)

        populate_metrics_df(
            req_id=req_id,
            vllm_req_id=vllm_req_id,
            timestamp=timestamp,
            num_images=0,
            n_context_tokens=n_text_tokens,
            n_generated_tokens=n_generated_tokens,
            num_image_tokens=0,
            cluster_manager_waiting_time=cluster_manager_waiting_time,
            total_image_processing_time=0.0,
            total_llm_backend_processing_time=vllm_processing_time,
            http_code=response.status_code,
            e2e_time=(time.perf_counter() - e2e_start_time) * 1000
            + cluster_manager_waiting_time,
        )

    # The production pipeline reserves Text and Image capacity before encoding.
    elif req_type == "IMAGE":
        arrival_image_tokens = sum(_image_weight(i) for i in image_ids)
        image_load_window.add(arrival_image_tokens)
        text_load_window.add(_text_load(n_text_tokens, arrival_image_tokens))
        if INSTANCE_PRIORITY:
            data["priority"] = _text_load(n_text_tokens, arrival_image_tokens)
        try:
            outcome = image_pipeline.run(data)
        except ProtocolError as error:
            populate_metrics_df(
                req_id=req_id, vllm_req_id="", num_images=len(image_ids),
                n_context_tokens=n_text_tokens, n_generated_tokens=n_generated_tokens,
                http_code=error.status, e2e_time=(time.perf_counter() - e2e_start_time) * 1000,
            )
            return jsonify(req_id=req_id, error=str(error)), error.status
        response_json = outcome.response
        vllm_req_id = response_json.get("vllm_req_id", "")
        content = response_json.get("content", "")
        response = requests.Response()
        response.status_code = 200
        populate_metrics_df(
            req_id=req_id, vllm_req_id=vllm_req_id, timestamp=timestamp,
            num_images=len(image_ids), n_context_tokens=n_text_tokens,
            n_generated_tokens=n_generated_tokens,
            num_image_tokens=sum(i["num_image_tokens"] for i in outcome.images),
            cluster_manager_waiting_time=cluster_manager_waiting_time,
            total_image_processing_time=outcome.image_ms,
            total_llm_backend_processing_time=outcome.backend_ms,
            e2e_time=(time.perf_counter() - e2e_start_time) * 1000,
        )

    response_ = requests.Response()
    response_.status_code = response.status_code
    response_._content = bytes(
        json.dumps({"vllm_req_id": vllm_req_id, "content": content}).encode("utf8")
    )
    return jsonify(response_.json()), response_.status_code


def signal_handler(sig: int, frame: Any) -> None:
    logger.info(
        "Received signal %s; writing metrics to %s and %s, then shutting down",
        signal.Signals(sig).name,
        METRICS_FILE,
        IMAGE_METRICS_FILE,
    )
    global metrics_df
    global image_metrics_df

    os.makedirs(METRICS_FILE[: METRICS_FILE.rfind("/")], exist_ok=True)

    metrics_df = metrics_df.round(2)
    csv_string = metrics_df.to_csv()
    with open(METRICS_FILE, "w+") as f:
        f.write(csv_string)

    image_metrics_df = image_metrics_df.round(2)
    csv_string = image_metrics_df.to_csv()
    with open(IMAGE_METRICS_FILE, "w+") as f:
        f.write(csv_string)

    logger.info(
        "Metrics written: %s (%d rows), %s (%d rows); shutdown complete",
        METRICS_FILE,
        len(metrics_df),
        IMAGE_METRICS_FILE,
        len(image_metrics_df),
    )
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if AUTOSCALE:
        enc_mc, bak_mc = _load_profile_mc()
        autoscale_metrics = (
            METRICS_FILE[: METRICS_FILE.rfind("/") + 1] + "autoscale_metrics.csv"
        )
        autoscaler = autoscaling.Autoscaler(
            pools=[
                {
                    "name": "image",
                    "scheduler": image_node_scheduler,
                    "window": image_load_window,
                    "m_c": enc_mc,
                    "n_max": len(modserve_image_nodes),
                },
                {
                    "name": "text",
                    "scheduler": text_node_scheduler,
                    "window": text_load_window,
                    "m_c": bak_mc,
                    "n_max": len(modserve_text_nodes),
                },
            ],
            interval_s=AUTOSCALE_INTERVAL_S,
            logger=logger,
            metrics_path=autoscale_metrics,
        )
        autoscaler.start()

    logger.info(
        "Cluster manager ready, serving on http://%s:%s/generate (text_nodes=%d, image_nodes=%d, autoscale=%s)",
        CLUSTER_MANAGER_IP,
        CLUSTER_MANAGER_PORT,
        len(modserve_text_nodes),
        len(modserve_image_nodes),
        AUTOSCALE,
    )
    app.run(
        debug=False, host=CLUSTER_MANAGER_IP, port=CLUSTER_MANAGER_PORT, threaded=True
    )
