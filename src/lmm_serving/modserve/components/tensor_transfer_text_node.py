import base64
import os
import pickle
import json
from datetime import timedelta
import signal
import sys
import threading

import torch
from flask import Flask, request, jsonify
from torch.multiprocessing.reductions import reduce_tensor

from lmm_serving.common.logging_config import get_logger
from lmm_serving import config as cfg
from lmm_serving.common.features import FeatureRegistry
from lmm_serving.common.tensor_transfer import TorchTransport, TransferService

logger = get_logger("modserve.tensor_transfer_text")
app = Flask(__name__)


def parse_transfer_ranks(spec: str) -> dict:
    owned = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if pair:
            g, gpu = pair.split(":")
            owned[int(g)] = int(gpu)
    return owned


OWNED_RANKS = parse_transfer_ranks(os.environ["TRANSFER_RANKS"])
assert len(OWNED_RANKS) == 1, f"recv node owns exactly one rank, got {OWNED_RANKS}"
RANK, GPU = next(iter(OWNED_RANKS.items()))
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
# Image nodes are ranks 0..N_IMAGE-1; text recv nodes are N_IMAGE..WORLD_SIZE-1.
N_IMAGE = int(os.environ["N_IMAGE_NODES"])
IMAGE_RANKS = list(range(N_IMAGE))

FEATURES = FeatureRegistry(
    int(cfg.RECV_FEATURE_BYTE_BUDGET), int(cfg.TRANSFER_MAX_FEATURES),
    history_s=float(cfg.MODSERVE_REQUEST_TIMEOUT_S) + 60,
    history_limit=cfg.TRANSFER_HISTORY_LIMIT,
)
_transfer_service = None


def _ipc_handle(tensor: torch.Tensor) -> str:
    _, args = reduce_tensor(tensor)
    return base64.b64encode(pickle.dumps(args)).decode()


def _build_pair_groups():
    groups = {}
    for img in IMAGE_RANKS:
        for txt in range(N_IMAGE, WORLD_SIZE):
            g = torch.distributed.new_group(ranks=[img, txt], timeout=timedelta(seconds=float(cfg.TENSOR_TRANSFER_TIMEOUT_S)))
            if RANK in (img, txt):
                groups[(img, txt)] = g
    return groups


def _warmup_pairs(groups):
    dummy = torch.zeros(1, dtype=torch.bfloat16, device=f"cuda:{GPU}")
    for img in IMAGE_RANKS:
        for txt in range(N_IMAGE, WORLD_SIZE):
            g = groups.get((img, txt))
            if RANK == img:
                torch.distributed.send(dummy, dst=txt, group=g)
            elif RANK == txt:
                torch.distributed.recv(torch.empty_like(dummy), src=img, group=g)
            torch.distributed.barrier()


def nccl_worker():
    global _transfer_service
    try:
        torch.cuda.set_device(GPU)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            rank=RANK, world_size=WORLD_SIZE,
            timeout=timedelta(seconds=float(cfg.NCCL_INIT_TIMEOUT_S)),
        )
        torch.distributed.barrier()
        groups = _build_pair_groups()
        _warmup_pairs(groups)
        transport = TorchTransport(f"cuda:{GPU}", {p: groups[(p, RANK)] for p in IMAGE_RANKS})
        sources = {int(k): v for k, v in json.loads(os.environ["IMAGE_SOURCE_URLS"]).items()}
        _transfer_service = TransferService(
            "receiver", RANK, IMAGE_RANKS, FEATURES, transport,
            timeout_s=cfg.TENSOR_TRANSFER_TIMEOUT_S, queue_size=cfg.TRANSFER_QUEUE_SIZE,
            source_urls=sources, make_handle=_ipc_handle,
            history_s=float(cfg.MODSERVE_REQUEST_TIMEOUT_S) + 60,
            history_limit=cfg.TRANSFER_HISTORY_LIMIT,
        )
        _transfer_service.install(app)
        _ready.set()
    except Exception:
        logger.exception("NCCL receiver initialization failed")
        os._exit(1)


_ready = threading.Event()


def signal_handler(*_):
    logger.info("Shutting down tensor-transfer RECV node")
    try:
        if _transfer_service:
            _transfer_service.close()
        torch.distributed.destroy_process_group()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    logger.info(
        "Starting tensor-transfer RECV node: rank=%s gpu=%s world=%s master=%s:%s CVD=%s",
        RANK,
        GPU,
        WORLD_SIZE,
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"],
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    threading.Thread(target=nccl_worker, daemon=True).start()
    _ready.wait()  # block until NCCL is up so we don't serve before the world is formed
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app.run(
        debug=False,
        port=int(os.environ.get("TENSOR_TRANSFER_PORT", "5050")),
        host="0.0.0.0",
        threaded=True,
    )
