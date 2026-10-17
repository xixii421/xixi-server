from flask import Flask, request, jsonify, g
import requests
import sys
import time
import threading
import subprocess
import json
import os
from datetime import timedelta
import numpy as np

from lmm_serving import config as cfg
from lmm_serving.common.features import FeatureRegistry, transfer_key
from lmm_serving.common.resources import ProtocolError
from lmm_serving.common.tensor_transfer import TorchTransport, TransferService
from lmm_serving.common.logging_config import get_logger
from lmm_serving.common.vision_tokens import (
    calculate_internvl_targets,
    get_internvl_target_ratios,
)

logger = get_logger("modserve.image_node")

HOSTNAME = sys.argv[1]
PORT = sys.argv[2]
TENSOR_PARALLEL = sys.argv[3]
GPU_ID = sys.argv[4]
MODEL = sys.argv[5]

logger.info(
    "Starting modserve image node on %s:%s | model=%s | parallel=%s | GPUs=%s",
    HOSTNAME,
    PORT,
    MODEL,
    TENSOR_PARALLEL,
    GPU_ID,
)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import (
    AutoModel,
    CLIPImageProcessor,
    MllamaVisionModel,
    MllamaProcessor,
)
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_mllama_projector(model_id: str, device, dtype=torch.bfloat16) -> nn.Linear:
    weight_map = json.load(
        open(hf_hub_download(model_id, "model.safetensors.index.json"))
    )["weight_map"]
    keys = ("multi_modal_projector.weight", "multi_modal_projector.bias")
    shards = {}
    for key in keys:
        shards.setdefault(weight_map[key], []).append(key)
    state = {}
    for shard, shard_keys in shards.items():
        loaded = load_file(hf_hub_download(model_id, shard))
        for key in shard_keys:
            state[key.split(".")[-1]] = loaded[key]
    out_dim, in_dim = state["weight"].shape
    projector = nn.Linear(in_dim, out_dim, bias=True).to(device=device, dtype=dtype)
    projector.load_state_dict(state)
    return projector.eval()


DATA_LOCK = threading.Lock()

app = Flask(__name__)

IMG_DIR = cfg.IMAGE_DIR

assert TENSOR_PARALLEL in [
    "TP1",
    "TP2",
    "TP4",
    "TP8",
], f"Invalid parallelism {TENSOR_PARALLEL}"
# MODEL is a full HF repo id; detect the architecture from it.
IS_LLAMA = "Llama" in MODEL
IS_INTERNVL = "InternVL" in MODEL
assert (
    IS_LLAMA or IS_INTERNVL
), f"Unsupported MODEL={MODEL!r} (expected a Llama or InternVL repo id)"

# load the vision encoder and preprocessor
if IS_LLAMA:
    # The Llama vision tower lives in the same repo as the text model.
    model_id = MODEL
    model = (
        MllamaVisionModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    processor = MllamaProcessor.from_pretrained(model_id)
    projector = load_mllama_projector(model_id, model.device)
elif IS_INTERNVL:
    # InternVL uses a separate encoder repo (not derivable from the model id).
    model_id = "OpenGVLab/InternViT-6B-448px-V2_5"
    model = (
        AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .cuda()
        .eval()
    )
    processor = CLIPImageProcessor.from_pretrained("OpenGVLab/InternViT-6B-448px-V2_5")
    vit_hidden_size = 3200
    downsample_ratio = 0.5
    llm_intermediate_size = 29568
    llm_hidden_size = 6144
    mlp = (
        torch.nn.Sequential(
            torch.nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2),
            torch.nn.Linear(
                vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size
            ),
            torch.nn.GELU(),
            torch.nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        .to("cuda:0")
        .to(torch.bfloat16)
    )

    filename = hf_hub_download(MODEL, filename="model-00011-of-00011.safetensors")
    state_dict = load_file(filename)
    prefix_in_file = "mlp1."
    sequential_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix_in_file):
            new_key = key[len(prefix_in_file) :]
            sequential_state_dict[new_key] = value

    mlp.load_state_dict(sequential_state_dict)

else:
    raise ValueError(
        f"Unsupported MODEL={MODEL!r} (expected a Llama or InternVL repo id)"
    )

logger.info("Vision encoder loaded for %s", model_id)


IMAGE_NODE_RANK = int(os.environ["IMAGE_NODE_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
# Image nodes are ranks 0..N_IMAGE-1; text recv nodes are N_IMAGE..WORLD_SIZE-1.
N_IMAGE = int(os.environ["N_IMAGE_NODES"])
TEXT_RANKS = list(range(N_IMAGE, WORLD_SIZE))

FEATURES = FeatureRegistry(
    int(cfg.IMAGE_FEATURE_BYTE_BUDGET), int(cfg.TRANSFER_MAX_FEATURES),
    history_s=float(cfg.MODSERVE_REQUEST_TIMEOUT_S) + 60,
    history_limit=cfg.TRANSFER_HISTORY_LIMIT,
)
_nccl_ready = threading.Event()
_transfer_service = None


def _build_pair_groups():
    groups = {}
    for img in range(N_IMAGE):
        for txt in TEXT_RANKS:
            g = torch.distributed.new_group(ranks=[img, txt], timeout=timedelta(seconds=float(cfg.TENSOR_TRANSFER_TIMEOUT_S)))
            if IMAGE_NODE_RANK in (img, txt):
                groups[(img, txt)] = g
    return groups


def _warmup_pairs(groups):
    dummy = torch.zeros(1, dtype=torch.bfloat16, device="cuda:0")
    for img in range(N_IMAGE):
        for txt in TEXT_RANKS:
            g = groups.get((img, txt))
            if IMAGE_NODE_RANK == img:
                torch.distributed.send(dummy, dst=txt, group=g)
            elif IMAGE_NODE_RANK == txt:
                torch.distributed.recv(torch.empty_like(dummy), src=img, group=g)
            torch.distributed.barrier()


def nccl_send_worker():
    global _transfer_service
    try:
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            rank=IMAGE_NODE_RANK, world_size=WORLD_SIZE,
            timeout=timedelta(seconds=float(cfg.NCCL_INIT_TIMEOUT_S)),
        )
        torch.distributed.barrier()
        groups = _build_pair_groups()
        _warmup_pairs(groups)
        transport = TorchTransport("cuda:0", {peer: groups[(IMAGE_NODE_RANK, peer)] for peer in TEXT_RANKS})
        _transfer_service = TransferService(
            "sender", IMAGE_NODE_RANK, TEXT_RANKS, FEATURES, transport,
            timeout_s=cfg.TENSOR_TRANSFER_TIMEOUT_S, queue_size=cfg.TRANSFER_QUEUE_SIZE,
            history_s=float(cfg.MODSERVE_REQUEST_TIMEOUT_S) + 60,
            history_limit=cfg.TRANSFER_HISTORY_LIMIT,
        )
        _transfer_service.install(app)
        _nccl_ready.set()
    except Exception:
        logger.exception("NCCL sender initialization failed")
        os._exit(1)


@app.before_request
def before_request():
    g.request_start_time = time.perf_counter()


@app.route("/generate", methods=["POST"])
def process_request():
    data = request.get_json()
    image_id = data.get("image_id", None)
    try:
        _transfer_service.health.check()
        transfer_id = transfer_key(data.get("transfer_id"))
        FEATURES.begin(transfer_id, data.get("request_id"), "encoding")
    except ProtocolError as error:
        return jsonify(error=str(error)), error.status
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        with DATA_LOCK:
            FEATURES.check_encoding(transfer_id, data.get("request_id"))
            start_time = time.perf_counter()
            waiting_time = round((time.perf_counter() - g.request_start_time) * 1000, 4)
            with Image.open(f"{IMG_DIR}/{image_id}") as image_file:
                image = image_file.convert("RGB")
            if IS_LLAMA:
                inputs = processor(images=[image], return_tensors="pt").to(model.device)
            elif IS_INTERNVL:
                inputs = image_to_pixel_values_internvl(
                    image, input_size=448, min_num=1, max_num=12, use_thumbnail=True
                )
                inputs = inputs.to(torch.bfloat16).to(model.device)
            img_preprocessing_time_ms = round(
                (time.perf_counter() - start_time) * 1000, 4
            )
            logger.debug(
                "Image preprocessed: %s duration = %s ms",
                image_id,
                img_preprocessing_time_ms,
            )

            start.record()
            with torch.no_grad():
                if IS_LLAMA:
                    outputs = model(**inputs)
                    outputs = projector(outputs.last_hidden_state)
                elif IS_INTERNVL:
                    outputs = model(inputs)
                    outputs = outputs.last_hidden_state
                    outputs = outputs[:, 1:, :]
                    h = w = int(outputs.shape[1] ** 0.5)
                    outputs = outputs.reshape(outputs.shape[0], h, w, -1)
                    outputs = pixel_shuffle(outputs)
                    outputs = mlp(outputs)
            outputs = outputs.contiguous()
            end.record()
            # Publish only after device work completes; cancellation cannot free
            # an encoding reservation while this stream still uses its buffers.
            torch.cuda.synchronize()
            FEATURES.publish(transfer_id, outputs)
    except Exception as e:
        logger.exception("Failed to encode image_id=%s", image_id)
        try:
            torch.cuda.synchronize()
            FEATURES.finish_device_use(transfer_id)
        except Exception:
            _transfer_service.health.fail("encoder CUDA failure; restart required")
        return (
            jsonify(
                {"error": str(e), "image_id": image_id, "req_id": data.get("req_id", 0)}
            ),
            e.status if isinstance(e, ProtocolError) else 500,
        )

    torch.cuda.synchronize()
    img_encoding_time_ms = round(start.elapsed_time(end), 4)
    logger.debug("Image encoded: %s duration = %s ms", image_id, img_encoding_time_ms)

    # create a response that contains the host ip, number of image tokens, request id
    response = requests.Response()
    response.status_code = 200
    response._content = bytes(
        json.dumps(
            {
                "host": HOSTNAME,
                "protocol_version": 2,
                "transfer_id": transfer_id,
                "request_id": data["request_id"],
                "dtype": str(outputs.dtype).split(".")[-1],
                "feature_bytes": outputs.numel() * outputs.element_size(),
                "num_image_tokens": int(np.prod(list(outputs.shape[:-1]))),
                "req_id": data.get("req_id", 0),
                "image_id": image_id,
                "image_encoding_time": img_encoding_time_ms,
                "preprocessing_time": img_preprocessing_time_ms,
                "waiting_time": waiting_time,
                "tensor_shape": list(outputs.shape),
            }
        ),
        "utf-8",
    )

    return jsonify(response.json()), response.status_code


def dynamic_preprocess_internvl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_internvl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def image_to_pixel_values_internvl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_internvl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )

    x = x.permute(0, 2, 1, 3).contiguous()
    return x


if __name__ == "__main__":
    # Join the NCCL world (with the recv rank) before serving requests.
    threading.Thread(target=nccl_send_worker, daemon=True).start()
    _nccl_ready.wait()
    logger.info("Image node ready, serving on http://%s:%s/generate", HOSTNAME, PORT)
    app.run(debug=False, port=int(PORT), host="0.0.0.0", threaded=True)
