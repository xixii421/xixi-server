"""Three Idefics3 stages with Gloo control and Gloo/NCCL tensor handoffs."""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import timedelta
from functools import cached_property
import gc
import hashlib
import io
import json
import queue
from pathlib import Path
import tempfile
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.distributed as dist

from .pd_local import LocalPD, ModelSpec, receive_header, send_header, validate_device
from .vision_cache import VisionFeatureCache
from .vpd_transport import TensorChannel, send_kv, receive_kv
from .vpd_router import StageRouter


class LimitError(ValueError):
    status_code = 413


class CapacityError(RuntimeError):
    status_code = 429


@dataclass
class _RequestState:
    request: dict
    deadline: float
    inbox: queue.Queue = field(default_factory=queue.Queue)
    completed: set = field(default_factory=set)
    rejected: bool = False
    queue_wait_ms: float = 0


@dataclass(frozen=True)
class VModelSpec(ModelSpec):
    max_prompt_tokens: int = 1024
    max_images: int = 2
    max_image_bytes: int = 2*1024*1024
    max_image_pixels: int = 1024*1024
    max_feature_bytes: int = 4*1024*1024
    vision_cache_bytes: int = 0
    stage_devices: tuple[str, str, str] | None = None
    stage_pools: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None
    transport: str = "gloo"
    verify_payload: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name in ("max_images", "max_image_bytes", "max_image_pixels", "max_feature_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.vision_cache_bytes) is not int or self.vision_cache_bytes < 0:
            raise ValueError("vision_cache_bytes must be a nonnegative integer")
        if self.transport not in ("gloo", "nccl"):
            raise ValueError("transport must be gloo or nccl")
        if type(self.verify_payload) is not bool:
            raise ValueError("verify_payload must be boolean")
        if self.stage_pools is not None:
            if self.stage_devices is not None:
                raise ValueError("use stage_pools or stage_devices, not both")
            if (not isinstance(self.stage_pools, tuple) or len(self.stage_pools) != 3 or
                    any(not isinstance(pool, tuple) or not pool for pool in self.stage_pools)):
                raise ValueError("stage_pools must contain nonempty V/P/D device tuples")
            for pool in self.stage_pools:
                for device in pool:
                    validate_device(device)
        if self.stage_devices is not None:
            if not isinstance(self.stage_devices, tuple) or len(self.stage_devices) != 3:
                raise ValueError("stage_devices must be a tuple of V/P/D devices")
            for device in self.stage_devices:
                validate_device(device)
        if self.transport == "nccl":
            if self.stage_devices is None and self.stage_pools is None:
                object.__setattr__(self, "stage_devices", ("cuda:0", "cuda:1", "cuda:2"))
            if len(set(self.devices)) != self.worker_count or any(not d.startswith("cuda:") for d in self.devices):
                raise ValueError("NCCL requires distinct CUDA devices, one per worker")

    @property
    def pools(self):
        if self.stage_pools is not None:
            return self.stage_pools
        return tuple((device,) for device in (self.stage_devices or (self.device,)*3))

    @property
    def devices(self):
        return tuple(device for pool in self.pools for device in pool)

    @property
    def worker_count(self):
        return len(self.devices)

    @property
    def worker_stages(self):
        # global rank 按 V 池、P 池、D 池的顺序连续分配，不再把 rank 当作阶段。
        return tuple(stage for stage,pool in enumerate(self.pools) for _ in pool)

    def ranks_for(self,stage):
        return tuple(rank for rank,role in enumerate(self.worker_stages) if role == stage)

    def for_rank(self, rank):
        return replace(self, device=self.devices[rank])

    @property
    def scope(self):
        payload = ("NCCL CUDA payloads" if self.transport == "nccl" else "host-staged Gloo payloads")
        return f"single-host V/P/D; Gloo control; {payload}; bounded concurrent stage routing"

    def check_hardware(self):
        """Reject unavailable cards/backends before spawning expensive workers."""
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("VPD requires a PyTorch build with Gloo")
        indices = [int(d.split(":")[1]) for d in self.devices if d.startswith("cuda:")]
        if indices and (not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count()):
            raise RuntimeError(f"requested V/P/D devices {self.devices}; only {torch.cuda.device_count()} CUDA GPU(s) visible")
        if self.transport == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("NCCL transport requires a CUDA PyTorch build with NCCL on Linux/WSL")

    @cached_property
    def _full_config(self):
        if self.model_path:
            config = json.loads((Path(self.model_path)/"config.json").read_text())
            if config.get("model_type") != "idefics3":
                raise ValueError("VPD supports Idefics3/SmolVLM feature-insertion models")
            return config
        return {"model_type": "idefics3", "image_token_id": 63, "scale_factor": 2,
                "text_config": {"model_type": "llama", "vocab_size": 64, "hidden_size": 32,
                    "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 512,
                    "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
                "vision_config": {"hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 4, "image_size": 8,
                    "patch_size": 2, "num_channels": 3}}

    def full_config(self):
        """启动时的配置快照；运行期间不重新读取磁盘，调用方按只读使用。"""
        return self._full_config

    def config(self):
        return self.full_config()["text_config"]

    @cached_property
    def _signature(self):
        source = str(Path(self.model_path).resolve()) if self.model_path else "tiny-idefics3"
        return hashlib.sha256(json.dumps([source, self.full_config(), self.dtype, self.seed],
                                         sort_keys=True).encode()).hexdigest()

    def image_seq_len(self):
        config = self.full_config()
        vision = config["vision_config"]
        return (vision["image_size"]//vision["patch_size"]//config["scale_factor"])**2


@dataclass(frozen=True)
class PreparedVPDRequest:
    """入口验证后的内部请求；HTTP 和直接调用共用同一条执行路径。"""
    messages: list
    images: list[bytes]
    new_tokens: int


def prepare_request(spec, messages, new_tokens):
    spec.validate_generation(new_tokens)
    normalized, images = validate_messages(spec, messages)
    return PreparedVPDRequest(normalized, images, new_tokens)


def validate_messages(spec, messages):
    """Bound and snapshot requests; only inline PNG/JPEG avoids remote fetching."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be nonempty")
    normalized, images = [], []
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")):
            raise ValueError("invalid message")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise ValueError("invalid message content")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("invalid content block")
            if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                if len(images) >= spec.max_images:
                    raise LimitError("too many images")
                info = part["image_url"]
                if not isinstance(info, dict) or set(info) != {"url"} or not isinstance(info["url"], str):
                    raise ValueError("invalid image_url")
                prefix, separator, encoded = info["url"].partition(",")
                if not separator or prefix not in ("data:image/png;base64", "data:image/jpeg;base64"):
                    raise ValueError("only inline PNG/JPEG data URLs are supported")
                if len(encoded) > 4*((spec.max_image_bytes+2)//3):
                    raise LimitError("image exceeds byte budget")
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > spec.max_image_bytes:
                    raise LimitError("image exceeds byte budget")
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format not in ("PNG", "JPEG"):
                            raise ValueError("unsupported image format")
                        if image.width*image.height > spec.max_image_pixels:
                            raise LimitError("image pixel budget exceeded")
                        image.verify()
                except (OSError, SyntaxError, Image.DecompressionBombError) as error:
                    raise ValueError("invalid image") from error
                parts.append({"type": "image", "index": len(images)})
                images.append(raw)
            else:
                raise ValueError("unsupported multimodal content block")
        normalized.append({"role": message["role"], "content": parts})
    feature_bytes = len(images)*spec.image_seq_len()*spec.config()["hidden_size"]*(4 if spec.dtype=="float32" else 2)
    if feature_bytes > spec.max_feature_bytes:
        raise LimitError("visual feature budget exceeded before encoding")
    return normalized, images


def load_processor(spec):
    if not spec.model_path:
        return None
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(spec.model_path, local_files_only=True, use_fast=False)
    processor.image_processor.do_image_splitting = False
    edge = spec.full_config()["vision_config"]["image_size"]
    processor.image_processor.size = {"longest_edge": edge}
    processor.image_processor.max_image_size = {"longest_edge": edge}
    if processor.image_seq_len != spec.image_seq_len():
        raise ValueError("processor image sequence length differs from model")
    return processor


def prepare_inputs(spec, processor, messages, raw_images):
    images = []
    for raw in raw_images:
        with Image.open(io.BytesIO(raw)) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    if processor:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        kwargs = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kwargs["images"] = [images]
        inputs = dict(processor(**kwargs))
    else:
        tokens = [1]
        for message in messages:
            for part in message["content"]:
                if part["type"] == "image":
                    tokens += [spec.full_config()["image_token_id"]]*spec.image_seq_len()
                else:
                    tokens += [3+(byte % 59) for byte in part["text"].encode()]
        inputs = {"input_ids": torch.tensor([tokens], dtype=torch.long)}
        if images:
            edge = spec.full_config()["vision_config"]["image_size"]
            pixels = [torch.from_numpy(np.array(image.resize((edge, edge)), copy=True)).permute(2,0,1)
                      for image in images]
            inputs["pixel_values"] = torch.stack(pixels).unsqueeze(0).float()/127.5-1
            inputs["pixel_attention_mask"] = torch.ones((1,len(images),edge,edge), dtype=torch.bool)
    tokens = inputs["input_ids"][0].tolist()
    count = tokens.count(spec.full_config()["image_token_id"])
    if count != len(images)*spec.image_seq_len():
        raise ValueError("image placeholders and ordered visual tokens do not align")
    return inputs


def load_full_model(spec):
    from transformers import Idefics3Config, Idefics3ForConditionalGeneration
    torch.set_num_threads(spec.threads)
    torch.manual_seed(spec.seed)
    # Transformers 构造器可能补写嵌套配置，复制一次以保持运行时配置快照不变。
    config = Idefics3Config(**deepcopy(spec.full_config()))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    if spec.model_path:
        model = Idefics3ForConditionalGeneration.from_pretrained(spec.model_path, config=config,
                    local_files_only=True, torch_dtype=getattr(torch,spec.dtype), attn_implementation="eager")
    else:
        model = Idefics3ForConditionalGeneration(config).to(getattr(torch,spec.dtype))
    return model.eval()


def load_stage(spec, rank):
    model = load_full_model(spec)
    if rank == 0:
        modules = torch.nn.ModuleDict({"vision": model.model.vision_model,
                                      "connector": model.model.connector})
        del model
        gc.collect()
        return modules.to(spec.device).eval()
    model.model.vision_model = None
    model.model.connector = None
    gc.collect()
    return model.to(spec.device).eval()


@torch.inference_mode()
def encode_vision(spec, modules, inputs):
    pixels = inputs["pixel_values"][0].to(spec.device, dtype=getattr(torch,spec.dtype))
    mask = inputs.get("pixel_attention_mask")
    mask = mask[0].to(spec.device) if mask is not None else torch.ones(
        (pixels.shape[0],*pixels.shape[-2:]), device=spec.device, dtype=torch.bool)
    patch = spec.full_config()["vision_config"]["patch_size"]
    patch_mask = mask.unfold(1,patch,patch).unfold(2,patch,patch).sum((-1,-2)) > 0
    hidden = modules["vision"](pixel_values=pixels, patch_attention_mask=patch_mask).last_hidden_state
    return modules["connector"](hidden)


def encode_cached(spec, modules, inputs, image_digests, cache):
    """Reuse image-level outputs independently of question and image order."""
    if not cache.capacity:
        started=time.perf_counter()
        output=encode_vision(spec,modules,inputs).detach().contiguous()
        # 这里只负责编码与计时；通信设备的转换统一由 channel.prepare 完成。
        if spec.device.startswith("cuda"):
            torch.cuda.synchronize(spec.device)
        return output,{"cache_hits":0,"cache_misses":len(image_digests),
                       "vision_images_encoded":len(image_digests),"intra_request_reuses":0,
                       "vision_encoder_batch_images":len(image_digests),"vision_padding_images":0,
                       "vision_compute_ms":(time.perf_counter()-started)*1000,
                       "cache_evictions":0,"cache_resident_bytes":0,"cache_entries":0}
    # BF16 编码的舍入可能随 batch 形状变化；同一图片的单图/双图结果不能混用。
    namespace=spec.signature()+f":exif_rgb_no_split_fixed_longest_edge_v2:batch{len(image_digests)}:"
    keys=[namespace+digest for digest in image_digests]
    values={}
    misses={}
    hits=0
    duplicate_misses=0
    for index,key in enumerate(keys):
        if key in values:
            # A repeated resident hit is still a hit at this image position.
            hits+=1
            continue
        if key in misses:
            duplicate_misses+=1
            continue
        value=cache.get(key)
        if value is None:
            misses[key]=index
        else:
            hits+=1
            values[key]=value
    compute_ms=0.0
    evictions_before=cache.evictions
    if misses:
        indices=list(misses.values())
        # 部分命中、重复图片去重后，仍保留完整请求的 encoder batch 大小。
        # 补齐项只参与计算，不写缓存；显式记录补齐成本，不能把它算作省掉的计算。
        encoder_indices=indices+[indices[0]]*(len(image_digests)-len(indices))
        subset={"pixel_values":inputs["pixel_values"][:,encoder_indices]}
        if "pixel_attention_mask" in inputs:
            subset["pixel_attention_mask"]=inputs["pixel_attention_mask"][:,encoder_indices]
        started=time.perf_counter()
        encoded=encode_vision(spec,modules,subset)[:len(indices)].detach().cpu().contiguous()
        compute_ms=(time.perf_counter()-started)*1000
        for offset,key in enumerate(misses):
            values[key]=encoded[offset]
            cache.put(key,encoded[offset])
    output=torch.stack([values[key] for key in keys]).contiguous()
    state=cache.snapshot()
    return output,{"cache_hits":hits,"cache_misses":len(image_digests)-hits,
                   "vision_images_encoded":len(misses),"intra_request_reuses":duplicate_misses,
                   "vision_encoder_batch_images":len(image_digests) if misses else 0,
                   "vision_padding_images":len(image_digests)-len(misses) if misses else 0,
                   "vision_compute_ms":compute_ms,"cache_evictions":cache.evictions-evictions_before,
                   "cache_resident_bytes":state["resident_bytes"],"cache_entries":state["entries"]}


@torch.inference_mode()
def language_forward(spec, model, tokens, *, cache=None, features=None):
    return model(input_ids=torch.tensor([tokens],device=spec.device), past_key_values=cache,
                 image_hidden_states=features, use_cache=True, logits_to_keep=1)


def reference_generate(spec, messages, new_tokens=8):
    """Unsplit VLM: pixel_values enter the complete pretrained model directly."""
    normalized, images = validate_messages(spec,messages)
    inputs = prepare_inputs(spec,load_processor(spec),normalized,images)
    spec.validate(inputs["input_ids"][0].tolist(),new_tokens)
    model = load_full_model(spec).to(spec.device)
    inputs = {k:v.to(spec.device, dtype=getattr(torch,spec.dtype)) if k == "pixel_values"
              else v.to(spec.device) for k,v in inputs.items()}
    with torch.inference_mode():
        result = model(**inputs,use_cache=True,logits_to_keep=1)
        output = [int(result.logits[0,-1].argmax())]
        for _ in range(new_tokens-1):
            result = language_forward(spec,model,[output[-1]],cache=result.past_key_values)
            output.append(int(result.logits[0,-1].argmax()))
    del model,result,inputs
    gc.collect()
    if spec.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def feature_header(spec, request, tensor, channel):
    """V -> P 的元数据：绑定请求、文本和有序图片，描述特征形状及可选摘要。"""
    return {"version": 1,"request_id":request["request_id"],"sequence":request["sequence"],
            "signature":spec.signature(),"context_digest":request["context_digest"],
            "route":request.get("route"),
            "shape":list(tensor.shape),"dtype":spec.dtype,"digest":channel.digest(tensor),
            "verify_payload":channel.verify_payload,
            "image_digests":request["image_digests"],
            "prompt_digest":hashlib.sha256(json.dumps(request["tokens"]).encode()).hexdigest()}


def receive_features(spec,request,channel):
    """P 接收视觉特征：先验头和容量，再收张量，确认可用后向 V 发 ACK。"""
    source = request.get("route",{}).get("vision",0)
    header = receive_header(source)
    # 形状取自本地已知的图片数、每图 token 数和语言隐藏维度，不由对端随意决定。
    shape = [request["image_count"],spec.image_seq_len(),spec.config()["hidden_size"]]
    expected = {"version":1,"request_id":request["request_id"],"sequence":request["sequence"],
                "signature":spec.signature(),"context_digest":request["context_digest"],
                "route":request.get("route"),
                "shape":shape,"dtype":spec.dtype,"image_digests":request["image_digests"],
                "verify_payload":channel.verify_payload,
                "prompt_digest":hashlib.sha256(json.dumps(request["tokens"]).encode()).hexdigest()}
    if any(header.get(key) != value for key,value in expected.items()):
        raise ValueError("visual handoff identity, order or shape mismatch")
    if np.prod(shape)*(4 if spec.dtype == "float32" else 2) > spec.max_feature_bytes:
        raise ValueError("visual feature byte budget exceeded")
    # channel 决定接收在 CPU 还是当前 GPU；stage 再保证张量位于 P 的计算设备。
    tensor = channel.receive(shape,getattr(torch,spec.dtype),source)
    channel.validate_digest(tensor,header.get("digest"))
    tensor = channel.stage(tensor)
    # 只有通过检查且搬运完成才能确认。ACK 携带本次通道序号，走默认 Gloo 组。
    dist.send(torch.tensor([request["sequence"]],dtype=torch.int64),dst=source)
    return tensor,header["digest"]


def vpd_worker(rank,spec,store,control,events,timeout):
    """一个 worker 对应阶段池内的一个副本，rank 由配置布局决定。

    注意三类通信的区别：
    1. control/events：主进程与 worker 之间的 multiprocessing.Queue。
    2. 默认 Gloo 组：worker 之间的 JSON 头和 CPU ACK。
    3. channel：worker 之间的大张量，按配置使用 CPU/Gloo 或 CUDA/NCCL。
    """
    def publish(event):
        events.put({**event,"rank":rank})

    channel = None
    try:
        # rank 是进程编号，不等于 GPU 编号；实际映射由 stage_devices 决定。
        spec = spec.for_rank(rank)
        stage = spec.worker_stages[rank]
        if spec.device.startswith("cuda:"):
            # 在加载 GPU 权重、创建 NCCL 组之前绑定本进程的当前设备。
            torch.cuda.set_device(spec.device)
        model = load_stage(spec,stage)
        processor = load_processor(spec) if stage == 0 else None
        cache_store=VisionFeatureCache(spec.vision_cache_bytes) if stage==0 else None
        # FileStore 用本机共享文件做启动 rendezvous，不用于传输模型数据。
        # 所有副本都加入组；只有为本次请求选中的 V/P/D 参与载荷交接。
        dist.init_process_group("gloo",init_method="file://"+store,rank=rank,world_size=spec.worker_count,
                                timeout=timedelta(seconds=timeout))
        channel = TensorChannel.start(spec,timeout)
        # ready 表示权重加载和通信预热均已成功，主进程收齐所有副本后才接请求。
        publish({"type":"ready","rank":rank,"device":spec.device,
                    "stage":("vision","prefill","decode")[stage],
                    "transport":spec.transport,"verify_payload":spec.verify_payload})
        while True:
            request = control.get()
            if request is None:
                break
            if stage == 0:
                # V：预处理图片/文本，运行真实视觉编码或复用 CPU 视觉缓存。
                started = time.perf_counter()
                inputs = tensor = None
                try:
                    inputs = prepare_inputs(spec,processor,request["messages"],request["images"])
                    request["tokens"] = inputs["input_ids"][0].tolist()
                    spec.validate(request["tokens"],request["new_tokens"])
                    prepared=time.perf_counter()
                    tensor,cache_stats=encode_cached(spec,model,inputs,request["image_digests"],cache_store)
                    expected_shape = (request["image_count"],spec.image_seq_len(),spec.config()["hidden_size"])
                    if tuple(tensor.shape) != expected_shape or tensor.numel()*tensor.element_size() > spec.max_feature_bytes:
                        raise ValueError("visual output exceeds expected shape or budget")
                except ValueError as error:
                    publish({"type":"reject","request_id":request["request_id"],"error":str(error)})
                    del inputs,tensor,request
                    continue  # 此时还没开始阶段间通信，拒绝本请求后仍可安全接下一次。
                tensor = channel.prepare(tensor)
                header = feature_header(spec,request,tensor,channel)
                # 必须先把展开后的 tokens 通知主进程，再发送特征。
                # 主进程据此向 P/D 下发请求；如果 V 先阻塞等 P 接收，而主进程
                # 还不知道 tokens、未唤醒 P，就会形成相互等待。
                publish({"type":"vision_ready","request_id":request["request_id"],
                            "tokens":request["tokens"],"feature_digest":header["digest"]})
                encoded = time.perf_counter()
                # V -> P：头部(Gloo) -> 特征(channel) -> 接管确认(Gloo)。
                destination = request["route"]["prefill"]
                send_header(header,destination)
                channel.send(tensor,destination)
                ack = torch.zeros(1,dtype=torch.int64)
                dist.recv(ack,src=destination)
                if ack.item() != request["sequence"]:
                    raise ValueError("visual acknowledgement mismatch")
                publish({"type":"vision_handoff","request_id":request["request_id"],
                            "vision_ms":(encoded-started)*1000,
                            "vision_transfer_ms":(time.perf_counter()-encoded)*1000,
                            "feature_bytes":tensor.numel()*tensor.element_size(),
                            "vision_preprocess_ms":(prepared-started)*1000,**cache_stats})
                del tensor,inputs  # 对端已确认接管，才释放本次视觉交接持有的引用。
            elif stage == 1:
                # P：有图就等待 V 的特征；纯文本请求直接绕过视觉通信。
                features,feature_digest = (receive_features(spec,request,channel) if request["image_count"] else (None,None))
                started = time.perf_counter()
                result = language_forward(spec,model,request["tokens"],features=features)
                first = int(result.logits[0,-1].argmax())
                ready_at = time.perf_counter()
                # Prefill 得到“prompt 的 KV”和“首个生成 Token”。先通知客户端首
                # Token，再交接 KV，因此 KV 传输开销还应看首/第二 Token 间隔。
                publish({"type":"token","request_id":request["request_id"],"index":0,"token":first})
                send_kv(channel,spec,request,result,first,ready_at,destination=request["route"]["decode"])
                publish({"type":"handoff","request_id":request["request_id"],
                            "prefill_ms":(ready_at-started)*1000,
                            "handoff_ms":(time.perf_counter()-ready_at)*1000,
                            "prefill_feature_digest":feature_digest})
                del result,features  # send_kv 已等待 D 的 ACK，此时才能释放 P 侧缓存。
            else:
                # D：用收到的 prompt KV 续算。首 Token 已有，不重算完整 prompt。
                cache,header = receive_kv(channel,spec,request,source=request["route"]["prefill"])
                decode_started = time.perf_counter()
                output,times = [header["first_token"]],[header["first_token_ready_s"]]
                for index in range(1,request["new_tokens"]):
                    # 每步仅输入上一个 Token；历史上下文由不断增长的 KV 提供。
                    result = language_forward(spec,model,[output[-1]],cache=cache)
                    cache = result.past_key_values
                    output.append(int(result.logits[0,-1].argmax()))
                    times.append(time.perf_counter())
                    publish({"type":"token","request_id":request["request_id"],"index":index,"token":output[-1]})
                decode_finished = time.perf_counter()
                publish({"type":"result","request_id":request["request_id"],"token_ids":output,
                            "decode_started_s":decode_started,"decode_finished_s":decode_finished,
                            "token_ready_times_s":times,"decode_input_lengths":[1]*(len(output)-1),
                            "received_cache_tokens":len(request["tokens"]),
                            "decode_context_digest":header["context_digest"]})
                del cache
                if len(output)>1:
                    del result
            del request
    except BaseException as error:
        # 一旦开始阶段间通信，异常可能让另一端停在 recv/send。
        # 通知主进程统一关闭全部 worker，避免在状态不确定的通道上继续服务。
        publish({"type":"error","rank":rank,"error":f"{type(error).__name__}: {error}"})
    finally:
        if dist.is_initialized():
            try:
                if channel is not None:
                    # 各 worker 按相同顺序清理：先数据子组，再默认控制组。
                    channel.close()
            finally:
                dist.destroy_process_group()


class LocalVPD(LocalPD):
    def __init__(self,spec=VModelSpec(),*,max_inflight=4,**kwargs):
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        super().__init__(spec,**kwargs)
        self.controls.extend(self.context.Queue() for _ in range(spec.worker_count-len(self.controls)))
        self.max_inflight = max_inflight
        self._condition = threading.Condition()
        self._stopped = threading.Event()
        self._pump_thread = None
        self._requests = {}
        self._pending = deque()
        self._router = StageRouter(spec)
        self._stop_error = None
        self.processor = load_processor(spec)
        self.scope = spec.scope
        self.worker_info = []

    def __enter__(self):
        # 主进程不加入通信组，只负责预检、启动与调度。
        try:
            self.spec.check_hardware()
            self._directory = tempfile.TemporaryDirectory(prefix="modserve-vpd-")
            store = str(Path(self._directory.name)/"gloo-store")
            for rank in range(self.spec.worker_count):
                process = self.context.Process(target=vpd_worker,args=(rank,self.spec,store,
                    self.controls[rank],self.events,max(self.startup_timeout_s,self.request_timeout_s)))
                self.processes.append(process)
                process.start()
            ready = set()
            deadline = time.monotonic()+self.startup_timeout_s
            while len(ready)<self.spec.worker_count:
                event = self._event(deadline)
                if (event["type"] != "ready" or event.get("rank") in ready or
                        event.get("rank") not in range(self.spec.worker_count) or
                        event.get("device") != self.spec.devices[event["rank"]] or
                        event.get("stage") != ("vision","prefill","decode")[self.spec.worker_stages[event["rank"]]] or
                        event.get("transport") != self.spec.transport or
                        event.get("verify_payload") is not self.spec.verify_payload):
                    raise RuntimeError("invalid VPD startup event")
                ready.add(event["rank"])
                self.worker_info.append(event)
            self.worker_info.sort(key=lambda event:event["rank"])
            self._pump_thread = threading.Thread(target=self._pump_events,name="vpd-events",daemon=True)
            self._pump_thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def _pump_events(self):
        """唯一读取 worker 事件的线程：验归属、释放阶段槽位、按请求投递。"""
        completion_stages={"vision_handoff":"vision","handoff":"prefill","result":"decode"}
        try:
            while not self._closed:
                try:
                    event = self._event(time.monotonic()+.1)
                except TimeoutError:
                    event = None
                with self._condition:
                    if self._closed:
                        return
                    for state in self._requests.values():
                        route = state.request.get("route")
                        required={name for name,rank in route.items() if rank is not None} if route else set()
                        if (route and not state.rejected and state.completed != required
                                and time.monotonic() >= state.deadline):
                            raise TimeoutError("active VPD request deadline exceeded")
                    if event is None:
                        continue
                    state = self._requests.get(event.get("request_id"))
                    if state is None:
                        raise RuntimeError("unknown VPD request event")
                    kind = event["type"]
                    stage = completion_stages.get(kind)
                    if kind in ("vision_ready","reject"):
                        stage = "vision"
                    elif kind == "token":
                        stage = "prefill" if event["index"] == 0 else "decode"
                    if stage is None or event.get("rank") != state.request["route"][stage]:
                        raise RuntimeError("worker event does not match request route")
                    if kind in completion_stages:
                        self._router.release_stage(state.request["request_id"],stage)
                        state.completed.add(stage)
                    elif kind == "reject":
                        state.rejected = True
                    state.inbox.put(event)
                    self._condition.notify_all()
        except BaseException as error:
            self.close(error=RuntimeError(f"VPD runtime stopped: {error}"))

    def _wait_route(self,state):
        request = state.request
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError(str(self._stop_error or "VPD runtime is closed"))
                remaining = state.deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("VPD route queue deadline exceeded")
                if self._pending[0] == request["request_id"]:
                    route = self._router.acquire(request["request_id"],bool(request["image_count"]))
                    if route is not None:
                        request["route"] = route
                        self._pending.popleft()
                        state.queue_wait_ms = (time.perf_counter()-request["started_s"])*1000
                        self._condition.notify_all()
                        return route
                self._condition.wait(timeout=min(.1,remaining))

    def _request_event(self,state,deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("VPD request deadline exceeded")
        try:
            event = state.inbox.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("VPD request deadline exceeded") from None
        if isinstance(event,BaseException):
            raise event
        return event

    def prepare(self,messages,new_tokens=8):
        return prepare_request(self.spec,messages,new_tokens)

    def run(self,messages,new_tokens=8,*,on_token=None):
        return self.run_prepared(self.prepare(messages,new_tokens),on_token=on_token)

    def run_prepared(self,prepared,*,on_token=None):
        normalized,images,new_tokens = prepared.messages,prepared.images,prepared.new_tokens
        digests = [hashlib.sha256(image).hexdigest() for image in images]
        # 只在登记/路由/释放时持锁。模型执行与用户回调不持有调度锁。
        with self._condition:
            if self._closed or len(self.processes)!=self.spec.worker_count:
                raise RuntimeError("VPD runtime is not active")
            if len(self._requests) >= self.max_inflight:
                raise CapacityError("VPD admission capacity exhausted")
            self._sequence += 1
            started = time.perf_counter()
            request = {"request_id":uuid.uuid4().hex,"sequence":self._sequence,
                       "started_s":started,
                       "new_tokens":new_tokens,"image_count":len(images),"image_digests":digests,
                       "context_digest":hashlib.sha256(json.dumps(digests).encode()).hexdigest()}
            if not images:
                # 只保护共享 processor 的短 CPU 调用，不锁住后续模型执行。
                inputs = prepare_inputs(self.spec,self.processor,normalized,[])
                request["tokens"] = inputs["input_ids"][0].tolist()
                self.spec.validate(request["tokens"],new_tokens)
            state = _RequestState(request,time.monotonic()+self.request_timeout_s)
            self._requests[request["request_id"]] = state
            self._pending.append(request["request_id"])
        finished = False
        rejected = False
        try:
            route = self._wait_route(state)
            feature_digest = None
            if images:
                # 先只唤醒 V，拿到处理器展开图像占位符后的完整 token 序列。
                self.controls[route["vision"]].put({**request,"messages":normalized,"images":images})
                event = self._request_event(state,state.deadline)
                if event["type"] == "reject":
                    rejected = True
                    raise ValueError(event["error"])
                if event["type"] != "vision_ready":
                    raise RuntimeError("unexpected vision preparation event")
                request["tokens"] = event["tokens"]
                feature_digest = event["feature_digest"]
            # P/D 现在都知道同一份 tokens 和图片上下文，才能核对交接元数据。
            for rank in (route["prefill"],route["decode"]):
                self.controls[rank].put(request)
            expected = ("vision_handoff","handoff","result") if images else ("handoff","result")
            # 主进程通过事件队列收 Token 和阶段完成消息；不亲自搬运特征/KV。
            received = self.collect(request,state.deadline,on_token,expected,
                event_source=lambda deadline:self._request_event(state,deadline))
            result = {**received["result"],**received["handoff"],**received.get("vision_handoff",{})}
            result.pop("type")
            result.pop("rank")  # 多阶段结果没有单一来源 rank；实际归属统一看 route。
            if result["prefill_feature_digest"] != feature_digest:
                raise RuntimeError("vision feature identity changed before prefill")
            times = result["token_ready_times_s"]
            result.update(mode="idefics3_three_stage",device=self.spec.devices[0],transport=self.spec.transport,
                stage_devices=list(self.spec.devices),control_transport="gloo",
                stage_pools=[list(pool) for pool in self.spec.pools],route=route,
                verify_payload=self.spec.verify_payload,worker_info=self.worker_info,
                pipeline_concurrent=True,queue_wait_ms=state.queue_wait_ms,max_inflight=self.max_inflight,
                worker_pids=[p.pid for p in self.processes],image_count=len(images),
                image_digests=digests,vision_bypassed=not images,
                vision_ms=result.get("vision_ms",0),vision_transfer_ms=result.get("vision_transfer_ms",0),
                kv_payload_bytes=self.spec.validate(request["tokens"],new_tokens),
                e2e_ms=(time.perf_counter()-started)*1000,backend_ttft_ms=(times[0]-started)*1000,
                backend_tbt_ms=[(b-a)*1000 for a,b in zip(times,times[1:])],
                scope=self.scope)
            finished = True
            return result
        except BaseException:
            # 只有明确的 V 预处理拒绝或尚未获配路由的等待失败可以局部回收。
            if "route" in request and not rejected:
                self.close(error=RuntimeError("an active VPD request failed; restart runtime"))
            raise
        finally:
            with self._condition:
                if not self._closed:
                    if finished:
                        self._router.finish(request["request_id"])
                    elif rejected:
                        self._router.abandon(request["request_id"])
                if request["request_id"] in self._pending:
                    self._pending.remove(request["request_id"])
                self._requests.pop(request["request_id"],None)
                self._condition.notify_all()

    def close(self,*,error=None):
        """选出一个关闭者，先唤醒所有请求，再无锁等待泵和进程退出。"""
        with self._condition:
            owner = not self._closed
            if owner:
                self._closed = True
                self._stop_error = error or RuntimeError("VPD runtime closed")
                for state in self._requests.values():
                    state.inbox.put(RuntimeError(str(self._stop_error)))
                self._condition.notify_all()
        if not owner:
            if threading.current_thread() is not self._pump_thread:
                self._stopped.wait()
            return
        try:
            if self._pump_thread is not None and threading.current_thread() is not self._pump_thread:
                self._pump_thread.join()
            self._cleanup_processes()
        finally:
            self._stopped.set()
