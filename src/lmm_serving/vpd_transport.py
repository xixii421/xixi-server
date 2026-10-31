"""单机三阶段通信：Gloo 传控制消息，Gloo 或 NCCL 传大张量。

阅读顺序：TensorChannel -> send_kv -> receive_kv；视觉特征的交接在 vpd_local.py。
rank 是进程在通信组中的编号，按 V 池、P 池、D 池展开，不能直接当作阶段编号。

一次交接的顺序是：发送元数据 -> 发送张量 -> 接收端验证/就绪 -> 返回 ACK。
ACK 是应用层的接收确认；本地发送完成并不代表对端已经验证并接管了数据。
这里使用异步通信 API 后立即等待和同步，尚未实现计算与通信重叠。
NCCL 的实际硬件路径由库和拓扑决定，使用 NCCL 不等于使用 NVLink 或 RDMA。
"""
from datetime import timedelta
import math

import torch
import torch.distributed as dist

from .pd_local import (receive_header, send_header, tensor_digest, validate_header)


class TensorChannel:
    """只封装张量通道；JSON 头和 ACK 始终由默认 Gloo 组处理。"""

    def __init__(self, backend, device, *, group=None, verify_payload=True):
        if backend not in ("gloo", "nccl"):
            raise ValueError("payload backend must be gloo or nccl")
        if backend == "nccl" and torch.device(device).type != "cuda":
            raise ValueError("NCCL payloads require a CUDA device")
        self.backend = backend
        # stage_device：该阶段做模型计算的设备。
        # payload_device：通信缓冲区所在设备。Gloo 使用 CPU；NCCL 使用本阶段 GPU。
        # 因此 Gloo 模式可能需要 GPU -> CPU -> 对端 CPU -> 对端 GPU 的搬运。
        self.stage_device = torch.device(device)
        self.payload_device = self.stage_device if backend == "nccl" else torch.device("cpu")
        # group=None 表示使用默认 Gloo 组；NCCL 模式传入单独创建的数据组。
        self.group = group
        self.verify_payload = verify_payload

    @classmethod
    def start(cls, spec, timeout):
        """在默认 Gloo 组初始化后调用，建立并预热可选的 NCCL 数据组。"""
        group = None
        if spec.transport == "nccl":
            # 所有进程必须以相同顺序创建通信组，并提前设置各自的当前 CUDA 设备。
            # 先做一次全员 collective，建立组通信，再让部分 rank 进行点对点传输。
            group = dist.new_group(ranks=list(range(spec.worker_count)), backend="nccl",
                                   timeout=timedelta(seconds=timeout))
            # 每个 rank 提供 1，SUM 后每张卡都应得到 worker_count。
            probe = torch.ones(1, device=spec.device)
            dist.all_reduce(probe, group=group)
            torch.cuda.synchronize(spec.device)
            if probe.item() != spec.worker_count:
                raise RuntimeError("NCCL worker-group bootstrap failed")
        return cls(spec.transport, spec.device, group=group,
                   verify_payload=spec.verify_payload)

    def prepare(self, tensor):
        """准备连续的通信缓冲区；不保证复制，调用方仍须管理其生命周期。"""
        # detach 只断开梯度关系，不会自动复制数据。
        # .to / .contiguous 在设备和布局已满足条件时也可能复用原来的存储。
        # 因而发送过程中不能修改源数据，不能把 prepare 当作不可变快照。
        return tensor.detach().to(self.payload_device).contiguous()

    def synchronize(self):
        if self.payload_device.type == "cuda":
            torch.cuda.synchronize(self.payload_device)

    def send(self, tensor, destination):
        if tensor.device != self.payload_device or not tensor.is_contiguous():
            raise ValueError("payload must be contiguous and on its channel device")
        dist.isend(tensor, dst=destination, group=self.group).wait()
        # CUDA 下 Work.wait 主要建立当前 stream 对通信的等待依赖，不能简单
        # 理解为 CPU 返回时 GPU 工作已经全部完成。这里进一步同步本设备，
        # 用保守方式保证实际完成；这会牺牲重叠机会，但让缓冲区管理更直接。
        # 此处完成的是本地发送，应用层接管还要等 send_kv / 视觉交接里的 ACK。
        self.synchronize()

    def receive(self, shape, dtype, source):
        # 接收端主动分配自己的存储。shape/dtype 应由已验证的本地协议决定，
        # 不能直接信任对端任意给出的尺寸，否则可能被诱导分配过大的显存。
        tensor = torch.empty(shape, dtype=dtype, device=self.payload_device)
        dist.irecv(tensor, src=source, group=self.group).wait()
        self.synchronize()
        return tensor

    def digest(self, tensor):
        # tensor_digest 会把 CUDA 张量复制到 CPU 做 SHA256，完整校验有额外成本。
        # 关闭时返回 None，只跳过载荷哈希；身份、形状和 ACK 检查仍在上层执行。
        return tensor_digest(tensor) if self.verify_payload else None

    def validate_digest(self, tensor, digest):
        # 双方校验模式必须一致。关闭校验时仍不接受对端悄悄带入摘要。
        if self.verify_payload:
            if not isinstance(digest, str) or len(digest) != 64 or self.digest(tensor) != digest:
                raise ValueError("payload checksum mismatch")
        elif digest is not None:
            raise ValueError("unexpected checksum while verification is disabled")

    def stage(self, tensor):
        """把已收到的张量放到模型设备，并保证后续计算可安全读取。"""
        # NCCL 下通常已在本阶段 GPU，.to 可以直接复用；Gloo 下需从 CPU 上传。
        if tensor.device != self.stage_device:
            tensor = tensor.to(self.stage_device)
            if self.stage_device.type == "cuda":
                torch.cuda.synchronize(self.stage_device)
        return tensor

    def close(self):
        # 这里只销毁单独的 NCCL 数据组；外层 worker 随后销毁默认 Gloo 组。
        if self.group is not None and dist.is_initialized():
            dist.destroy_process_group(self.group)
            self.group = None


def cache_shape(spec, request):
    """单层 K 或 V 的形状：[batch=1, KV 头数, 已缓存序列长度, 每头维度]。"""
    cfg = spec.config()
    # 调用方已检查请求/预算；此处只负责计算布局，不重复完整验证。
    # GQA 的 KV 头数可能少于 Query 头数，这里必须取 num_key_value_heads。
    return (1, cfg["num_key_value_heads"], len(request["tokens"]), cfg["head_dim"])


def send_kv(channel, spec, request, result, first_token, ready_at, destination=2):
    """P -> D：发送 prompt 的 KV 与首 Token，收到接管确认后才能释放源数据。"""
    import hashlib
    import json
    # legacy cache 按层排列，每层是一对 (K, V)。展平后顺序固定为
    # [K0, V0, K1, V1, ...]，接收端必须按相同规则恢复。
    tensors = [channel.prepare(t) for pair in result.past_key_values.to_legacy_cache() for t in pair]
    shape = cache_shape(spec, request)
    if (len(tensors) != 2*spec.config()["num_hidden_layers"] or
            any(tuple(t.shape) != shape or t.dtype != getattr(torch, spec.dtype) for t in tensors)):
        raise ValueError("model produced unexpected KV shape or dtype")
    # 校验所有层后，拼成一个一维连续缓冲区，减少逐层 send 的调用次数。
    # cat 会分配新的存储，所以交接期间源 KV 和 packed 会同时占用内存/显存。
    packed = torch.cat([t.reshape(-1) for t in tensors])
    # 头部走 Gloo，描述“这是谁的哪一次交接”，而不把大张量塞进 JSON。
    # prompt_digest 对齐文本 token；context_digest 对齐有序图片上下文；
    # digests 按 K/V 顺序验证实际数据。signature 不是权重文件的完整哈希。
    header = {"version": 1, "request_id": request["request_id"],
              "sequence": request["sequence"], "signature": spec.signature(),
              "prompt_length": len(request["tokens"]), "new_tokens": request["new_tokens"],
              "dtype": spec.dtype, "first_token": first_token,
              "prompt_digest": hashlib.sha256(json.dumps(request["tokens"]).encode()).hexdigest(),
              "context_digest": request.get("context_digest", ""),
              "route": request.get("route"),
              "first_token_ready_s": ready_at, "digests": [channel.digest(t) for t in tensors],
              "payload_format": "packed_kv_v1", "verify_payload": channel.verify_payload}
    # 接收端先拿到并验证头部，再按双方约定的尺寸接收 packed。
    send_header(header, destination)
    channel.send(packed, destination)
    # ACK=[通道序号, prompt 长度]，用于防止把迟到/错误的确认当成本次成功。
    # 等待期间 result、tensors 和 packed 都还持有存储，不能提前释放或改写。
    ack = torch.zeros(2, dtype=torch.int64)
    dist.recv(ack, src=destination)  # CPU 张量 + 默认 Gloo 组，不进入 NCCL 数据组。
    if ack.tolist() != [request["sequence"], len(request["tokens"])]:
        raise ValueError("invalid handoff acknowledgement")


def receive_kv(channel, spec, request, source=1):
    """D 接收 KV：先验身份和预算，再收数据，恢复缓存，最后发送 ACK。"""
    from transformers import DynamicCache
    header = receive_header(source)
    # 请求 UUID、序号、模型、prompt、图片上下文、dtype 等先匹配。
    # 若这一步失败，不分配接收缓冲区，也不发送成功 ACK；外层会关闭整组。
    validate_header(header, spec, request, verify_payload=channel.verify_payload)
    if (header.get("payload_format") != "packed_kv_v1" or
            header.get("verify_payload") is not channel.verify_payload):
        raise ValueError("KV payload protocol mismatch")
    shape = cache_shape(spec, request)
    count = 2*spec.config()["num_hidden_layers"]  # 每层两个张量：一个 K、一个 V。
    elements = math.prod(shape)
    # 总元素数由本地模型配置与已验证请求推导，不能采用对端任意指定的分配量。
    packed = channel.receive((count*elements,), getattr(torch, spec.dtype), source)
    # view/unbind 按原顺序切成多个视图；此时没有给每层复制一份完整数据。
    tensors = list(packed.view(count, *shape).unbind(0))
    for tensor, digest in zip(tensors, header["digests"]):
        channel.validate_digest(tensor, digest)
    # Gloo 模式下要把整个 packed 上传一次，再重新创建位于 GPU 上的 K/V 视图。
    # 视图持有底层存储，因此函数返回后，cache 中的张量仍能保有这份数据。
    packed = channel.stage(packed)
    tensors = list(packed.view(count, *shape).unbind(0))
    # 偶数位置是 K，奇数位置是 V；zip 恢复 ((K0,V0),(K1,V1),...)。
    cache = DynamicCache.from_legacy_cache(tuple(zip(tensors[::2], tensors[1::2])))
    if cache.get_seq_length() != len(request["tokens"]):
        raise ValueError("received cache length mismatch")
    # ACK 表示接收、校验、设备搬运及缓存恢复已经完成，可以释放发送端副本。
    # 它不表示 Decode 已完成。首 Token 已由 P 算出，D 用它续算第二个 Token。
    dist.send(torch.tensor([request["sequence"], len(request["tokens"])], dtype=torch.int64), dst=source)
    return cache, header
