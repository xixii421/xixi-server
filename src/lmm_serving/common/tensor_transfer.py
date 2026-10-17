"""Production transfer protocol shared by NCCL services and CPU/Gloo tests.

The receiver owns each pair's FIFO, epoch and sequence number. API frontends
submit ONE receive request; they never independently trigger the two endpoints.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import requests

from .features import FeatureRegistry, feature_bytes, transfer_key
from .resources import ProtocolError, positive_timeout, time_left


def parse_spec(data):
    if not isinstance(data, dict):
        raise ProtocolError("transfer body must be an object")
    if data.get("protocol_version") != 2:
        raise ProtocolError("transfer protocol v2 required; rebuild patched vLLM", 409)
    tid = transfer_key(data.get("transfer_id"))
    owner = data.get("request_id")
    if not isinstance(owner, str) or not owner:
        raise ProtocolError("request_id is required")
    shape, dtype = data.get("tensor_shape"), data.get("dtype", "bfloat16")
    feature_bytes(shape, dtype)
    src, dst = data.get("src_rank"), data.get("dest_rank")
    if type(src) is not int or type(dst) is not int or min(src, dst) < 0 or src == dst:
        raise ProtocolError("invalid transfer ranks")
    return {"protocol_version": 2, "transfer_id": tid, "request_id": owner,
            "tensor_shape": list(shape), "dtype": dtype, "src_rank": src, "dest_rank": dst}


def identity(spec, epoch, sequence):
    return hashlib.sha256(json.dumps(
        [spec, epoch, sequence], sort_keys=True, separators=(",", ":")
    ).encode()).digest()


class TorchTransport:
    def __init__(self, device, groups):
        self.device, self.groups = device, groups
        self._buffers = {}
        self._buffer_lock = threading.Lock()

    def _sync(self):
        import torch
        if str(self.device).startswith("cuda"):
            torch.cuda.current_stream(self.device).synchronize()

    def send(self, spec, tensor, header):
        import torch
        import torch.distributed as dist
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        peer = spec["dest_rank"]
        metadata = torch.tensor(list(header), dtype=torch.uint8, device=self.device)
        with self._buffer_lock:
            self._buffers[peer] = (metadata, tensor)
        dist.send(metadata, dst=peer, group=self.groups[peer])
        dist.send(tensor, dst=peer, group=self.groups[peer])
        self._sync()  # Keep BOTH buffers alive until the actual device work ends.
        with self._buffer_lock:
            self._buffers.pop(peer, None)

    def recv(self, spec, header):
        import torch
        import torch.distributed as dist
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        peer = spec["src_rank"]
        metadata = torch.empty(32, dtype=torch.uint8, device=self.device)
        tensor = torch.empty(spec["tensor_shape"], dtype=getattr(torch, spec["dtype"]), device=self.device)
        with self._buffer_lock:
            self._buffers[peer] = (metadata, tensor)
        dist.recv(metadata, src=peer, group=self.groups[peer])
        dist.recv(tensor, src=peer, group=self.groups[peer])
        self._sync()
        if bytes(metadata.tolist()) != header:
            raise ProtocolError("wire transfer identity mismatch", 503)
        with self._buffer_lock:
            self._buffers.pop(peer, None)
        return tensor


class NodeHealth:
    def __init__(self):
        self.failed = threading.Event()
        self.reason = ""

    def check(self):
        if self.failed.is_set():
            raise ProtocolError("transfer node quarantined; restart all NCCL ranks: " + self.reason, 503)

    def fail(self, error):
        self.reason = str(error)
        self.failed.set()


class DeviceGate:
    """At most one communicator uses a device in this transfer process.

    Prepared permits may expire. Claimed permits never expire automatically:
    their device operation must complete or the process must be restarted.
    """
    def __init__(self):
        self._cv = threading.Condition()
        self._owner = self._token = None
        self._deadline = 0
        self._claimed = False

    def prepare(self, owner, deadline):
        with self._cv:
            while True:
                if self._owner and not self._claimed and self._deadline <= time.monotonic():
                    self._owner = None
                if self._owner == owner:
                    return self._token
                if self._owner is None:
                    self._owner, self._token = owner, uuid.uuid4().hex
                    self._deadline, self._claimed = deadline, False
                    return self._token
                self._cv.wait(min(time_left(deadline), 0.1))

    def claim(self, owner, token):
        with self._cv:
            if self._owner != owner or self._token != token or (
                not self._claimed and self._deadline <= time.monotonic()
            ):
                raise ProtocolError("expired or stale device permit", 409)
            self._claimed = True

    def release(self, owner, token, *, prepared_only=False):
        with self._cv:
            if self._owner == owner and self._token == token and not (prepared_only and self._claimed):
                self._owner = self._token = None
                self._claimed = False
                self._cv.notify_all()


@dataclass
class Job:
    spec: dict
    deadline: float
    epoch: str = ""
    sequence: int = -1
    source_url: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)
    error: Exception | None = None
    started: bool = False
    cancelled: bool = False
    finished_at: float = 0
    wire_token: str = ""


class TransferService:
    """One service per process; bounded jobs, bounded bytes and per-peer workers."""

    def __init__(self, role, rank, peers, store, transport, *, timeout_s=60,
                 queue_size=64, source_urls=None, make_handle=None, history_s=660, history_limit=65536):
        self.role, self.rank, self.peers = role, rank, set(peers)
        self.store: FeatureRegistry = store
        self.transport = transport
        self.timeout_s = positive_timeout(timeout_s)
        self.history_s = history_s
        self.history_limit = history_limit
        self.source_urls = source_urls or {}
        self.make_handle = make_handle
        self.health = NodeHealth()
        self.device_gate = DeviceGate()
        self.epoch = uuid.uuid4().hex
        self._channels = {}
        self._jobs = {}
        self._completed = deque()
        self._lock = threading.RLock()
        self._queues = {p: queue.Queue(maxsize=queue_size) for p in peers}
        self._stop = threading.Event()
        self._threads = []
        for peer in peers:
            thread = threading.Thread(target=self._worker, args=(peer,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _prune(self):
        now = time.monotonic()
        while self._completed and self._completed[0][0] + self.history_s <= now:
            _, tid = self._completed.popleft()
            self._jobs.pop(tid, None)

    def submit(self, data):
        spec = parse_spec(data)
        timeout_s = min(positive_timeout(data.get("timeout_s", self.timeout_s)), self.timeout_s)
        peer = spec["dest_rank"] if self.role == "sender" else spec["src_rank"]
        own = spec["src_rank"] if self.role == "sender" else spec["dest_rank"]
        if own != self.rank or peer not in self.peers:
            raise ProtocolError("rank does not belong to this service")
        tid = spec["transfer_id"]
        with self._lock:
            self.health.check()
            self._prune()
            existing = self._jobs.get(tid)
            if existing:
                if existing.spec != spec:
                    raise ProtocolError("duplicate transfer_id with changed metadata", 409)
                if self.role == "sender" and (existing.epoch != data.get("channel_epoch")
                                              or existing.sequence != data.get("sequence")):
                    raise ProtocolError("duplicate transfer with changed sequence", 409)
                if self.role == "receiver" and existing.source_url != data.get("source_url"):
                    raise ProtocolError("duplicate transfer with changed source", 409)
                return existing
            if len(self._jobs) >= self.history_limit or self._queues[peer].full():
                raise ProtocolError("transfer queue/history capacity exhausted", 429)
            job = Job(spec, time.monotonic() + timeout_s)
            if self.role == "sender":
                epoch, seq = data.get("channel_epoch"), data.get("sequence")
                if not isinstance(epoch, str) or len(epoch) != 32 or type(seq) is not int:
                    raise ProtocolError("invalid channel epoch/sequence")
                old_epoch, expected = self._channels.get(peer, (epoch, 0))
                if epoch != old_epoch or seq != expected:
                    raise ProtocolError("stale channel epoch or out-of-order sequence", 409)
                job.epoch, job.sequence = epoch, seq
                self.device_gate.claim(tid, data.get("wire_token"))
                job.wire_token = data.get("wire_token")
                self._channels[peer] = (epoch, expected + 1)
            else:
                source = data.get("source_url")
                if source != self.source_urls.get(peer):
                    raise ProtocolError("source_url does not match configured rank")
                job.source_url = source
                self.store.reserve(tid, spec["request_id"],
                                   feature_bytes(spec["tensor_shape"], spec["dtype"]), timeout_s)
            self._jobs[tid] = job
            self._queues[peer].put_nowait(job)
            return job

    def wait(self, job):
        if not job.done.wait(max(0, job.deadline - time.monotonic())):
            # Atomic with the worker's started flag: a queued cancellation must
            # never race into a new wire operation after returning HTTP 504.
            with self._lock:
                job.cancelled = True
                self.store.cancel(job.spec["transfer_id"], job.spec["request_id"])
                if job.started or self.role == "sender":
                    self.health.fail("HTTP deadline expired during a scheduled transfer")
            raise ProtocolError("transfer deadline exceeded", 504)
        if job.error:
            raise job.error
        self.health.check()
        if self.role == "receiver":
            return self.store.lease(job.spec["transfer_id"], self.make_handle)
        return job.result

    def _source_call(self, job, endpoint, **extra):
        payload = {**job.spec, "channel_epoch": job.epoch, "sequence": job.sequence,
                   "timeout_s": time_left(job.deadline), **extra}
        try:
            response = requests.post(
                f"http://{job.source_url}{endpoint}",
                json=payload, timeout=time_left(job.deadline),
            )
            if response.status_code != 200:
                raise ProtocolError(f"source rejected transfer: {response.status_code} {response.text[:200]}", 503)
            return response.json()
        except requests.RequestException as error:
            raise ProtocolError(f"source transfer control failed: {error}", 503) from error

    def _worker(self, peer):
        sequence = 0
        while not self._stop.is_set():
            try:
                job = self._queues[peer].get(timeout=0.2)
            except queue.Empty:
                self.store.snapshot()
                continue
            entered_device = False
            local_token = source_token = None
            tid = job.spec["transfer_id"]
            try:
                with self._lock:
                    self.health.check()
                    if job.cancelled:
                        raise ProtocolError("queued transfer cancelled", 410)
                    time_left(job.deadline)
                    job.started = True
                if self.role == "sender":
                    tensor = self.store.begin(tid, job.spec["request_id"], "sending")
                    if (list(tensor.shape) != job.spec["tensor_shape"]
                            or str(tensor.dtype).split(".")[-1] != job.spec["dtype"]):
                        raise ProtocolError("encoded tensor metadata mismatch", 409)
                    entered_device = True
                    self.transport.send(job.spec, tensor, identity(job.spec, job.epoch, job.sequence))
                    self.store.finish_device_use(tid)
                    self.device_gate.release(tid, job.wire_token)
                    job.result = {"transfer_id": tid, "sequence": job.sequence}
                else:
                    # Image ranks are lower than all Text ranks. Acquire the
                    # source device BEFORE the destination device everywhere;
                    # this prevents cycles across multiple pair communicators.
                    source_token = self._source_call(job, "/prepare_transfer")["wire_token"]
                    local_token = self.device_gate.prepare(tid, job.deadline)
                    self.device_gate.claim(tid, local_token)
                    self.store.begin(tid, job.spec["request_id"], "receiving")
                    job.epoch, job.sequence = self.epoch, sequence
                    sequence += 1
                    # Acknowledge enqueue before entering recv. The sender's
                    # HTTP handler returns immediately while its worker sends.
                    entered_device = True
                    self._source_call(job, "/initiate_tensor_transfer_image_node",
                                      wire_token=source_token, enqueue_only=True)
                    tensor = self.transport.recv(job.spec, identity(job.spec, job.epoch, job.sequence))
                    self._source_call(job, "/initiate_tensor_transfer_image_node", wire_token=source_token)
                    entered_device = False  # Both ranks confirmed device completion.
                    self.device_gate.release(tid, local_token)
                    self.store.publish(tid, tensor)
            except Exception as error:
                job.error = error if isinstance(error, ProtocolError) else ProtocolError(str(error), 503)
                job.error.__traceback__ = None
                if entered_device or self.role == "sender":
                    # Keep any uncertain device/IPC buffers pinned until restart.
                    self.health.fail(error)
                else:
                    if local_token:
                        self.device_gate.release(tid, local_token)
                    if source_token:
                        try:
                            self._source_call(job, "/cancel_prepared_transfer", wire_token=source_token)
                        except Exception:
                            pass  # Unclaimed source permit has a bounded lease.
                    self.store.finish_device_use(tid)
            finally:
                tensor = None  # Do not retain the last GPU allocation in a worker local.
                with self._lock:
                    job.finished_at = time.monotonic()
                    self._completed.append((job.finished_at, tid))
                job.done.set()
                self._queues[peer].task_done()

    def snapshot(self):
        with self._lock:
            self._prune()
            return {"protocol_version": 2, "ready": not self.health.failed.is_set(),
                    "error": self.health.reason, "rank": self.rank,
                    "queue_depth": {str(p): q.qsize() for p, q in self._queues.items()},
                    "features": self.store.snapshot()}

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)

    def install(self, app):
        from flask import jsonify, request

        app.config["MAX_CONTENT_LENGTH"] = 1048576

        @app.errorhandler(ProtocolError)
        def error_response(error):
            return jsonify(error=str(error), protocol_version=2), error.status

        @app.get("/transfer_status")
        def status():
            snapshot = self.snapshot()
            return jsonify(snapshot), 200 if snapshot["ready"] else 503

        @app.post("/check_transfer_ready")
        def check_ready():
            self.health.check()
            return jsonify(protocol_version=2, ready=True)

        @app.post("/cancel_feature")
        def cancel():
            data = request.get_json()
            self.store.cancel(transfer_key(data.get("transfer_id")), data.get("request_id"))
            return jsonify(cancelled=True)

        if self.role == "sender":
            @app.post("/reserve_feature")
            def reserve():
                self.health.check()
                data = request.get_json()
                self.store.reserve(transfer_key(data.get("transfer_id")), data.get("request_id"),
                                   data.get("feature_bytes"), data.get("timeout_s"))
                return jsonify(reserved=True, protocol_version=2)

            @app.post("/prepare_transfer")
            def prepare():
                self.health.check()
                data = request.get_json()
                spec = parse_spec(data)
                if spec["src_rank"] != self.rank or spec["dest_rank"] not in self.peers:
                    raise ProtocolError("invalid source/destination rank")
                self.store.check_ready(spec["transfer_id"], spec["request_id"])
                deadline = time.monotonic() + min(positive_timeout(data.get("timeout_s")), self.timeout_s)
                token = self.device_gate.prepare(spec["transfer_id"], deadline)
                return jsonify(wire_token=token)

            @app.post("/cancel_prepared_transfer")
            def cancel_prepared():
                data = request.get_json()
                self.device_gate.release(data.get("transfer_id"), data.get("wire_token"), prepared_only=True)
                return jsonify(cancelled=True)

        endpoint = ("/initiate_tensor_transfer_image_node" if self.role == "sender"
                    else "/initiate_tensor_transfer_text_node")

        @app.post(endpoint)
        def transfer():
            data = request.get_json()
            job = self.submit(data)
            if self.role == "sender" and data.get("enqueue_only") is True:
                return jsonify(accepted=True, transfer_id=job.spec["transfer_id"])
            return jsonify(self.wait(job))

        if self.role == "receiver":
            @app.post("/release_tensor")
            def release():
                data = request.get_json()
                self.store.acknowledge(transfer_key(data.get("transfer_id")), data.get("lease_token"))
                return jsonify(released=True)
