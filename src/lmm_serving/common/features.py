"""Production feature ownership. No CUDA allocation is freed by TTL mid-use."""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from .resources import Capacity, ProtocolError, positive_timeout

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def feature_bytes(shape, dtype="bfloat16"):
    if (not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 6
            or any(type(d) is not int or d <= 0 for d in shape)):
        raise ProtocolError("invalid tensor_shape")
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise ProtocolError("unsupported tensor dtype")
    return math.prod(shape) * DTYPE_BYTES[dtype]


def transfer_key(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 160:
        raise ProtocolError("transfer_id is required (protocol v2)")
    return value


@dataclass
class Feature:
    transfer_id: str
    request_id: str
    nbytes: int
    deadline: float
    state: str = "reserved"
    tensor: object = None
    cancelled: bool = False
    lease_token: str = ""
    handle: str = ""
    initial_bytes: int = 0


class FeatureRegistry:
    """reserved -> encoding/receiving -> ready -> sending/leased -> released.

    Cancellation of encoding/sending/receiving/leased only sets a flag. The
    worker must confirm device completion (or the IPC consumer acknowledge its
    completed copy) before credits and backing memory can be released.
    """

    BUSY = {"encoding", "sending", "receiving", "leased"}

    def __init__(self, max_bytes, max_features, history_s=660, history_limit=65536):
        self.capacity = Capacity(bytes=int(max_bytes), slots=int(max_features))
        self.history_s = float(history_s)
        self.history_limit = int(history_limit)
        self._items: dict[str, Feature] = {}
        self._finished = OrderedDict()
        self._lock = threading.RLock()

    def _finish(self, item):
        self._items.pop(item.transfer_id, None)
        self.capacity.release(item.transfer_id)
        self._finished[item.transfer_id] = (item.request_id, time.monotonic() + self.history_s)

    def _sweep(self):
        now = time.monotonic()
        while self._finished and next(iter(self._finished.values()))[1] <= now:
            self._finished.popitem(last=False)
        for item in list(self._items.values()):
            if item.deadline <= now:
                item.cancelled = True
                if item.state not in self.BUSY:
                    self._finish(item)

    def reserve(self, tid, request_id, nbytes, timeout_s):
        transfer_key(tid)
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolError("request_id is required")
        timeout_s = positive_timeout(timeout_s)
        with self._lock:
            self._sweep()
            if tid in self._finished:
                raise ProtocolError("transfer already finished or cancelled", 410)
            existing = self._items.get(tid)
            if existing:
                if existing.request_id != request_id or existing.initial_bytes != nbytes:
                    raise ProtocolError("transfer reservation changed", 409)
                if existing.cancelled:
                    raise ProtocolError("transfer cancelled", 410)
                return existing
            # Refuse admission instead of evicting live cancellation tombstones.
            if len(self._finished) + len(self._items) >= self.history_limit:
                raise ProtocolError("transfer history full; retry after expiry", 429)
            self.capacity.reserve(tid, bytes=nbytes, slots=1)
            item = Feature(tid, request_id, nbytes, time.monotonic() + timeout_s)
            item.initial_bytes = nbytes
            self._items[tid] = item
            return item

    def _get(self, tid, request_id=None):
        self._sweep()
        item = self._items.get(tid)
        if item is None:
            raise ProtocolError("transfer unavailable", 410 if tid in self._finished else 404)
        if request_id is not None and item.request_id != request_id:
            raise ProtocolError("transfer ownership mismatch", 409)
        return item

    def begin(self, tid, request_id, state):
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolError("request_id is required")
        if state not in {"encoding", "receiving", "sending"}:
            raise ProtocolError("invalid feature state")
        with self._lock:
            item = self._get(tid, request_id)
            expected = "ready" if state == "sending" else "reserved"
            if item.cancelled or item.state != expected:
                raise ProtocolError(f"cannot enter {state} from {item.state}", 409)
            item.state = state
            return item.tensor

    def check_ready(self, tid, request_id):
        with self._lock:
            item = self._get(tid, request_id)
            if item.cancelled or item.state != "ready":
                raise ProtocolError("feature is not ready", 409)

    def check_encoding(self, tid, request_id):
        with self._lock:
            item = self._get(tid, request_id)
            if item.cancelled or item.state != "encoding":
                raise ProtocolError("queued encoding cancelled or expired", 410)

    def publish(self, tid, tensor):
        with self._lock:
            item = self._get(tid)
            if item.state not in {"encoding", "receiving"}:
                raise ProtocolError("invalid publish state", 409)
            if item.cancelled or item.deadline <= time.monotonic():
                self._finish(item)
                raise ProtocolError("transfer cancelled or expired", 410)
            nbytes = tensor.numel() * tensor.element_size()
            try:
                self.capacity.resize(tid, bytes=nbytes, slots=1)
            except ProtocolError:
                self._finish(item)
                raise
            item.nbytes, item.tensor, item.state = nbytes, tensor, "ready"

    def finish_device_use(self, tid):
        """Only call after the device operation is known complete."""
        with self._lock:
            item = self._items.get(tid)
            if item:
                self._finish(item)

    def lease(self, tid, make_handle):
        with self._lock:
            item = self._get(tid)
            if item.cancelled:
                raise ProtocolError("transfer cancelled", 410)
            if item.state == "ready":
                item.handle = make_handle(item.tensor)
                item.lease_token = uuid.uuid4().hex
                item.state = "leased"
            elif item.state == "leased":
                # Reissuing the same handle allows two IPC readers to race:
                # reader A could acknowledge while reader B is still copying.
                raise ProtocolError("IPC lease already claimed by a consumer", 409)
            else:
                raise ProtocolError("feature not ready for IPC", 409)
            return {"transfer_id": tid, "ipc_handle": item.handle, "lease_token": item.lease_token}

    def acknowledge(self, tid, token):
        with self._lock:
            item = self._items.get(tid)
            if item is None:
                if tid in self._finished:
                    return
                raise ProtocolError("unknown transfer", 404)
            if item.state != "leased" or not token or token != item.lease_token:
                raise ProtocolError("IPC release requires matching lease_token", 409)
            self._finish(item)

    def cancel(self, tid, request_id):
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolError("request_id is required")
        with self._lock:
            self._sweep()
            if tid in self._finished:
                if self._finished[tid][0] != request_id:
                    raise ProtocolError("transfer ownership mismatch", 409)
                return
            if tid not in self._items:
                if not isinstance(request_id, str) or not request_id:
                    raise ProtocolError("request_id is required")
                if len(self._finished) + len(self._items) >= self.history_limit:
                    raise ProtocolError("transfer history full", 429)
                self._finished[tid] = (request_id, time.monotonic() + self.history_s)
                return
            item = self._get(tid, request_id)
            item.cancelled = True
            if item.state not in self.BUSY:
                self._finish(item)

    def snapshot(self):
        with self._lock:
            self._sweep()
            return {**self.capacity.snapshot(), "states": {
                state: sum(i.state == state for i in self._items.values())
                for state in ("reserved", "encoding", "ready", "sending", "receiving", "leased")
            }, "cancelled_in_use": sum(i.cancelled for i in self._items.values())}
