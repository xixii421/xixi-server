from __future__ import annotations

import hashlib
import struct
import threading
import time
from collections import OrderedDict

FEATURE_SHAPE = (4, 8)
FEATURE_BYTES = 4 * 8 * 4  # float32


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise RequestError("request deadline exceeded", 504)
    return value


def tensor_checksum(tensor):
    values = tensor.reshape(-1).tolist()
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


class FeatureStore:
    """Bounded storage with deadlines and cancellation tombstones for late writes."""

    def __init__(self, max_bytes, ttl_s):
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self._entries = {}
        self._cancelled = {}
        self._lock = threading.Lock()

    def _sweep(self):
        now = time.monotonic()
        self._entries = {k: v for k, v in self._entries.items() if v[2] > now}
        self._cancelled = {k: expiry for k, expiry in self._cancelled.items() if expiry > now}

    def put(self, transfer_id, request_key, tensor, deadline):
        with self._lock:
            self._sweep()
            remaining(deadline)
            if request_key in self._cancelled:
                raise RequestError("request was cancelled", 410)
            if transfer_id in self._entries:
                raise RequestError("duplicate transfer_id", 409)
            used = sum(v[1].numel() * v[1].element_size() for v in self._entries.values())
            if used + tensor.numel() * tensor.element_size() > self.max_bytes:
                raise RequestError("feature buffer capacity exceeded", 429)
            self._entries[transfer_id] = (
                request_key, tensor, min(deadline, time.monotonic() + self.ttl_s)
            )

    def pop(self, transfer_id, request_key):
        with self._lock:
            self._sweep()
            value = self._entries.get(transfer_id)
            if value is None or value[0] != request_key:
                raise RequestError("feature not found for this request", 404)
            del self._entries[transfer_id]
            return value[1]

    def release(self, request_key):
        with self._lock:
            self._sweep()
            self._entries = {k: v for k, v in self._entries.items() if v[0] != request_key}
            self._cancelled[request_key] = time.monotonic() + self.ttl_s

    def stats(self):
        with self._lock:
            self._sweep()
            return {
                "pending_features": len(self._entries),
                "pending_bytes": sum(v[1].numel() * v[1].element_size() for v in self._entries.values()),
            }


class RequestRegistry:
    """Reject active/recent duplicate public IDs; retain at most 1024 completed IDs."""

    def __init__(self):
        self._active = set()
        self._completed = OrderedDict()
        self._lock = threading.Lock()

    def begin(self, req_id):
        with self._lock:
            if req_id in self._active or req_id in self._completed:
                raise RequestError("duplicate req_id (active or recently completed)", 409)
            self._active.add(req_id)

    def finish(self, req_id):
        with self._lock:
            self._active.discard(req_id)
            self._completed[req_id] = None
            while len(self._completed) > 1024:
                self._completed.popitem(last=False)

    def active_count(self):
        with self._lock:
            return len(self._active)
