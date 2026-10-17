"""Authoritative Text-node budgets, reserved BEFORE image encoding starts."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from .resources import Capacity, ProtocolError, positive_timeout


class TextAdmission:
    def __init__(self, tokens, feature_bytes, requests=128, history_s=660, history_limit=65536):
        self.capacity = Capacity(tokens=int(tokens), bytes=int(feature_bytes), slots=int(requests))
        self.history_s = history_s
        self.history_limit = history_limit
        self._records = {}
        self._finished = OrderedDict()
        self._lock = threading.RLock()
        self.failed = False

    def _sweep(self):
        now = time.monotonic()
        while self._finished and next(iter(self._finished.values())) <= now:
            self._finished.popitem(last=False)
        for key, record in list(self._records.items()):
            if record["state"] == "reserved" and record["deadline"] <= now:
                self._finish(key)

    def _finish(self, key):
        self._records.pop(key, None)
        self.capacity.release(key)
        self._finished.pop(key, None)
        self._finished[key] = time.monotonic() + self.history_s

    def reserve(self, key, tokens, payload_bytes, timeout_s):
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ProtocolError("invalid request_id")
        timeout_s = positive_timeout(timeout_s)
        if type(tokens) is not int or type(payload_bytes) is not int or min(tokens, payload_bytes) < 0:
            raise ProtocolError("image_tokens and feature_bytes must be nonnegative integers")
        with self._lock:
            self._sweep()
            if self.failed:
                raise ProtocolError("Text node quarantined; restart required", 503)
            if key in self._finished:
                raise ProtocolError("request already completed or cancelled", 410)
            amounts = dict(tokens=tokens, bytes=payload_bytes * 2, slots=1)
            if key in self._records:
                if self._records[key]["initial"] != amounts:
                    raise ProtocolError("request reservation changed", 409)
                return
            if len(self._finished) + len(self._records) >= self.history_limit:
                raise ProtocolError("admission history full", 429)
            # Two copies conservatively cover recv buffer + IPC clone overlap.
            self.capacity.reserve(key, **amounts)
            self._records[key] = {
                "state": "reserved", "initial": amounts, "current": amounts.copy(),
                "deadline": time.monotonic() + timeout_s,
            }

    def resize(self, key, tokens, payload_bytes):
        if type(tokens) is not int or type(payload_bytes) is not int or min(tokens, payload_bytes) < 0:
            raise ProtocolError("image_tokens and feature_bytes must be nonnegative integers")
        with self._lock:
            self._sweep()
            record = self._records.get(key)
            if not record or record["state"] != "reserved":
                raise ProtocolError("request cannot be resized", 409)
            self.capacity.resize(key, tokens=tokens, bytes=payload_bytes * 2, slots=1)
            record["current"] = dict(tokens=tokens, bytes=payload_bytes * 2, slots=1)

    def begin(self, key, tokens=None, payload_bytes=None):
        with self._lock:
            self._sweep()
            if self.failed:
                raise ProtocolError("Text node quarantined; restart required", 503)
            record = self._records.get(key)
            if not record or record["state"] != "reserved":
                raise ProtocolError("missing, expired or replayed request reservation", 409)
            if tokens is not None and record["current"] != dict(tokens=tokens, bytes=payload_bytes * 2, slots=1):
                raise ProtocolError("request features do not match reserved capacity", 409)
            record["state"] = "running"
            return record["deadline"]

    def complete(self, key, *, uncertain=False):
        with self._lock:
            if uncertain:
                # A timed-out HTTP call does not prove the engine stopped using
                # its embeddings/KV cache. Keep credits and stop new admission.
                self.failed = True
                if key in self._records:
                    self._records[key]["state"] = "quarantined"
            else:
                self._finish(key)

    def cancel(self, key):
        if not isinstance(key, str) or not key:
            raise ProtocolError("request_id is required")
        with self._lock:
            self._sweep()
            record = self._records.get(key)
            if record and record["state"] in {"running", "quarantined"}:
                return False  # Caller timeout cannot refund active engine work.
            if key not in self._finished and not record and len(self._finished) + len(self._records) >= self.history_limit:
                raise ProtocolError("admission history full", 429)
            self._finish(key)
            return True

    def snapshot(self):
        with self._lock:
            self._sweep()
            return {**self.capacity.snapshot(), "ready": not self.failed,
                    "states": {state: sum(r["state"] == state for r in self._records.values())
                               for state in ("reserved", "running", "quarantined")}}

    def install(self, app):
        from flask import jsonify, request

        def body():
            data = request.get_json()
            if not isinstance(data, dict):
                raise ProtocolError("admission body must be an object")
            return data

        @app.errorhandler(ProtocolError)
        def error_response(error):
            return jsonify(error=str(error)), error.status

        @app.get("/admission_status", endpoint="admission_status")
        def status():
            data = self.snapshot()
            return jsonify(data), 200 if data["ready"] else 503

        @app.post("/reserve_request", endpoint="admission_reserve")
        def reserve():
            data = body()
            self.reserve(data.get("request_id"), data.get("image_tokens"), data.get("feature_bytes"), data.get("timeout_s"))
            return jsonify(self.snapshot())

        @app.post("/resize_request", endpoint="admission_resize")
        def resize():
            data = body()
            self.resize(data.get("request_id"), data.get("image_tokens"), data.get("feature_bytes"))
            return jsonify(self.snapshot())

        @app.post("/release_request", endpoint="admission_release")
        def release():
            released = self.cancel(body().get("request_id"))
            return jsonify(released=released), 200 if released else 409
