"""Bounded, idempotent resource reservations used by the production data path."""
from __future__ import annotations

import math
import threading
import time


class ProtocolError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def positive_timeout(value) -> float:
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ProtocolError("invalid timeout_s") from None
    if not math.isfinite(value) or not 0 < value <= 86400:
        raise ProtocolError("timeout_s must be in (0, 86400]")
    return value


def time_left(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise ProtocolError("request deadline exceeded", 504)
    return value


class Capacity:
    """Atomically reserve several dimensions; reject oversized work, never clamp.

    Resize never waits while holding credits, avoiding hold-and-wait deadlocks.
    Reservations are released by their OWNER, not by an unrelated HTTP timeout.
    """

    def __init__(self, **limits: int):
        if not limits or any(type(v) is not int or v <= 0 for v in limits.values()):
            raise ValueError("capacity limits must be positive integers")
        self.limits = limits
        self._items: dict[str, dict[str, int]] = {}
        self._cv = threading.Condition(threading.RLock())

    def _validate(self, amounts):
        if set(amounts) != set(self.limits):
            raise ProtocolError("resource dimensions do not match")
        if any(type(v) is not int or v < 0 for v in amounts.values()):
            raise ProtocolError("resource amounts must be nonnegative integers")
        if any(amounts[k] > limit for k, limit in self.limits.items()):
            raise ProtocolError("single request exceeds node capacity", 413)

    def reserve(self, key: str, *, wait_s: float = 0, **amounts):
        self._validate(amounts)
        deadline = time.monotonic() + wait_s
        with self._cv:
            if key in self._items:
                if self._items[key] != amounts:
                    raise ProtocolError("reservation payload changed", 409)
                return
            while True:
                used = self._used()
                if all(used[k] + amounts[k] <= self.limits[k] for k in amounts):
                    self._items[key] = amounts.copy()
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProtocolError("node capacity exhausted", 429)
                self._cv.wait(remaining)
                if key in self._items:
                    if self._items[key] != amounts:
                        raise ProtocolError("reservation payload changed", 409)
                    return

    def resize(self, key: str, **amounts):
        self._validate(amounts)
        with self._cv:
            old = self._items.get(key)
            if old is None:
                raise ProtocolError("reservation not found", 404)
            used = self._used()
            if any(used[k] - old[k] + amounts[k] > self.limits[k] for k in amounts):
                raise ProtocolError("actual features exceed available capacity", 429)
            self._items[key] = amounts.copy()
            self._cv.notify_all()

    def release(self, key: str):
        with self._cv:
            self._items.pop(key, None)
            self._cv.notify_all()

    def _used(self):
        return {k: sum(item[k] for item in self._items.values()) for k in self.limits}

    def snapshot(self):
        with self._cv:
            return {"limits": self.limits.copy(), "used": self._used(), "reservations": len(self._items)}
