"""Dependency-light routing shared by production and the learning runtime."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable

Node = str


class Scheduler(ABC):

    def __init__(self, nodes: Iterable[Node]) -> None:
        self._nodes = list(nodes)
        self._active = len(self._nodes)
        self._lock = threading.Lock()

    def set_active(self, k: int) -> None:
        with self._lock:
            self._active = max(1, min(int(k), len(self._nodes)))

    @abstractmethod
    def acquire(self, load: int = 0) -> Node:
        pass

    @abstractmethod
    def release(self, node: Node, load: int = 0) -> None:
        pass


class RoundRobinScheduler(Scheduler):

    def __init__(self, nodes: Iterable[Node]) -> None:
        super().__init__(nodes)
        self._i = 0

    def acquire(self, load: int = 0) -> Node:
        with self._lock:
            node = self._nodes[self._i % self._active]
            self._i += 1
        return node

    def release(self, node: Node, load: int = 0) -> None:
        pass


class LeastLoadedScheduler(Scheduler):

    def __init__(self, nodes: Iterable[Node]) -> None:
        super().__init__(nodes)
        self._load: dict[Node, int] = {n: 0 for n in self._nodes}
        self._rr = 0

    def acquire(self, load: int = 0) -> Node:
        with self._lock:
            pool = self._nodes[: self._active]
            low = min(self._load[n] for n in pool)
            tied = [n for n in pool if self._load[n] == low]
            node = tied[self._rr % len(tied)]  # rotate among equally-loaded nodes
            self._rr += 1
            self._load[node] += load
        return node

    def release(self, node: Node, load: int = 0) -> None:
        with self._lock:
            if node in self._load:
                self._load[node] = max(0, self._load[node] - load)


def make_scheduler(name: str | None, nodes: Iterable[Node]) -> Scheduler:
    name = (name or "round_robin").lower()
    if name == "least_loaded":
        return LeastLoadedScheduler(nodes)
    if name == "round_robin":
        return RoundRobinScheduler(nodes)
    raise ValueError(
        f"Unknown scheduler={name!r} (expected round_robin or least_loaded)"
    )
