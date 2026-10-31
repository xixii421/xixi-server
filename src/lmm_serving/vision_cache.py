"""Byte-bounded CPU feature cache owned by the single vision worker.

Stored values and returned values are copies. A request cannot mutate another
request's cached feature, and a tensor view cannot retain an unaccounted batch.
The budget counts resident feature payload only, separately from active handoff.
"""
from collections import OrderedDict


class VisionFeatureCache:
    def __init__(self, capacity_bytes):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("cache capacity must be a nonnegative integer")
        self.capacity = capacity_bytes
        self.items = OrderedDict()
        self.used_bytes = 0
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        value = self.items.pop(key, None)
        if value is None:
            self.misses += 1
            return None
        self.items[key] = value
        self.hits += 1
        return value.clone()

    def put(self, key, value):
        if value.device.type != "cpu":
            raise ValueError("vision cache stores CPU tensors only")
        size = value.numel()*value.element_size()
        if not size or size > self.capacity:
            return False
        old = self.items.pop(key, None)
        if old is not None:
            self.used_bytes -= old.numel()*old.element_size()
        while self.used_bytes+size > self.capacity:
            _, victim = self.items.popitem(last=False)
            self.used_bytes -= victim.numel()*victim.element_size()
            self.evictions += 1
        self.items[key] = value.detach().contiguous().clone()
        self.used_bytes += size
        return True

    def snapshot(self):
        return {"capacity_bytes": self.capacity, "resident_bytes": self.used_bytes,
                "entries": len(self.items), "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}
