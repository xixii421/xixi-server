from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/single_gpu_mock/local.json"


@dataclass(frozen=True)
class LearningConfig:
    host: str = "127.0.0.1"
    port: int = 7070
    image_port: int = 7081
    text_port: int = 7082
    gloo_port: int = 29510
    image_dir: str = "images"
    metrics_dir: str = "metrics/single_gpu_mock"
    request_timeout_s: float = 15
    transfer_timeout_s: float = 5
    startup_timeout_s: float = 30
    max_inflight: int = 4
    max_images: int = 8
    max_image_bytes: int = 10485760
    max_image_pixels: int = 16777216
    max_pending_bytes: int = 1048576
    feature_ttl_s: float = 30
    mock_encode_ms: float = 5

    def __post_init__(self):
        if self.host != "127.0.0.1":
            raise ValueError("learning mode binds to 127.0.0.1 only")
        ports = [self.port, self.image_port, self.text_port, self.gloo_port]
        if any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
            raise ValueError("ports must be integers between 1024 and 65535")
        if len(set(ports)) != len(ports):
            raise ValueError("HTTP and Gloo ports must all be distinct")
        for key in ("max_inflight", "max_images", "max_image_bytes",
                    "max_image_pixels", "max_pending_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("request_timeout_s", "transfer_timeout_s", "startup_timeout_s",
                    "feature_ttl_s", "mock_encode_ms"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a finite number")
            if not math.isfinite(value) or value < 0 or (key != "mock_encode_ms" and value == 0):
                raise ValueError(f"invalid {key}")
        if self.transfer_timeout_s >= self.request_timeout_s:
            raise ValueError("transfer_timeout_s must be less than request_timeout_s")
        for key in ("image_dir", "metrics_dir"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty path")
            path = Path(value).expanduser()
            object.__setattr__(self, key, str((REPO_ROOT / path).resolve()))

    @classmethod
    def load(cls, path=DEFAULT_CONFIG, **overrides):
        data = json.loads(Path(path).read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown learning config fields: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self):
        return asdict(self)
