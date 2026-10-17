from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_CONFIG, LearningConfig


def add_arguments(parser):
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="learning JSON configuration")
    parser.add_argument("--port", type=int, help="cluster HTTP port")
    parser.add_argument("--image-dir", help="directory containing input images")
    parser.add_argument("--metrics-dir", help="directory for requests.jsonl")
    parser.add_argument("--demo", action="store_true", help="run text + a generated temporary image, then stop")


def run(args):
    cfg = LearningConfig.load(
        args.config, port=args.port, image_dir=args.image_dir, metrics_dir=args.metrics_dir,
    )
    try:
        from .runtime import run_stack
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Learning dependencies are missing. Install requirements-learning.txt "
            "in a virtual environment (no vLLM/CUDA needed)."
        ) from error
    run_stack(cfg, demo=args.demo)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the CPU/Gloo single_gpu_mock learning stack")
    add_arguments(parser)
    try:
        return run(parser.parse_args())
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        print(f"learning mode: {error}", file=sys.stderr)
        return 1
