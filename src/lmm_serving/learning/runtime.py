from __future__ import annotations

import json
import logging
import multiprocessing as mp
import signal
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from PIL import Image
from werkzeug.serving import make_server

from .cluster import create_cluster_app
from .workers import worker_main


class LearningStack:
    """Parent = cluster manager; two spawned workers = Gloo ranks 0 and 1."""

    def __init__(self, cfg):
        self.cfg = cfg
        context = mp.get_context("spawn")
        self.stop = context.Event()
        self.fatal = context.Event()
        self.processes = []
        self.server = None

    def _check_workers(self):
        for process in self.processes:
            if process.exitcode is not None:
                raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
        if self.fatal.is_set():
            raise RuntimeError("Gloo transport failed; all ranks must be restarted")

    def start(self):
        # Catch occupied ports before starting either rank. Do not kill processes
        # belonging to another experiment or touch production cleanup scripts.
        probes = []
        try:
            for port in (self.cfg.port, self.cfg.image_port, self.cfg.text_port, self.cfg.gloo_port):
                sock = socket.socket()
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probes.append(sock)
                sock.bind((self.cfg.host, port))
        finally:
            for sock in probes:
                sock.close()
        try:
            context = mp.get_context("spawn")
            for role in ("image", "text"):
                process = context.Process(
                    target=worker_main, name=f"learning-{role}",
                    args=(role, self.cfg.to_dict(), self.stop, self.fatal),
                )
                process.start()
                self.processes.append(process)
            deadline = time.monotonic() + self.cfg.startup_timeout_s
            ready = set()
            while len(ready) != 2:
                if self.stop.is_set():
                    raise RuntimeError("learning startup interrupted")
                self._check_workers()
                if time.monotonic() >= deadline:
                    raise TimeoutError("learning workers did not become ready in time")
                for role, port in (("image", self.cfg.image_port), ("text", self.cfg.text_port)):
                    if role in ready:
                        continue
                    try:
                        response = requests.get(f"http://{self.cfg.host}:{port}/health", timeout=0.3)
                        if response.status_code == 200 and response.json().get("ready"):
                            ready.add(role)
                    except (requests.RequestException, ValueError):
                        pass
                self.stop.wait(0.05)
            app = create_cluster_app(self.cfg, self.fatal)
            self.server = make_server(self.cfg.host, self.cfg.port, app, threaded=True)
            self.server.timeout = 0.2
            print(f"[single_gpu_mock] cluster ready http://{self.cfg.host}:{self.cfg.port} (CPU/Gloo; simulated model)", flush=True)
        except BaseException:
            self.close()
            raise

    def serve(self):
        while not self.stop.is_set():
            self._check_workers()
            self.server.handle_request()

    def close(self):
        self.stop.set()
        if self.server:
            self.server.server_close()
            self.server = None
        for process in self.processes:
            process.join(timeout=2)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)


def run_stack(cfg, demo=False):
    if demo:
        # Demo input is generated in isolation and removed after all workers exit.
        with TemporaryDirectory(prefix="modserve-demo-") as image_dir:
            Image.new("RGB", (2, 2), "red").save(Path(image_dir) / "red.ppm")
            return _run_stack(replace(cfg, image_dir=image_dir), demo=True)
    return _run_stack(cfg)


def _run_stack(cfg, demo=False):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    stack = LearningStack(cfg)
    errors = []
    demo_thread = None

    def demo_requests():
        try:
            for payload in (
                {"req_id": "demo-text", "prompt": "hello"},
                {"req_id": "demo-image", "prompt": "Describe the image.", "image_ids": ["red.ppm"]},
            ):
                response = requests.post(
                    f"http://{cfg.host}:{cfg.port}/generate", json=payload,
                    timeout=cfg.request_timeout_s + 5,
                )
                response.raise_for_status()
                print(json.dumps(response.json(), ensure_ascii=False, indent=2), flush=True)
        except Exception as error:
            errors.append(error)
        finally:
            stack.stop.set()

    previous_handlers = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, lambda *_: stack.stop.set())
        stack.start()
        if demo:
            demo_thread = threading.Thread(target=demo_requests, daemon=True)
            demo_thread.start()
        stack.serve()
        if errors:
            raise RuntimeError(f"learning demo failed: {errors[0]}")
    finally:
        stack.close()
        if demo_thread:
            demo_thread.join(timeout=1)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
