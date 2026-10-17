"""Real CPU/Gloo integration tests; run with PYTHONPATH=src python tests/test_learning.py."""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from PIL import Image

from lmm_serving.learning.config import LearningConfig, REPO_ROOT
from lmm_serving.learning.protocol import FeatureStore, RequestError, RequestRegistry


class ProtocolTests(unittest.TestCase):
    def test_store_ownership_capacity_expiry_and_cancellation(self):
        store = FeatureStore(128, 10)
        tensor = torch.zeros((4, 8))
        deadline = time.monotonic() + 10
        store.put("a:0", "a", tensor, deadline)
        with self.assertRaises(RequestError):
            store.pop("a:0", "b")
        with self.assertRaises(RequestError):
            store.put("b:0", "b", tensor, deadline)
        self.assertIs(store.pop("a:0", "a"), tensor)
        store.release("a")
        with self.assertRaises(RequestError):
            store.put("a:1", "a", tensor, deadline)
        store.put("b:0", "b", tensor, time.monotonic() + 0.01)
        time.sleep(0.02)
        self.assertEqual(store.stats()["pending_bytes"], 0)

    def test_registry_rejects_active_and_completed_duplicates(self):
        registry = RequestRegistry()
        registry.begin("id")
        with self.assertRaises(RequestError):
            registry.begin("id")
        registry.finish("id")
        self.assertEqual(registry.active_count(), 0)
        with self.assertRaises(RequestError):
            registry.begin("id")

    def test_config_rejects_invalid_topology_and_limits(self):
        for change in (
            {"port": 7081}, {"host": "0.0.0.0"}, {"max_inflight": 0},
            {"transfer_timeout_s": 20}, {"request_timeout_s": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LearningConfig(**change)


class RunningStack:
    def __init__(self, **overrides):
        self.temp = tempfile.TemporaryDirectory(prefix="modserve-learning-")
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for color in ("red", "blue"):
            Image.new("RGB", (2, 2), color).save(self.images / f"{color}.ppm")
        sockets = [socket.socket() for _ in range(4)]
        try:
            for sock in sockets:
                sock.bind(("127.0.0.1", 0))
            ports = [s.getsockname()[1] for s in sockets]
        finally:
            for sock in sockets:
                sock.close()
        self.cfg = LearningConfig(
            port=ports[0], image_port=ports[1], text_port=ports[2], gloo_port=ports[3],
            image_dir=str(self.images), metrics_dir=str(self.root / "metrics"),
            **overrides,
        )
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(self.cfg.to_dict()))
        self.url = f"http://127.0.0.1:{self.cfg.port}"
        self.process = None
        self.output = None

    def start(self):
        self.output = (self.root / "output.log").open("a")
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), OMP_NUM_THREADS="1")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "lmm_serving.learning", "--config", str(self.path)],
            cwd=REPO_ROOT, env=env, stdout=self.output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 35
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise AssertionError((self.root / "output.log").read_text())
                try:
                    response = requests.get(self.url + "/health", timeout=0.5)
                    if response.status_code == 200:
                        return self
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            raise AssertionError("stack startup timeout: " + (self.root / "output.log").read_text())
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        if self.output:
            self.output.close()

    def close(self):
        self.stop()
        self.temp.cleanup()

    def generate(self, **payload):
        payload.setdefault("req_id", uuid.uuid4().hex)
        payload.setdefault("prompt", "hello")
        return requests.post(self.url + "/generate", json=payload, timeout=20)

    def health(self):
        response = requests.get(self.url + "/health", timeout=3)
        response.raise_for_status()
        return response.json()


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = RunningStack().start()

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def tearDown(self):
        health = self.stack.health()
        self.assertEqual(health["active_requests"], 0)
        for worker in health["workers"].values():
            self.assertEqual(worker["pending_features"], 0)
            self.assertEqual(worker["pending_bytes"], 0)

    def test_text_bypasses_image_rank(self):
        before = self.stack.health()["workers"]["image"]["encoded_images"]
        response = self.stack.generate(prompt="hello CPU")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["route"], ["cluster", "text"])
        self.assertEqual(data["features"], [])
        self.assertIn("[MOCK]", data["content"])
        self.assertEqual(self.stack.health()["workers"]["image"]["encoded_images"], before)

    def test_multi_image_duplicate_filenames_and_content_verification(self):
        response = self.stack.generate(image_ids=["red.ppm", "red.ppm", "blue.ppm"])
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["route"], ["cluster", "image", "text"])
        features = data["features"]
        self.assertEqual([f["image_id"] for f in features], ["red.ppm", "red.ppm", "blue.ppm"])
        self.assertEqual(len({f["transfer_id"] for f in features}), 3)
        self.assertEqual(features[0]["checksum"], features[1]["checksum"])
        self.assertNotEqual(features[0]["checksum"], features[2]["checksum"])
        self.assertTrue(all(f["verified"] and f["shape"] == [4, 8] for f in features))
        self.assertGreater(data["metrics"]["transfer_ms"], 0)

    def test_concurrent_requests_using_same_image(self):
        def send(index):
            return self.stack.generate(prompt=f"request-{index}", image_ids=["red.ppm"])
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(send, range(4)))
        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)
        ids = [r.json()["features"][0]["transfer_id"] for r in responses]
        self.assertEqual(len(set(ids)), 4)
        for index, response in enumerate(responses):
            self.assertIn(f"request-{index}", response.json()["content"])

    def test_same_stem_different_extensions(self):
        shutil.copyfile(self.stack.images / "red.ppm", self.stack.images / "same.ppm")
        shutil.copyfile(self.stack.images / "blue.ppm", self.stack.images / "same.crop.ppm")
        response = self.stack.generate(image_ids=["same.ppm", "same.crop.ppm"])
        self.assertEqual(response.status_code, 200, response.text)
        features = response.json()["features"]
        self.assertNotEqual(features[0]["checksum"], features[1]["checksum"])

    def test_missing_image_reclaims_already_encoded_features(self):
        response = self.stack.generate(image_ids=["red.ppm", "missing.ppm"])
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(self.stack.generate(image_ids=["blue.ppm"]).status_code, 200)

    def test_duplicate_request_id(self):
        req_id = uuid.uuid4().hex
        self.assertEqual(self.stack.generate(req_id=req_id).status_code, 200)
        response = self.stack.generate(req_id=req_id)
        self.assertEqual(response.status_code, 409, response.text)

    def test_validation(self):
        for payload in (
            {"image_ids": "red.ppm"}, {"image_ids": ["../outside.ppm"]},
            {"image_ids": ["red.ppm"] * 9}, {"temperature": 0.1},
            {"req_id": ""}, {"prompt": 42},
        ):
            with self.subTest(payload=payload):
                response = self.stack.generate(**payload)
                self.assertEqual(response.status_code, 400, response.text)

    def test_metrics_are_appended_per_image_and_per_request(self):
        req_id = uuid.uuid4().hex
        response = self.stack.generate(req_id=req_id, image_ids=["red.ppm", "blue.ppm"])
        self.assertEqual(response.status_code, 200, response.text)
        rows = [json.loads(line) for line in (self.stack.root / "metrics/requests.jsonl").read_text().splitlines()]
        matching = [r for r in rows if r["req_id"] == req_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(len(matching[0]["features"]), 2)
        self.assertEqual(matching[0]["http_code"], 200)
        self.assertEqual(matching[0]["cleanup_errors"], [])
        self.assertEqual(matching[0]["simulated_stages"], ["encoding", "prefill", "decode"])


class LifecycleTests(unittest.TestCase):
    def test_timeout_cleanup_and_recovery(self):
        stack = RunningStack(request_timeout_s=3, transfer_timeout_s=1, mock_encode_ms=800).start()
        try:
            response = stack.generate(image_ids=["red.ppm"] * 4)
            self.assertEqual(response.status_code, 504, response.text)
            # The encoder may finish its deadline check just after HTTP times out.
            deadline = time.monotonic() + 2
            while True:
                health = stack.health()
                if not health["workers"]["image"]["pending_features"]:
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)
            self.assertEqual(stack.generate(prompt="after timeout").status_code, 200)
            self.assertEqual(stack.generate(image_ids=["blue.ppm"]).status_code, 200)
        finally:
            stack.close()

    def test_admission_limit(self):
        stack = RunningStack(max_inflight=1, mock_encode_ms=500).start()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(stack.generate, image_ids=["red.ppm"])
                deadline = time.monotonic() + 2
                while stack.health()["active_requests"] != 1:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                response = stack.generate()
                self.assertEqual(response.status_code, 429, response.text)
                self.assertEqual(first.result().status_code, 200)
            self.assertEqual(stack.generate().status_code, 200)
        finally:
            stack.close()

    def test_active_duplicate_request_id(self):
        stack = RunningStack(mock_encode_ms=500).start()
        try:
            req_id = uuid.uuid4().hex
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(stack.generate, req_id=req_id, image_ids=["red.ppm"])
                deadline = time.monotonic() + 2
                while stack.health()["active_requests"] != 1:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                response = stack.generate(req_id=req_id)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(first.result().status_code, 200)
        finally:
            stack.close()

    def test_gloo_timeout_stops_the_communicator(self):
        stack = RunningStack(transfer_timeout_s=1).start()
        try:
            workers = stack.health()["workers"]
            key = uuid.uuid4().hex
            # Deliberately receive without a matching send. This exercises an
            # actual Gloo timeout rather than just an HTTP timeout or a mock.
            try:
                response = requests.post(
                    f"http://127.0.0.1:{stack.cfg.text_port}/consume",
                    json={
                        "request_key": key, "deadline": time.monotonic() + 5,
                        "feature": {"transfer_id": key + ":0", "shape": [4, 8],
                                    "dtype": "float32", "checksum": "0" * 64},
                    }, timeout=5,
                )
                self.assertEqual(response.status_code, 503, response.text)
            except requests.ConnectionError:
                pass  # Supervisor may close the socket before the 503 arrives.
            self.assertNotEqual(stack.process.wait(timeout=10), 0)
            for worker in workers.values():
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker["pid"], 0)
        finally:
            stack.close()

    def test_worker_failure_stops_all_ranks(self):
        stack = RunningStack().start()
        try:
            workers = stack.health()["workers"]
            os.kill(workers["image"]["pid"], signal.SIGTERM)
            self.assertNotEqual(stack.process.wait(timeout=10), 0)
            for worker in workers.values():
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker["pid"], 0)
        finally:
            stack.close()

    def test_shutdown_and_restart_on_same_ports(self):
        stack = RunningStack().start()
        try:
            self.assertEqual(stack.generate(image_ids=["red.ppm"]).status_code, 200)
            stack.stop()
            self.assertEqual(stack.process.returncode, 0)
            stack.start()
            self.assertEqual(stack.generate(image_ids=["blue.ppm"]).status_code, 200)
        finally:
            stack.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
