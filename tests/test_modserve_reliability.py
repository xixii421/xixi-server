"""Production protocol/backpressure tests. Models are replaced; modules are not.

PYTHONPATH=src .venv/bin/python tests/test_modserve_reliability.py
The integration tests run production TransferService/TorchTransport with Gloo.
"""
from __future__ import annotations

import hashlib
import ast
import asyncio
import base64
import importlib
import json
import multiprocessing as mp
import socket
import threading
import sys
import types
import pickle
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import requests
import torch
import torch.distributed as dist
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from lmm_serving.common.admission import TextAdmission
from lmm_serving.common.features import FeatureRegistry, feature_bytes
from lmm_serving.common.resources import Capacity, ProtocolError
from lmm_serving.common.routing import make_scheduler
from lmm_serving.common.tensor_transfer import DeviceGate, TorchTransport, TransferService
from lmm_serving.modserve.pipeline import ImagePipeline


def ready_feature(store, tid, owner="request", shape=(2, 4)):
    tensor = torch.ones(shape, dtype=torch.bfloat16)
    store.reserve(tid, owner, tensor.numel() * tensor.element_size(), 5)
    store.begin(tid, owner, "encoding")
    store.publish(tid, tensor)
    return tensor


def spec(tid="request:0", shape=None):
    return {"protocol_version": 2, "transfer_id": tid, "request_id": "request",
            "tensor_shape": shape or [2, 4], "dtype": "bfloat16", "src_rank": 0,
            "dest_rank": 1, "timeout_s": 2}


class CoreTests(unittest.TestCase):
    def test_capacity_oversized_and_atomic_resize(self):
        budget = Capacity(bytes=100, tokens=10)
        with self.assertRaises(ProtocolError) as error:
            budget.reserve("large", bytes=1, tokens=11)
        self.assertEqual(error.exception.status, 413)
        budget.reserve("a", bytes=50, tokens=5)
        budget.reserve("a", bytes=50, tokens=5)
        budget.reserve("b", bytes=50, tokens=5)
        with self.assertRaises(ProtocolError):
            budget.resize("a", bytes=51, tokens=5)
        self.assertEqual(budget.snapshot()["used"], {"bytes": 100, "tokens": 10})
        budget.release("a")
        budget.release("a")
        self.assertEqual(budget.snapshot()["used"]["bytes"], 50)

    def test_capacity_wait_is_bounded_and_wakes_on_release(self):
        budget = Capacity(bytes=1)
        budget.reserve("a", bytes=1)
        with self.assertRaises(ProtocolError):
            budget.reserve("b", bytes=1, wait_s=0.01)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(budget.reserve, "b", bytes=1, wait_s=1)
            budget.release("a")
            pending.result(timeout=2)
        self.assertEqual(budget.snapshot()["reservations"], 1)

    def test_cancelled_encode_cannot_publish_late(self):
        store = FeatureRegistry(100, 10)
        store.reserve("a", "r", 16, 1)
        store.begin("a", "r", "encoding")
        store.cancel("a", "r")
        self.assertEqual(store.snapshot()["used"]["bytes"], 16)
        with self.assertRaises(ProtocolError):
            store.publish("a", torch.ones(8, dtype=torch.bfloat16))
        self.assertEqual(store.snapshot()["used"]["bytes"], 0)

    def test_cancel_before_reserve_and_duplicate_control(self):
        store = FeatureRegistry(100, 10)
        store.cancel("a", "r")
        store.cancel("a", "r")
        with self.assertRaises(ProtocolError):
            store.reserve("a", "r", 16, 1)
        with self.assertRaises(ProtocolError):
            store.cancel("a", "another-request")

    def test_ipc_lease_survives_cancel_and_expiry_until_ack(self):
        store = FeatureRegistry(100, 10)
        ready_feature(store, "r:0", "r")
        lease = store.lease("r:0", lambda tensor: "handle")
        with self.assertRaises(ProtocolError):
            store.lease("r:0", lambda tensor: "second-reader")
        store.cancel("r:0", "r")
        store._items["r:0"].deadline = time.monotonic() - 1
        self.assertEqual(store.snapshot()["used"]["bytes"], 16)
        with self.assertRaises(ProtocolError):
            store.acknowledge("r:0", "wrong-token")
        store.acknowledge("r:0", lease["lease_token"])
        store.acknowledge("r:0", lease["lease_token"])
        self.assertEqual(store.snapshot()["used"]["bytes"], 0)

    def test_expiry_does_not_refund_running_device_work(self):
        store = FeatureRegistry(100, 10)
        ready_feature(store, "r:0", "r")
        store.begin("r:0", "r", "sending")
        store._items["r:0"].deadline = time.monotonic() - 1
        self.assertEqual(store.snapshot()["used"]["bytes"], 16)
        store.finish_device_use("r:0")
        self.assertEqual(store.snapshot()["used"]["bytes"], 0)

    def test_actual_size_reconciliation_keeps_budget_bounded(self):
        store = FeatureRegistry(20, 10)
        store.reserve("a", "r", 8, 1)
        store.begin("a", "r", "encoding")
        store.publish("a", torch.ones(8, dtype=torch.bfloat16))
        self.assertEqual(store.snapshot()["used"]["bytes"], 16)
        store.reserve("b", "r", 4, 1)
        store.begin("b", "r", "encoding")
        with self.assertRaises(ProtocolError):
            store.publish("b", torch.ones(4, dtype=torch.bfloat16))
        self.assertEqual(store.snapshot()["used"]["bytes"], 16)

    def test_text_running_and_uncertain_work_cannot_be_refunded_by_client(self):
        guard = TextAdmission(10, 100, 2)
        guard.reserve("r", 5, 20, 1)
        guard.begin("r")
        guard._records["r"]["deadline"] = time.monotonic() - 1
        self.assertFalse(guard.cancel("r"))
        self.assertEqual(guard.snapshot()["used"], {"tokens": 5, "bytes": 40, "slots": 1})
        guard.complete("r", uncertain=True)
        with self.assertRaises(ProtocolError) as error:
            guard.reserve("next", 1, 1, 1)
        self.assertEqual(error.exception.status, 503)
        self.assertEqual(guard.snapshot()["states"]["quarantined"], 1)

    def test_text_expired_reservation_and_completed_replay(self):
        guard = TextAdmission(10, 100, 2)
        guard.reserve("a", 5, 20, 1)
        guard._records["a"]["deadline"] = time.monotonic() - 1
        self.assertEqual(guard.snapshot()["used"]["bytes"], 0)
        with self.assertRaises(ProtocolError):
            guard.begin("a")
        guard.reserve("b", 5, 20, 1)
        guard.begin("b")
        with self.assertRaises(ProtocolError):
            guard.begin("b")
        guard.complete("b")
        self.assertEqual(guard.snapshot()["used"]["tokens"], 0)

    def test_device_permit_expires_only_before_claim(self):
        gate = DeviceGate()
        old = gate.prepare("a", time.monotonic() - 1)
        current = gate.prepare("b", time.monotonic() + 1)
        with self.assertRaises(ProtocolError):
            gate.claim("a", old)
        gate.claim("b", current)
        gate.release("b", current, prepared_only=True)
        with self.assertRaises(ProtocolError):
            gate.prepare("c", time.monotonic() + 0.01)
        gate.release("b", current)
        gate.prepare("c", time.monotonic() + 1)

    def test_sender_idempotency_epoch_and_metadata(self):
        class Transport:
            count = 0
            def send(self, *_):
                self.count += 1
        store = FeatureRegistry(100, 10)
        ready_feature(store, "request:0")
        transport = Transport()
        service = TransferService("sender", 0, [1], store, transport)
        try:
            token = service.device_gate.prepare("request:0", time.monotonic() + 2)
            data = {**spec(), "channel_epoch": "a" * 32, "sequence": 0, "wire_token": token}
            job = service.submit(data)
            service.wait(job)
            service.wait(service.submit(data))
            self.assertEqual(transport.count, 1)
            with self.assertRaises(ProtocolError):
                service.submit({**data, "sequence": 1})
            with self.assertRaises(ProtocolError):
                service.submit({**data, "tensor_shape": [4, 2]})
            ready_feature(store, "request:1")
            with self.assertRaises(ProtocolError):
                service.submit({**data, "transfer_id": "request:1", "channel_epoch": "b" * 32, "sequence": 1})
        finally:
            service.close()

    def test_uncertain_send_quarantines_node_and_pins_credit(self):
        class BrokenTransport:
            def send(self, *_):
                raise RuntimeError("simulated device failure")
        store = FeatureRegistry(100, 10)
        ready_feature(store, "request:0")
        service = TransferService("sender", 0, [1], store, BrokenTransport())
        try:
            token = service.device_gate.prepare("request:0", time.monotonic() + 2)
            job = service.submit({**spec(), "channel_epoch": "a" * 32, "sequence": 0, "wire_token": token})
            with self.assertRaises(ProtocolError):
                service.wait(job)
            self.assertFalse(service.snapshot()["ready"])
            self.assertEqual(store.snapshot()["used"]["bytes"], 16)
        finally:
            service.close()

    def test_invalid_tensor_size_and_type(self):
        for shape, dtype in (([0], "bfloat16"), ([1.5], "float32"), ([2], []), ([True], "float16")):
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ProtocolError):
                feature_bytes(shape, dtype)

    def test_generation_cannot_exceed_its_reservation(self):
        guard = TextAdmission(10, 100)
        guard.reserve("r", 1, 10, 5)
        with self.assertRaises(ProtocolError):
            guard.begin("r", 2, 20)
        self.assertEqual(guard.snapshot()["states"]["reserved"], 1)
        guard.resize("r", 2, 20)
        guard.begin("r", 2, 20)
        guard.complete("r")

    def test_history_saturation_does_not_evict_live_tombstones(self):
        store = FeatureRegistry(100, 10, history_limit=1)
        store.cancel("a", "r")
        with self.assertRaises(ProtocolError):
            store.reserve("b", "r", 1, 1)
        with self.assertRaises(ProtocolError):
            store.reserve("a", "r", 1, 1)

    def test_source_then_destination_permits_have_no_cross_pair_cycle(self):
        gates = [DeviceGate() for _ in range(4)]
        def acquire_pair(pair):
            src, dst = pair
            owner = f"{src}:{dst}"
            deadline = time.monotonic() + 2
            source = gates[src].prepare(owner, deadline)
            gates[src].claim(owner, source)
            destination = gates[dst].prepare(owner, deadline)
            gates[dst].claim(owner, destination)
            gates[dst].release(owner, destination)
            gates[src].release(owner, source)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(acquire_pair, [(0, 2), (1, 3), (0, 3), (1, 2)]))

    def test_queued_receiver_timeout_never_starts_late_wire_work(self):
        entered, proceed = threading.Event(), threading.Event()
        calls = []
        class Transport:
            def recv(self, metadata, header):
                calls.append(metadata["transfer_id"])
                entered.set()
                if not proceed.wait(2):
                    raise RuntimeError("test stalled")
                return torch.ones((2, 4), dtype=torch.bfloat16)
        store = FeatureRegistry(100, 10)
        service = TransferService("receiver", 1, [0], store, Transport(),
                                  source_urls={0: "source:1"}, make_handle=lambda _: "handle")
        service._source_call = lambda *_, **__: {"wire_token": "token"}
        try:
            first = service.submit({**spec(), "source_url": "source:1"})
            self.assertTrue(entered.wait(1))
            second = service.submit({**spec("request:1"), "source_url": "source:1", "timeout_s": 0.01})
            with self.assertRaises(ProtocolError) as error:
                service.wait(second)
            self.assertEqual(error.exception.status, 504)
            proceed.set()
            lease = service.wait(first)
            store.acknowledge("request:0", lease["lease_token"])
            self.assertTrue(second.done.wait(1))
            self.assertEqual(calls, ["request:0"])
            self.assertTrue(service.snapshot()["ready"])
            self.assertEqual(store.snapshot()["used"]["bytes"], 0)
        finally:
            proceed.set()
            service.close()


class ClientContractTests(unittest.TestCase):
    def test_patched_ipc_ack_only_follows_successful_copy_sync(self):
        patch_path = Path(__file__).resolve().parents[1] / "llm-backend/vllm-lmm-serving.patch"
        source = patch_path.read_text()
        start = source.index("+    async def do_rdma(")
        end = source.index("+    async def fetch_tokens(", start)
        methods = "\n".join(line[1:] for line in source[start:end].splitlines() if line.startswith("+"))
        events = []
        fail_sync = [False]

        class Response:
            status = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass

        class Session(Response):
            def post(self, url, json, timeout):
                events.append("ack")
                self.assert_token = json["lease_token"] == "lease"
                return Response()

        class Tensor:
            def clone(self):
                events.append("clone")
                return self
            def reshape(self, *_):
                return self

        def synchronize():
            events.append("sync")
            if fail_sync[0]:
                raise RuntimeError("copy failed")

        namespace = {
            "List": list, "os": __import__("os"), "pickle": pickle, "base64": base64,
            "aiohttp": types.SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **_: None),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda: types.SimpleNamespace(synchronize=synchronize))),
            "rebuild_cuda_tensor": lambda *_: Tensor(),
        }
        exec(compile(ast.parse("class Parser:\n" + methods), str(patch_path), "exec"), namespace)
        parser = namespace["Parser"]()
        parser._rdma_timeout = 70
        async def receive(session, url, payload, tid):
            self.assertEqual(payload["transfer_id"], "request:0.jpg")
            self.assertEqual(payload["request_id"], "request")
            events.append("recv")
            return {"ipc_handle": base64.b64encode(pickle.dumps(())).decode(), "lease_token": "lease"}
        parser.recv_rdma = receive
        args = ("MLLAMA", "request:0.jpg", "image:8086", [2, 4096], 0, 1, "request", "bfloat16")
        asyncio.run(parser.do_rdma(*args))
        self.assertEqual(events, ["recv", "clone", "sync", "ack"])
        events.clear()
        fail_sync[0] = True
        with self.assertRaises(ValueError):
            asyncio.run(parser.do_rdma(*args))
        self.assertEqual(events, ["recv", "clone", "sync"])

    def test_production_text_proxy_disables_retries_and_retains_uncertain_budget(self):
        calls = []
        apps = []
        fail_backend = [False]
        class Client:
            def __init__(self, **_):
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
            def with_options(self, **options):
                calls.append(options)
                return self
            def create(self, **_):
                if fail_backend[0]:
                    raise TimeoutError("backend outcome unknown")
                return types.SimpleNamespace(id="req", choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = Client
        with patch.dict(sys.modules, {"openai": fake_openai}):
            text_node = importlib.import_module("lmm_serving.common.text_node")
        guard = TextAdmission(10, 100)
        import logging
        with patch.object(text_node, "OpenAI", Client), \
             patch.object(text_node, "_start_vllm", return_value=types.SimpleNamespace(pid=0)), \
             patch.object(text_node, "_wait_for_vllm"), \
             patch.object(text_node.atexit, "register"), \
             patch.object(Flask, "run", lambda app, **_: apps.append(app)):
            text_node.serve(logger=logging.getLogger("test"), hostname="127.0.0.1", port="8080",
                            model_id="test", command=[], out_log="unused", err_log="unused",
                            image_content=lambda _: [{"type": "image"}], admission=guard)
        client = apps[0].test_client()
        guard.reserve("a", 2, 16, 5)
        response = client.post("/generate", json={"request_id": "a", "image_id_shapes": [[2, 4]]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[0]["max_retries"], 0)
        self.assertEqual(guard.snapshot()["used"]["tokens"], 0)
        guard.reserve("b", 2, 16, 5)
        fail_backend[0] = True
        response = client.post("/generate", json={"request_id": "b", "image_id_shapes": [[2, 4]]})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(guard.snapshot()["ready"])
        self.assertEqual(guard.snapshot()["used"]["tokens"], 2)


def production_rank(rank, ports, stop):
    """Replace only encoding/IPC/model computation; use production services."""
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"tcp://127.0.0.1:{ports[2]}", timeout=timedelta(seconds=3))
    store = FeatureRegistry(262144, 16)
    transport = TorchTransport("cpu", {1 - rank: dist.group.WORLD})
    service = TransferService(
        "sender" if rank == 0 else "receiver", rank, [1 - rank], store, transport,
        timeout_s=3, source_urls={0: f"127.0.0.1:{ports[0]}"},
        make_handle=lambda tensor: json.dumps({"sum": tensor.float().sum().item()}),
    )
    app = Flask(f"production-test-{rank}")
    service.install(app)
    guard = TextAdmission(16, 262144, 8)
    if rank == 1:
        guard.install(app)

    @app.post("/generate")
    def generate():
        data = request.get_json()
        if rank == 0:
            tid = data["transfer_id"]
            store.begin(tid, data["request_id"], "encoding")
            try:
                if data["image_id"] == "missing":
                    raise ProtocolError("missing image", 404)
                if data["image_id"] == "slow":
                    time.sleep(0.4)
                rows = 4 if data["image_id"] == "larger" else 2
                value = hashlib.sha256(data["image_id"].encode()).digest()[0]
                tensor = torch.full((rows, 4096), value, dtype=torch.bfloat16)
                store.publish(tid, tensor)
                return jsonify(transfer_id=tid, request_id=data["request_id"],
                               tensor_shape=list(tensor.shape), dtype="bfloat16",
                               preprocessing_time=0, image_encoding_time=0)
            except Exception:
                store.finish_device_use(tid)
                raise
        key = data["request_id"]
        guard.begin(key)
        sums = []
        try:
            for index, tid in enumerate(data["transfer_ids"]):
                payload = {
                    "protocol_version": 2, "transfer_id": tid, "request_id": key,
                    "tensor_shape": data["image_id_shapes"][index], "dtype": data["transfer_dtypes"][index],
                    "src_rank": 0, "dest_rank": 1, "timeout_s": 3,
                    "source_url": data["source_node_ips"][index],
                }
                # Exercise the real receiver HTTP endpoint. CPU handle decoding
                # replaces CUDA IPC only; its lifetime follows the same lease API.
                response = requests.post(f"http://127.0.0.1:{ports[1]}/initiate_tensor_transfer_text_node",
                                         json=payload, timeout=5)
                response.raise_for_status()
                lease = response.json()
                sums.append(json.loads(lease["ipc_handle"])["sum"])
                ack = requests.post(f"http://127.0.0.1:{ports[1]}/release_tensor", json={
                    "transfer_id": tid, "lease_token": lease["lease_token"],
                }, timeout=2)
                ack.raise_for_status()
            return jsonify(content=sums, vllm_req_id="cpu-contract-test")
        finally:
            guard.complete(key)

    server = make_server("127.0.0.1", ports[rank], app, threaded=True)
    server.timeout = 0.1
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        server.server_close()
        service.close()
        dist.destroy_process_group()


class ProductionGlooTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sockets = [socket.socket() for _ in range(3)]
        try:
            for sock in sockets:
                sock.bind(("127.0.0.1", 0))
            cls.ports = [s.getsockname()[1] for s in sockets]
        finally:
            for sock in sockets:
                sock.close()
        ctx = mp.get_context("spawn")
        cls.stop = ctx.Event()
        cls.workers = [ctx.Process(target=production_rank, args=(r, cls.ports, cls.stop)) for r in (0, 1)]
        for worker in cls.workers:
            worker.start()
        deadline = time.monotonic() + 20
        try:
            while True:
                if any(w.exitcode is not None for w in cls.workers):
                    raise AssertionError("production Gloo worker failed at startup")
                try:
                    ready = [requests.get(f"http://127.0.0.1:{p}/transfer_status", timeout=0.2)
                             for p in cls.ports[:2]]
                    if all(r.status_code == 200 for r in ready):
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError("production Gloo startup timeout")
                time.sleep(0.05)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.stop.set()
        for worker in cls.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=3)

    def pipeline(self, tokens=2, timeout=5):
        image, text = [f"127.0.0.1:{p}" for p in self.ports[:2]]
        return ImagePipeline(
            image_scheduler=make_scheduler("least_loaded", [image]),
            text_scheduler=make_scheduler("least_loaded", [text]), image_ranks={image: 0},
            recv_addresses={text: (text, 1)}, token_estimator=lambda _: tokens,
            text_load=lambda t, i: t + i, model_arch="MLLAMA", timeout_s=timeout,
        )

    def assert_empty(self):
        deadline = time.monotonic() + 2
        while True:
            snapshots = [requests.get(f"http://127.0.0.1:{p}/transfer_status", timeout=1).json()
                         for p in self.ports[:2]]
            admission = requests.get(f"http://127.0.0.1:{self.ports[1]}/admission_status", timeout=1).json()
            if all(s["features"]["used"]["bytes"] == 0 for s in snapshots) and admission["used"]["tokens"] == 0:
                self.assertTrue(all(s["ready"] for s in snapshots), snapshots)
                return
            self.assertLess(time.monotonic(), deadline, (snapshots, admission))
            time.sleep(0.05)

    def tearDown(self):
        self.assert_empty()

    def test_concurrent_same_image_and_multi_image_transfers(self):
        pipeline = self.pipeline()
        def send(index):
            return pipeline.run({"req_id": f"r{index}", "prompt": "test", "num_text_tokens": 1,
                                 "image_ids": ["same.jpg", "same.jpg"]})
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(send, range(4)))
        tids = [f["transfer_id"] for r in results for f in r.images]
        self.assertEqual(len(set(tids)), 8)
        value = hashlib.sha256(b"same.jpg").digest()[0] * 2 * 4096
        self.assertTrue(all(r.response["content"] == [value, value] for r in results))

    def test_oversized_request_rejected_before_encoding(self):
        with self.assertRaises(ProtocolError) as error:
            self.pipeline(tokens=17).run({"image_ids": ["same.jpg"]})
        self.assertEqual(error.exception.status, 413)

    def test_partial_encoding_failure_rolls_back_all_reservations(self):
        with self.assertRaises(ProtocolError) as error:
            self.pipeline().run({"image_ids": ["same.jpg", "missing"]})
        self.assertEqual(error.exception.status, 404)

    def test_encoding_timeout_cannot_leave_a_late_feature(self):
        with self.assertRaises(ProtocolError) as error:
            self.pipeline(timeout=0.1).run({"image_ids": ["slow"]})
        self.assertEqual(error.exception.status, 504)
        self.assert_empty()
        self.pipeline().run({"image_ids": ["same.jpg"]})

    def test_actual_token_growth_updates_text_budget(self):
        result = self.pipeline().run({"image_ids": ["larger"]})
        self.assertEqual(result.images[0]["num_image_tokens"], 4)
        self.assertEqual(result.response["content"], [hashlib.sha256(b"larger").digest()[0] * 4 * 4096])


if __name__ == "__main__":
    unittest.main(verbosity=2)
