"""End-to-end HTTP -> real Qwen3 P -> Gloo KV -> real Qwen3 D -> SSE."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request
from lmm_serving.stream_benchmark import open_endpoint as urlopen

from lmm_serving.pd_local import LocalPD, reference_generate
from lmm_serving.pd_service import PDHTTPServer
from lmm_serving.stream_benchmark import measure


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LocalPD()
        cls.runtime.__enter__()
        cls.server = PDHTTPServer(("127.0.0.1", 0), cls.runtime)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}/v1/chat/completions"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()
        cls.runtime.close()

    def payload(self):
        return {"model": "tiny-qwen3", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4, "temperature": 0}

    def post(self, payload):
        with urlopen(Request(self.endpoint, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}), timeout=10) as response:
            return json.load(response)

    def test_nonstream_output_matches_unsplit_real_model(self):
        result = self.post(self.payload())
        tokens = [1]+[3+(b % 61) for b in b"hi"]
        expected = reference_generate(self.runtime.spec, tokens, 4)
        self.assertEqual(result["backend_metrics"]["token_ids"], expected)
        self.assertEqual(result["usage"]["completion_tokens"], 4)

    def test_streaming_benchmark_runs_against_real_pd_service(self):
        result = measure(self.endpoint, self.payload(), timeout_s=10)
        self.assertEqual(result["status"], "ok", result["error"])
        self.assertEqual(result["completion_tokens"], 4)
        self.assertGreaterEqual(result["ttft_ms"], 0)
        self.assertLessEqual(result["ttft_ms"], result["e2e_ms"])
        self.assertIsNone(result["tbt_ms"])

    def test_rejects_multimodal_and_unknown_generation_options(self):
        invalid = [{**self.payload(), "top_p": .9},
                   {**self.payload(), "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
                   {**self.payload(), "model": "other"},
                   {**self.payload(), "max_tokens": 10000}]
        for payload in invalid:
            with self.assertRaises(HTTPError) as error:
                self.post(payload)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        self.assertEqual(self.post(self.payload())["usage"]["completion_tokens"], 4)

    def test_admission_limit_returns_429_and_recovers(self):
        self.server.slots.acquire()
        self.server.slots.acquire()
        try:
            with self.assertRaises(HTTPError) as error:
                self.post(self.payload())
            self.assertEqual(error.exception.code, 429)
            error.exception.close()
        finally:
            self.server.slots.release()
            self.server.slots.release()
        self.assertEqual(self.post(self.payload())["usage"]["completion_tokens"], 4)

    def test_health_reflects_live_workers(self):
        with urlopen(self.endpoint.replace("/v1/chat/completions", "/health")) as response:
            self.assertTrue(json.load(response)["ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
