"""Exercise measurement against actual local HTTP/SSE, without model timing claims."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request

from lmm_serving.stream_benchmark import measure, sse_events, summarize, open_endpoint


def event(value):
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.payloads.append(payload)
        if self.path == "/failed":
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if self.path == "/json" else "text/event-stream")
        self.end_headers()
        if self.path == "/json":
            self.wfile.write(b"{}")
            return
        if self.path == "/stream-error":
            self.wfile.write(event({"error": {"message": "engine failure"}}))
            return
        self.wfile.write(event({"choices": [{"delta": {"role": "assistant", "content": ""}}]}))
        self.wfile.write(event({"choices": [{"delta": {"content": "你好，多个Token"}}]}))
        if self.path == "/truncated":
            return
        self.wfile.write(event({"choices": [{"delta": {"content": "！"}, "finish_reason": "stop"}]}))
        if self.path != "/no-usage":
            self.wfile.write(event({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}}))
        self.wfile.write(b"data: [DONE]\r\n\r\n")


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.payloads = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path="/ok"):
        return measure(f"http://127.0.0.1:{self.server.server_port}{path}",
                       {"model": "test", "messages": []}, timeout_s=2)

    def test_real_http_stream_and_usage(self):
        result = self.request()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], "你好，多个Token！")
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(len(result["content_event_times_s"]), 2)
        self.assertIsNone(result["tbt_ms"])
        self.assertLessEqual(result["ttft_ms"], result["e2e_ms"])
        self.assertTrue(self.server.payloads[-1]["stream"])

    def test_loopback_bypasses_proxy_even_without_no_proxy(self):
        with patch.dict(os.environ,{"http_proxy":"http://127.0.0.1:1",
                        "https_proxy":"http://127.0.0.1:1","no_proxy":"","NO_PROXY":""}), \
             patch("lmm_serving.stream_benchmark.urlopen",side_effect=AssertionError("default proxy route used")):
            result=self.request()
        self.assertEqual(result["status"],"ok",result["error"])
        self.assertEqual(result["connection_route"],"direct_loopback")

    def test_remote_endpoints_keep_default_proxy_policy(self):
        sentinel=object()
        with patch("lmm_serving.stream_benchmark.urlopen",return_value=sentinel) as default:
            self.assertIs(open_endpoint(Request("http://example.invalid/api")),sentinel)
        default.assert_called_once()

    def test_string_urls_support_health_checks(self):
        with open_endpoint(f"http://127.0.0.1:{self.server.server_port}/health",timeout=2) as response:
            self.assertTrue(json.load(response)["ok"])

    def test_role_only_event_does_not_start_ttft(self):
        times = iter([10., 10.5, 10.7, 11.])
        result = measure(f"http://127.0.0.1:{self.server.server_port}/ok",
                         {}, clock=lambda: next(times))
        self.assertEqual(result["ttft_ms"], 500)
        self.assertAlmostEqual(result["content_event_gaps_ms"][0], 200)

    def test_missing_usage_is_not_inferred_from_chunk_count(self):
        result = self.request("/no-usage")
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(summarize([result], 1)["successful_output_tokens_per_s"])

    def test_truncated_stream_is_a_failure(self):
        self.assertEqual(self.request("/truncated")["status"], "failed")

    def test_http_error_is_recorded(self):
        result = self.request("/failed")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["http_status"], 503)

    def test_stream_error_is_recorded(self):
        self.assertIn("engine failure", self.request("/stream-error")["error"])

    def test_nonstreaming_endpoint_rejected(self):
        self.assertEqual(self.request("/json")["status"], "failed")

    def test_sse_comments_multiline_and_partial_event(self):
        source = io.BytesIO(b': heartbeat\r\ndata: first\r\ndata: second\r\n\r\ndata: partial')
        self.assertEqual(list(sse_events(source)), ["first\nsecond"])

    def test_failures_count_against_slo_attainment(self):
        records = [{"status": "ok", "ttft_ms": 10, "completion_tokens": 8},
                   {"status": "ok", "ttft_ms": 100, "completion_tokens": 2},
                   {"status": "failed", "ttft_ms": 1, "completion_tokens": 100}]
        summary = summarize(records, 2, 50)
        self.assertEqual(summary["successful_output_tokens_per_s"], 5)
        self.assertEqual(summary["ttft_only_goodput_requests_per_s"], .5)
        self.assertAlmostEqual(summary["ttft_only_slo_attainment"], 1/3)
        self.assertEqual(summary["ttft_p99_ms"], 100)

    def test_empty_sample_and_invalid_duration(self):
        self.assertIsNone(summarize([], 1)["ttft_p99_ms"])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize([], value)

    def test_cli_records_each_request_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            payload, output = Path(folder)/"payload.json", Path(folder)/"results.jsonl"
            payload.write_text(json.dumps({"model": "test", "messages": []}), encoding="utf-8")
            command = [sys.executable, "-m", "lmm_serving.stream_benchmark",
                       "--endpoint", f"http://127.0.0.1:{self.server.server_port}/ok",
                       "--payload", str(payload), "--output", str(output),
                       "--requests", "3", "--concurrency", "2"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sorted(r["benchmark_index"] for r in records[:-1]), [0, 1, 2])
            self.assertEqual(records[-1]["summary"]["succeeded"], 3)
            before = output.read_bytes()
            repeated = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
