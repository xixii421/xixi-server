"""Real tiny Qwen3 forwards and two-process Gloo; no mocked model execution."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lmm_serving.pd_local import LocalPD, ModelSpec, reference_generate, validate_header


class MetadataTests(unittest.TestCase):
    def test_reject_invalid_and_oversized_requests(self):
        spec = ModelSpec()
        for tokens in ([], [True], [-1], [64], [1]*257):
            with self.assertRaises(ValueError):
                spec.validate(tokens, 1)
        for count in (0, -1, True, 33):
            with self.assertRaises(ValueError):
                spec.validate([1], count)
        with self.assertRaises(ValueError):
            replace(spec, max_cache_bytes=1).validate([1], 2)
        with self.assertRaises(ValueError):
            replace(spec, max_prompt_tokens=512).validate([1]*512, 1)

    def test_reject_invalid_configuration(self):
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                ModelSpec(threads=value)
        for value in (0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                LocalPD(request_timeout_s=value)

    def test_handoff_checks_request_model_prompt_and_sequence(self):
        spec = ModelSpec()
        request = {"request_id": "abc", "sequence": 1, "tokens": [1,2,3], "new_tokens": 2}
        header = {"version": 1, "request_id": "abc", "sequence": 1,
                  "signature": spec.signature(), "prompt_length": 3, "new_tokens": 2,
                  "dtype": "float32", "first_token": 4, "digests": ["0"*64]*4,
                  "prompt_digest": hashlib.sha256(json.dumps([1,2,3]).encode()).hexdigest()}
        validate_header(header, spec, request)
        for key, value in (("request_id", "other"), ("sequence", 2), ("signature", "wrong"),
                           ("dtype", "bfloat16"), ("first_token", 64), ("digests", []),
                           ("prompt_digest", "wrong")):
            with self.assertRaises(ValueError, msg=key):
                validate_header({**header, key: value}, spec, request)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ModelSpec()
        cls.runtime = LocalPD(cls.spec)
        cls.runtime.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_matches_unsplit_model_and_does_not_recompute_prompt(self):
        tokens = [1, 5, 9, 12, 7]
        expected = reference_generate(self.spec, tokens, 5)
        result = self.runtime.run(tokens, 5)
        self.assertEqual(result["token_ids"], expected)
        self.assertEqual(result["decode_input_lengths"], [1]*4)
        self.assertEqual(result["received_cache_tokens"], len(tokens))
        self.assertEqual(len(set(result["worker_pids"])), 2)
        self.assertNotIn(os.getpid(), result["worker_pids"])
        self.assertGreater(result["kv_payload_bytes"], 0)
        self.assertTrue(all(t >= 0 for t in result["backend_tbt_ms"]))

    def test_first_token_only_still_transfers_and_acknowledges_cache(self):
        result = self.runtime.run([1, 2, 3], 1)
        self.assertEqual(result["token_ids"], reference_generate(self.spec, [1,2,3], 1))
        self.assertEqual(result["decode_input_lengths"], [])
        self.assertEqual(result["backend_tbt_ms"], [])

    def test_sequential_requests_do_not_reuse_another_prompt_cache(self):
        prompts = [[1, 5], [1, 5, 9], [1, 5]]
        results = [self.runtime.run(tokens, 3) for tokens in prompts]
        for tokens, result in zip(prompts, results):
            self.assertEqual(result["token_ids"], reference_generate(self.spec, tokens, 3))
        self.assertEqual(len({r["request_id"] for r in results}), 3)

    def test_concurrent_callers_are_serialized_without_identity_mixups(self):
        prompts = [[1, 4, 5], [1, 6, 7]]
        expected = [reference_generate(self.spec, tokens, 3) for tokens in prompts]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda tokens: self.runtime.run(tokens, 3), prompts))
        self.assertEqual([r["token_ids"] for r in results], expected)
        self.assertNotEqual(results[0]["request_id"], results[1]["request_id"])

    def test_rejected_request_does_not_poison_channel(self):
        with self.assertRaises(ValueError):
            self.runtime.run([9999], 2)
        self.assertEqual(len(self.runtime.run([1,2], 2)["token_ids"]), 2)

    def test_token_callback_matches_generation_order(self):
        seen = []
        result = self.runtime.run([1,4,5], 5, on_token=seen.append)
        self.assertEqual(seen, result["token_ids"])


class LifecycleTests(unittest.TestCase):
    def test_bfloat16_cache_transfer_matches_reference(self):
        spec = ModelSpec(dtype="bfloat16")
        expected = reference_generate(spec, [1, 4, 8], 4)
        with LocalPD(spec) as runtime:
            result = runtime.run([1, 4, 8], 4)
        self.assertEqual(result["token_ids"], expected)
        self.assertEqual(result["kv_payload_bytes"], 2*2*2*3*8*2)
        self.assertTrue(all(not p.is_alive() for p in runtime.processes))

    def test_worker_failure_closes_both_workers_and_rejects_reuse(self):
        with LocalPD() as runtime:
            runtime.processes[1].terminate()
            runtime.processes[1].join(timeout=3)
            with self.assertRaises(RuntimeError):
                runtime.run([1,2], 3)
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))
            with self.assertRaises(RuntimeError):
                runtime.run([1,2], 3)

    def test_startup_failure_reaps_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = LocalPD(ModelSpec(model_path=folder), startup_timeout_s=20)
            with self.assertRaises(RuntimeError):
                runtime.__enter__()
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))

    def test_consumer_failure_stops_uncertain_channel(self):
        def disconnected(_):
            raise BrokenPipeError("consumer disconnected")
        with LocalPD() as runtime:
            with self.assertRaises(BrokenPipeError):
                runtime.run([1,2,3], 5, on_token=disconnected)
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))

    def test_cli_end_to_end_and_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"result.json"
            output = output.parent/"new-results"/output.name
            command = [sys.executable, "-m", "lmm_serving.pd_local", "--verify",
                       "--new-tokens", "3", "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text())
            self.assertTrue(data["greedy_matches_reference"])
            before = output.read_bytes()
            completed = subprocess.run(command, capture_output=True, timeout=20)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
