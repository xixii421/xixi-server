"""Feature reuse correctness, bounded residency, and real-model integration."""
import unittest
import threading

import torch

from lmm_serving.vision_cache import VisionFeatureCache
from lmm_serving.vpd_local import LocalVPD,VModelSpec,reference_generate
from lmm_serving.vpd_service import VPDHTTPServer
from lmm_serving.stream_benchmark import measure
from lmm_serving.monolithic_local import MonolithicVLM
from test_vpd import messages


class CacheTests(unittest.TestCase):
    def test_persistent_monolithic_baseline_matches_independent_reference(self):
        spec=VModelSpec()
        request=messages(["blue"])
        with MonolithicVLM(spec) as runtime:
            model=runtime.model
            first=runtime.run(request,3)
            second=runtime.run(request,3)
            self.assertIs(runtime.model,model)
        self.assertEqual(first["token_ids"],reference_generate(spec,request,3))
        self.assertEqual(first["token_ids"],second["token_ids"])

    def test_byte_budget_lru_and_oversized_entry(self):
        cache=VisionFeatureCache(16)
        cache.put("a",torch.ones(2,dtype=torch.float32))
        cache.put("b",torch.ones(2,dtype=torch.float32))
        self.assertIsNotNone(cache.get("a"))
        cache.put("c",torch.ones(2,dtype=torch.float32))
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.snapshot()["resident_bytes"],16)
        self.assertEqual(cache.evictions,1)
        self.assertFalse(cache.put("large",torch.ones(5)))
        self.assertEqual(cache.snapshot()["entries"],2)

    def test_copy_ownership_and_no_batch_storage_retention(self):
        cache=VisionFeatureCache(8)
        batch=torch.ones(100,2)
        cache.put("a",batch[0])
        batch[0].fill_(9)
        returned=cache.get("a")
        self.assertEqual(returned.tolist(),[1,1])
        returned.fill_(7)
        self.assertEqual(cache.get("a").tolist(),[1,1])
        self.assertEqual(cache.items["a"].untyped_storage().nbytes(),8)

    def test_disabled_cache_and_replacement_accounting(self):
        cache=VisionFeatureCache(0)
        self.assertFalse(cache.put("a",torch.ones(1)))
        self.assertIsNone(cache.get("a"))
        cache=VisionFeatureCache(16)
        cache.put("a",torch.ones(2))
        cache.put("a",torch.ones(1))
        self.assertEqual(cache.used_bytes,4)


class ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=VModelSpec(vision_cache_bytes=4096)
        cls.runtime=LocalVPD(cls.spec)
        cls.runtime.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_same_image_new_question_reuses_features_without_reusing_text_kv(self):
        self.runtime.run(messages(["red"],"first"),3)
        request=messages(["red"],"another question")
        result=self.runtime.run(request,3)
        self.assertEqual(result["cache_hits"],1)
        self.assertEqual(result["vision_images_encoded"],0)
        self.assertEqual(result["vision_compute_ms"],0)
        self.assertEqual(result["token_ids"],reference_generate(self.spec,request,3))

    def test_reordered_partial_hits_preserve_image_and_token_alignment(self):
        self.runtime.run(messages(["red"]),3)
        self.runtime.run(messages(["red","blue"]),3)
        request=messages(["blue","red"])
        result=self.runtime.run(request,3)
        self.assertEqual(result["cache_hits"],2)
        self.assertEqual(result["vision_images_encoded"],0)
        self.assertEqual(result["token_ids"],reference_generate(self.spec,request,3))

    def test_sse_reports_cache_observations_for_actual_request(self):
        self.runtime.run(messages(["red"]),3)
        server=VPDHTTPServer(("127.0.0.1",0),self.runtime)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            result=measure(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                {"model":"tiny-idefics3","messages":messages(["red"]),"max_tokens":3,
                 "stream_options":{"include_backend_metrics":True}})
            self.assertEqual(result["status"],"ok",result["error"])
            self.assertEqual(result["backend_metrics"]["cache_hits"],1)
            self.assertEqual(result["backend_metrics"]["vision_images_encoded"],0)
            self.assertEqual(result["prompt_tokens"],result["backend_metrics"]["received_cache_tokens"])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


class EvictionTests(unittest.TestCase):
    def test_eviction_and_duplicate_images_remain_correct(self):
        spec=VModelSpec(vision_cache_bytes=512)
        with LocalVPD(spec) as runtime:
            duplicate=runtime.run(messages(["red","red"]),3)
            self.assertEqual(duplicate["vision_images_encoded"],1)
            self.assertEqual(duplicate["intra_request_reuses"],1)
            request=messages(["red","blue"])
            result=runtime.run(request,3)
            self.assertLessEqual(result["cache_resident_bytes"],512)
            self.assertGreaterEqual(result["cache_evictions"],1)
            self.assertEqual(result["token_ids"],reference_generate(spec,request,3))
            self.assertEqual(runtime.run(messages(["red"]),2)["cache_misses"],1)


if __name__=="__main__":
    unittest.main(verbosity=2)
