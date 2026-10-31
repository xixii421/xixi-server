"""Real tiny Idefics3 V/P/D forwards and inter-process tensor transfers."""
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request
from lmm_serving.stream_benchmark import open_endpoint as urlopen

from PIL import Image

from lmm_serving.vpd_local import (LocalVPD,VModelSpec,load_stage,reference_generate,validate_messages)
from lmm_serving.vpd_service import VPDHTTPServer
from lmm_serving.stream_benchmark import measure


def image_part(color="red"):
    out=io.BytesIO()
    Image.new("RGB",(16,16),color).save(out,format="PNG")
    return {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(out.getvalue()).decode()}}


def messages(colors=(),text="color?"):
    return [{"role":"user","content":[*[image_part(c) for c in colors],{"type":"text","text":text}]}]


class ContractTests(unittest.TestCase):
    def test_model_configuration_is_a_runtime_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            config=VModelSpec().full_config()
            path=Path(directory)/"config.json"
            path.write_text(json.dumps(config))
            spec=VModelSpec(model_path=directory)
            signature=spec.signature()
            config["text_config"]["vocab_size"]=128
            path.write_text(json.dumps(config))
            self.assertEqual(spec.config()["vocab_size"],64)
            self.assertEqual(spec.signature(),signature)
            self.assertNotEqual(VModelSpec(model_path=directory).signature(),signature)

    def test_stage_models_contain_only_their_assigned_weights(self):
        vision=load_stage(VModelSpec(),0)
        self.assertEqual(set(vision.keys()),{"vision","connector"})
        prefill=load_stage(VModelSpec(),1)
        self.assertIsNone(prefill.model.vision_model)
        self.assertIsNone(prefill.model.connector)

    def test_invalid_and_over_budget_images_rejected_before_encoding(self):
        spec=VModelSpec()
        for limited in (replace(spec,max_image_bytes=1),replace(spec,max_image_pixels=10),
                        replace(spec,max_feature_bytes=1)):
            with self.assertRaises(ValueError):
                validate_messages(limited,messages(["red"]))
        with self.assertRaises(ValueError):
            validate_messages(spec,messages(["red","red","red"]))
        invalid={"type":"image_url","image_url":{"url":"https://example.com/image.png"}}
        with self.assertRaises(ValueError):
            validate_messages(spec,[{"role":"user","content":[invalid]}])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=VModelSpec()
        cls.runtime=LocalVPD(cls.spec)
        cls.runtime.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_single_image_matches_unsplit_pixels_to_model(self):
        request=messages(["red"])
        result=self.runtime.run(request,4)
        self.assertEqual(result["token_ids"],reference_generate(self.spec,request,4))
        self.assertEqual(len(set(result["worker_pids"])),3)
        self.assertFalse(result["vision_bypassed"])
        self.assertGreater(result["feature_bytes"],0)
        self.assertEqual(result["decode_input_lengths"],[1,1,1])

    def test_same_text_different_images_have_distinct_feature_and_context_identity(self):
        red=self.runtime.run(messages(["red"]),3)
        blue=self.runtime.run(messages(["blue"]),3)
        self.assertNotEqual(red["prefill_feature_digest"],blue["prefill_feature_digest"])
        self.assertNotEqual(red["decode_context_digest"],blue["decode_context_digest"])
        self.assertEqual(blue["token_ids"],reference_generate(self.spec,messages(["blue"]),3))

    def test_multiple_images_order_and_duplicate_content(self):
        results=[]
        for colors in (["red","blue"],["blue","red"],["red","red"]):
            request=messages(colors)
            result=self.runtime.run(request,3)
            self.assertEqual(result["token_ids"],reference_generate(self.spec,request,3))
            self.assertEqual(result["image_count"],2)
            results.append(result)
        self.assertNotEqual(results[0]["prefill_feature_digest"],results[1]["prefill_feature_digest"])
        self.assertEqual(results[0]["image_digests"],list(reversed(results[1]["image_digests"])))
        self.assertEqual(results[2]["image_digests"][0],results[2]["image_digests"][1])

    def test_text_only_bypasses_vision(self):
        request=messages()
        result=self.runtime.run(request,3)
        self.assertTrue(result["vision_bypassed"])
        self.assertEqual(result["vision_ms"],0)
        self.assertIsNone(result["prefill_feature_digest"])
        self.assertEqual(result["token_ids"],reference_generate(self.spec,request,3))

    def test_prepared_request_owns_a_snapshot_of_user_messages(self):
        original=messages(["red"])
        prepared=self.runtime.prepare(original,3)
        original[0]["content"].clear()
        result=self.runtime.run_prepared(prepared)
        self.assertEqual(result["token_ids"],reference_generate(self.spec,messages(["red"]),3))
        self.assertEqual(result["image_count"],1)

    def test_concurrent_callers_preserve_ordered_tokens_and_request_ownership(self):
        requests=[messages(["red"]),messages(["blue"])]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda r:self.runtime.run(r,3),requests))
        self.assertNotEqual(results[0]["request_id"],results[1]["request_id"])
        for request,result in zip(requests,results):
            self.assertEqual(result["token_ids"],reference_generate(self.spec,request,3))


class FailureTests(unittest.TestCase):
    def test_consumer_cancellation_reaps_all_stages(self):
        def disconnected(_):
            raise BrokenPipeError("consumer disconnected")
        with LocalVPD() as runtime:
            with self.assertRaises(BrokenPipeError):
                runtime.run(messages(["red"]),4,on_token=disconnected)
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))

    def test_preprocessing_rejection_does_not_poison_wire(self):
        with LocalVPD(VModelSpec(max_prompt_tokens=8)) as runtime:
            with self.assertRaises(ValueError):
                runtime.run(messages(["red"],"too much text"),2)
            self.assertEqual(len(runtime.run(messages(["red"],"a"),2)["token_ids"]),2)

    def test_worker_failure_reclaims_all_three_processes(self):
        with LocalVPD() as runtime:
            runtime.processes[2].terminate()
            runtime.processes[2].join(timeout=3)
            with self.assertRaises(RuntimeError):
                runtime.run(messages(["red"]),2)
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime=LocalVPD()
        cls.runtime.__enter__()
        cls.server=VPDHTTPServer(("127.0.0.1",0),cls.runtime)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.endpoint=f"http://127.0.0.1:{cls.server.server_port}/v1/chat/completions"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()
        cls.runtime.close()

    def test_real_http_sse_consumes_image_and_reports_expanded_prompt_tokens(self):
        payload={"model":"tiny-idefics3","messages":messages(["red","blue"]),"max_tokens":3}
        result=measure(self.endpoint,payload,timeout_s=20)
        self.assertEqual(result["status"],"ok",result["error"])
        self.assertEqual(result["completion_tokens"],3)
        self.assertGreaterEqual(result["prompt_tokens"],8)

    def test_image_limit_413_and_recovery(self):
        payload={"model":"tiny-idefics3","messages":messages(["red"]*3),"max_tokens":2}
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.endpoint,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"}))
        self.assertEqual(error.exception.code,413)
        error.exception.close()
        payload["messages"]=messages(["blue"])
        self.assertEqual(measure(self.endpoint,payload,timeout_s=20)["status"],"ok")

    def test_http_validates_images_once_for_stream_and_nonstream(self):
        payload={"model":"tiny-idefics3","messages":messages(["red"]),"max_tokens":3}
        with patch("lmm_serving.vpd_local.validate_messages",wraps=validate_messages) as validate:
            with urlopen(Request(self.endpoint,data=json.dumps(payload).encode(),
                         headers={"Content-Type":"application/json"}),timeout=20) as response:
                self.assertEqual(len(json.load(response)["backend_metrics"]["token_ids"]),3)
            self.assertEqual(validate.call_count,1)
        with patch("lmm_serving.vpd_local.validate_messages",wraps=validate_messages) as validate:
            self.assertEqual(measure(self.endpoint,payload,timeout_s=20)["status"],"ok")
            self.assertEqual(validate.call_count,1)

    def test_invalid_stream_request_rejected_before_sse_headers(self):
        payload={"model":"tiny-idefics3","messages":messages(["red"]*3),"max_tokens":2,"stream":True}
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.endpoint,data=json.dumps(payload).encode(),
                            headers={"Content-Type":"application/json"}),timeout=10)
        self.assertEqual(error.exception.code,413)
        error.exception.close()

    def test_body_limit_rejected_without_waiting_for_payload(self):
        request=Request(self.endpoint,data=b"{}",headers={"Content-Type":"application/json",
                         "Content-Length":str(self.server.max_body_bytes+1)})
        with self.assertRaises(HTTPError) as error:
            urlopen(request,timeout=3)
        self.assertEqual(error.exception.code,413)
        error.exception.close()


class LauncherTests(unittest.TestCase):
    def test_unified_launcher_serves_real_three_stage_request(self):
        process=subprocess.Popen([sys.executable,"-m","lmm_serving.launcher","vpd",
                                  "--port","0","--dtype","float32"],
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,start_new_session=True)
        try:
            deadline=time.monotonic()+40
            info=None
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    self.fail(process.stderr.read())
                ready,_,_=select.select([process.stdout],[],[],.2)
                if ready:
                    line=process.stdout.readline()
                    if line.startswith("{"):
                        info=json.loads(line)
                        break
            self.assertIsNotNone(info,"launcher readiness timeout")
            self.assertEqual(info["stages"],["vision","prefill","decode"])
            result=measure(info["url"],{"model":info["model"],"messages":messages(["blue"]),
                                      "max_tokens":3},timeout_s=20)
            self.assertEqual(result["status"],"ok",result["error"])
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__=="__main__":
    unittest.main(verbosity=2)
