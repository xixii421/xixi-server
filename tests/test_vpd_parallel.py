"""真实模型的并发路由回归；人工延迟只用于同步验证，不代表性能收益。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import time
import unittest
from unittest.mock import patch

from lmm_serving import vpd_local
from lmm_serving.vpd_local import CapacityError, LocalVPD, VModelSpec, reference_generate
from lmm_serving.vpd_service import VPDHTTPServer
from lmm_serving.stream_benchmark import measure
from test_vpd import messages


POOLS=(("cpu",),("cpu",),("cpu","cpu"))
BASE_WORKER=vpd_local.vpd_worker


def slow_decode_worker(rank,spec,store,control,events,timeout):
    """保留真实计算，只放慢 D，给并发区间断言留下确定的重叠窗口。"""
    if spec.worker_stages[rank]==2:
        forward=vpd_local.language_forward
        def delayed_forward(*args,**kwargs):
            time.sleep(.04)
            return forward(*args,**kwargs)
        vpd_local.language_forward=delayed_forward
    return BASE_WORKER(rank,spec,store,control,events,timeout)


@contextmanager
def concurrent_runtime(*,max_inflight=4):
    spec=VModelSpec(stage_pools=POOLS,vision_cache_bytes=4096)
    runtime=LocalVPD(spec,max_inflight=max_inflight,request_timeout_s=20)
    # spawn 目标是本文件的顶层函数；child 内才包装 forward，无产品测试开关。
    with patch("lmm_serving.vpd_local.vpd_worker",slow_decode_worker):
        runtime.__enter__()
    try:
        yield runtime
    finally:
        runtime.close()


class ParallelPipelineTests(unittest.TestCase):
    def assert_decode_overlap(self,results):
        self.assertEqual({r["route"]["decode"] for r in results},{2,3})
        self.assertLess(max(r["decode_started_s"] for r in results),
                        min(r["decode_finished_s"] for r in results))
        for result in results:
            self.assertTrue(result["pipeline_concurrent"])
            self.assertGreaterEqual(result["queue_wait_ms"],0)

    def test_actual_decode_intervals_overlap_and_tokens_remain_request_owned(self):
        requests=[messages(["red"],"first question"),messages(["blue"],"second question")]
        spec=VModelSpec(stage_pools=POOLS)
        references=[reference_generate(spec,request,24) for request in requests]
        barrier=threading.Barrier(2)
        with concurrent_runtime() as runtime:
            def generate(request):
                tokens=[]
                barrier.wait(timeout=5)
                result=runtime.run(request,24,on_token=tokens.append)
                return result,tokens
            with ThreadPoolExecutor(max_workers=2) as pool:
                outputs=list(pool.map(generate,requests))
            results=[result for result,_ in outputs]
            self.assert_decode_overlap(results)
            self.assertEqual(len({r["request_id"] for r in results}),2)
            self.assertNotEqual(results[0]["decode_context_digest"],results[1]["decode_context_digest"])
            for (result,tokens),reference in zip(outputs,references):
                self.assertEqual(tokens,reference)
                self.assertEqual(result["token_ids"],reference)

    def test_concurrent_http_sse_preserves_each_stream_and_backend_route(self):
        requests=[messages(["red"],"first"),messages(["blue"],"second")]
        references=[reference_generate(VModelSpec(),request,16) for request in requests]
        with concurrent_runtime() as runtime:
            server=VPDHTTPServer(("127.0.0.1",0),runtime)
            server_thread=threading.Thread(target=server.serve_forever,daemon=True)
            server_thread.start()
            barrier=threading.Barrier(2)
            try:
                endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                def generate(request):
                    barrier.wait(timeout=5)
                    return measure(endpoint,{"model":server.model_name,"messages":request,
                        "max_tokens":16,"stream_options":{"include_backend_metrics":True}},timeout_s=20)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    records=list(pool.map(generate,requests))
                for record,reference in zip(records,references):
                    self.assertEqual(record["status"],"ok",record["error"])
                    self.assertEqual(record["content"],"".join(f"{token} " for token in reference))
                    self.assertEqual(record["backend_metrics"]["token_ids"],reference)
                self.assert_decode_overlap([r["backend_metrics"] for r in records])
            finally:
                server.shutdown()
                server_thread.join(timeout=5)
                server.server_close()

    def test_four_admitted_requests_queue_and_reuse_decode_slots(self):
        requests=[messages([color],f"question {index}")
                  for index,color in enumerate(("red","blue","red","blue"))]
        references=[reference_generate(VModelSpec(),request,8) for request in requests]
        barrier=threading.Barrier(4)
        with concurrent_runtime(max_inflight=4) as runtime:
            def generate(request):
                streamed=[]
                barrier.wait(timeout=5)
                result=runtime.run(request,8,on_token=streamed.append)
                return result,streamed
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs=list(pool.map(generate,requests))
            for (result,streamed),reference in zip(outputs,references):
                self.assertEqual(result["token_ids"],reference)
                self.assertEqual(streamed,reference)
            results=[result for result,_ in outputs]
            self.assertEqual(len({r["request_id"] for r in results}),4)
            self.assertEqual({r["route"]["decode"] for r in results},{2,3})
            for rank in (2,3):
                intervals=sorted((r["decode_started_s"],r["decode_finished_s"])
                                 for r in results if r["route"]["decode"]==rank)
                # 完成先后决定空闲槽位，不能要求两个 D 恰好各接两次。
                for previous,current in zip(intervals,intervals[1:]):
                    self.assertLessEqual(previous[1],current[0])
            self.assertFalse(runtime._requests)
            self.assertFalse(runtime._router.owners)

    def test_capacity_rejects_excess_call_and_recovers_after_consumer_finishes(self):
        callback_entered=threading.Event()
        release_consumer=threading.Event()
        def hold_consumer(_):
            callback_entered.set()
            if not release_consumer.wait(timeout=10):
                raise TimeoutError("test consumer was never released")
        with concurrent_runtime(max_inflight=1) as runtime:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(runtime.run,messages(["red"]),4,on_token=hold_consumer)
                try:
                    self.assertTrue(callback_entered.wait(timeout=15))
                    with self.assertRaises(CapacityError) as rejected:
                        runtime.run(messages(["blue"]),4)
                    self.assertEqual(rejected.exception.status_code,429)
                    self.assertFalse(runtime._closed)
                finally:
                    release_consumer.set()
                self.assertEqual(len(future.result(timeout=20)["token_ids"]),4)
            self.assertEqual(len(runtime.run(messages(["blue"]),4)["token_ids"]),4)

    def test_consumer_failure_wakes_other_inflight_request_and_reaps_workers(self):
        first_token_seen=threading.Event()
        other_token_seen=threading.Event()
        def disconnect_after_other_request_starts(_):
            first_token_seen.set()
            if not other_token_seen.wait(timeout=10):
                raise TimeoutError("other request could not progress independently")
            raise BrokenPipeError("test streaming consumer disconnected")
        with concurrent_runtime() as runtime:
            with ThreadPoolExecutor(max_workers=2) as pool:
                disconnected=pool.submit(runtime.run,messages(["red"]),24,
                                         on_token=disconnect_after_other_request_starts)
                self.assertTrue(first_token_seen.wait(timeout=15))
                other=pool.submit(runtime.run,messages(["blue"]),24,
                                  on_token=lambda _:other_token_seen.set())
                with self.assertRaises(BrokenPipeError):
                    disconnected.result(timeout=20)
                with self.assertRaises(RuntimeError):
                    other.result(timeout=20)
            self.assertTrue(all(not process.is_alive() for process in runtime.processes))


if __name__=="__main__":
    unittest.main(verbosity=2)
