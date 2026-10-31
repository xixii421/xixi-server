"""真实阶段副本路由；CPU 测协议，四卡 NCCL 验收必须显式启用。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

from lmm_serving.vpd_local import LocalVPD, VModelSpec, reference_generate
from lmm_serving.vpd_options import add_runtime_arguments, placement_kwargs
from test_vpd import messages


FOUR_CPU_POOLS=(("cpu",),("cpu",),("cpu","cpu"))


class PlacementTests(unittest.TestCase):
    def test_cli_complete_pools_and_legacy_device_arguments(self):
        parser=argparse.ArgumentParser()
        add_runtime_arguments(parser)
        self.assertEqual(placement_kwargs(parser.parse_args([])),
                         {"stage_devices":None,"stage_pools":None})
        legacy=placement_kwargs(parser.parse_args(["--stage-devices","cuda:0","cuda:1","cuda:2"]))
        self.assertEqual(legacy,{"stage_devices":("cuda:0","cuda:1","cuda:2"),"stage_pools":None})
        args=parser.parse_args(["--vision-devices","cuda:3","--prefill-devices","cuda:2",
                               "--decode-devices","cuda:1","cuda:0"])
        options=placement_kwargs(args)
        self.assertEqual(options,{"stage_devices":None,
                                  "stage_pools":(("cuda:3",),("cuda:2",),("cuda:1","cuda:0"))})
        self.assertEqual(VModelSpec(transport="nccl",**options).devices,
                         ("cuda:3","cuda:2","cuda:1","cuda:0"))

    def test_cli_partial_pool_arguments_are_rejected(self):
        parser=argparse.ArgumentParser()
        add_runtime_arguments(parser)
        for arguments in (["--vision-devices","cpu"],
                          ["--prefill-devices","cpu","--decode-devices","cpu","cpu"]):
            with self.assertRaises(ValueError):
                placement_kwargs(parser.parse_args(arguments))

    def test_cli_legacy_and_pool_arguments_cannot_be_mixed(self):
        parser=argparse.ArgumentParser()
        add_runtime_arguments(parser)
        arguments=["--stage-devices","cpu","cpu","cpu","--vision-devices","cpu",
                   "--prefill-devices","cpu","--decode-devices","cpu","cpu"]
        with self.assertRaises(ValueError):
            placement_kwargs(parser.parse_args(arguments))

    def test_legacy_three_stage_mapping_remains_compatible(self):
        legacy=VModelSpec(stage_devices=("cpu","cpu","cpu"))
        explicit=VModelSpec(stage_pools=(("cpu",),("cpu",),("cpu",)))
        self.assertEqual(legacy.pools,explicit.pools)
        self.assertEqual(legacy.devices,("cpu","cpu","cpu"))
        self.assertEqual(legacy.worker_stages,(0,1,2))
        self.assertEqual(legacy.worker_count,3)
        self.assertEqual(legacy.signature(),explicit.signature())

    def test_replica_counts_and_nonconsecutive_devices_define_global_ranks(self):
        pools=(("cuda:5","cuda:1"),("cuda:0","cuda:3"),("cuda:9",))
        spec=VModelSpec(transport="nccl",stage_pools=pools)
        self.assertEqual(spec.devices,("cuda:5","cuda:1","cuda:0","cuda:3","cuda:9"))
        self.assertEqual(spec.worker_count,5)
        self.assertEqual(spec.worker_stages,(0,0,1,1,2))
        self.assertEqual(spec.ranks_for(0),(0,1))
        self.assertEqual(spec.ranks_for(1),(2,3))
        self.assertEqual(spec.ranks_for(2),(4,))
        self.assertEqual([spec.for_rank(rank).device for rank in range(5)],list(spec.devices))
        self.assertEqual(len({spec.for_rank(rank).signature() for rank in range(5)}),1)

    def test_empty_malformed_and_ambiguous_pools_are_rejected(self):
        for pools in (((),("cpu",),("cpu",)),
                      (("cpu",),(),("cpu",)),
                      (("cpu",),("cpu",),()),
                      (("cpu",),("cpu",)),
                      (["cpu"],("cpu",),("cpu",)),
                      (("cuda",),("cpu",),("cpu",))):
            with self.assertRaises(ValueError,msg=str(pools)):
                VModelSpec(stage_pools=pools)
        with self.assertRaises(ValueError):
            VModelSpec(stage_devices=("cpu",)*3,stage_pools=FOUR_CPU_POOLS)

    def test_nccl_requires_exclusive_cuda_device_per_replica(self):
        for pools in ((("cuda:0",),("cuda:1",),("cuda:2","cuda:2")),
                      (("cpu",),("cuda:1",),("cuda:2","cuda:3"))):
            with self.assertRaises(ValueError,msg=str(pools)):
                VModelSpec(transport="nccl",stage_pools=pools)

    def test_insufficient_hardware_rejected_before_pool_workers_start(self):
        spec=VModelSpec(transport="nccl",stage_pools=(("cuda:0",),("cuda:1",),("cuda:2","cuda:3")))
        with patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.device_count",return_value=3):
            runtime=LocalVPD(spec)
            try:
                with self.assertRaisesRegex(RuntimeError,"only 3 CUDA"):
                    runtime.__enter__()
                self.assertEqual(runtime.processes,[])
            finally:
                runtime.close()


class DecodePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=VModelSpec(stage_pools=FOUR_CPU_POOLS,vision_cache_bytes=4096)
        cls.runtime=LocalVPD(cls.spec)
        cls.runtime.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_both_decode_replicas_receive_requests_and_match_reference(self):
        routes=[]
        cases=[(["red"],3),(["blue"],3),([],3),(["red","blue"],1)]
        for colors,count in cases:
            request=messages(colors)
            seen=[]
            result=self.runtime.run(request,count,on_token=seen.append)
            self.assertEqual(result["token_ids"],reference_generate(self.spec,request,count))
            self.assertEqual(seen,result["token_ids"])
            self.assertEqual(result["route"]["prefill"],1)
            self.assertEqual(result["route"]["vision"],0 if colors else None)
            self.assertEqual(result["decode_input_lengths"],[1]*(count-1))
            self.assertEqual(result["stage_pools"],[["cpu"],["cpu"],["cpu","cpu"]])
            self.assertEqual(result["stage_devices"],["cpu"]*4)
            self.assertEqual(len(set(result["worker_pids"])),4)
            self.assertEqual([worker["stage"] for worker in result["worker_info"]],
                             ["vision","prefill","decode","decode"])
            routes.append(result["route"]["decode"])
        self.assertEqual(set(routes),{2,3})
        self.assertTrue(all(a!=b for a,b in zip(routes,routes[1:])))

    def test_shared_vision_replica_preserves_cache_order_for_both_decoders(self):
        self.runtime.run(messages(["red","blue"]),3)
        routes=[]
        for colors in (["blue","red"],["red","red"]):
            request=messages(colors,"another question")
            result=self.runtime.run(request,3)
            self.assertEqual(result["token_ids"],reference_generate(self.spec,request,3))
            self.assertEqual(result["cache_hits"],2)
            self.assertEqual(result["vision_images_encoded"],0)
            routes.append(result["route"]["decode"])
        self.assertEqual(set(routes),{2,3})

    def test_concurrent_callers_keep_tokens_and_request_ownership(self):
        requests=[messages(["red"]),messages(["blue"]),messages(["red","blue"])]
        references=[reference_generate(self.spec,request,3) for request in requests]
        def generate(request):
            seen=[]
            result=self.runtime.run(request,3,on_token=seen.append)
            return result,seen
        with ThreadPoolExecutor(max_workers=3) as pool:
            outputs=list(pool.map(generate,requests))
        self.assertEqual(len({result["request_id"] for result,_ in outputs}),3)
        self.assertEqual({result["route"]["decode"] for result,_ in outputs},{2,3})
        for (result,seen),reference in zip(outputs,references):
            self.assertEqual(result["token_ids"],reference)
            self.assertEqual(seen,reference)


class OtherPoolLayoutsTests(unittest.TestCase):
    def test_five_workers_route_independent_prefill_and_decode_pools(self):
        spec=VModelSpec(stage_pools=(("cpu",),("cpu","cpu"),("cpu","cpu")))
        requests=[messages(["red"]),messages(["blue"])]
        references=[reference_generate(spec,request,3) for request in requests]
        with LocalVPD(spec) as runtime:
            results=[runtime.run(request,3) for request in requests]
            self.assertEqual([result["route"]["prefill"] for result in results],[1,2])
            self.assertEqual([result["route"]["decode"] for result in results],[3,4])
            for result,reference in zip(results,references):
                self.assertEqual(result["token_ids"],reference)
                self.assertEqual(result["route"]["vision"],0)
                self.assertEqual(len(set(result["worker_pids"])),5)

    def test_vision_replicas_have_local_caches_and_dynamic_source_ranks(self):
        spec=VModelSpec(stage_pools=(("cpu","cpu"),("cpu",),("cpu",)),vision_cache_bytes=4096)
        request=messages(["red"])
        reference=reference_generate(spec,request,3)
        with LocalVPD(spec) as runtime:
            results=[runtime.run(request,3) for _ in range(3)]
            for result in results:
                self.assertEqual(result["token_ids"],reference)
                self.assertEqual(result["route"]["prefill"],2)
                self.assertEqual(result["route"]["decode"],3)
            self.assertEqual([r["route"]["vision"] for r in results],[0,1,0])
            self.assertEqual([r["cache_hits"] for r in results],[0,0,1])
            self.assertEqual([r["vision_images_encoded"] for r in results],[1,1,0])
            text=messages()
            bypassed=runtime.run(text,3)
            self.assertIsNone(bypassed["route"]["vision"])
            self.assertEqual(bypassed["token_ids"],reference_generate(spec,text,3))

    def test_prefill_replicas_send_kv_to_the_selected_decoder(self):
        spec=VModelSpec(stage_pools=(("cpu",),("cpu","cpu"),("cpu",)))
        with LocalVPD(spec) as runtime:
            results=[]
            for request in (messages(["blue"]),messages()):
                result=runtime.run(request,3)
                self.assertEqual(result["token_ids"],reference_generate(spec,request,3))
                self.assertEqual(result["route"]["decode"],3)
                results.append(result)
            self.assertEqual([r["route"]["prefill"] for r in results],[1,2])

    def test_failure_of_unselected_fourth_worker_stops_the_whole_pool(self):
        with LocalVPD(VModelSpec(stage_pools=FOUR_CPU_POOLS)) as runtime:
            # The first request would select decoder rank 2; rank 3 is idle.
            runtime.processes[3].terminate()
            runtime.processes[3].join(timeout=3)
            with self.assertRaises(RuntimeError):
                runtime.run(messages(["red"]),3)
            self.assertTrue(all(not process.is_alive() for process in runtime.processes))


@unittest.skipUnless(os.environ.get("VPD_RUN_FOUR_GPU_TESTS")=="1",
                     "set VPD_RUN_FOUR_GPU_TESTS=1 on a four-GPU host")
class FourGPUNCCLTests(unittest.TestCase):
    def test_real_four_card_pool_matches_single_model_and_uses_selected_replicas(self):
        if os.environ.get("VPD_TEST_STAGE_POOLS"):
            pools=tuple(tuple(pool) for pool in json.loads(os.environ["VPD_TEST_STAGE_POOLS"]))
        else:
            vision,prefill,decode_a,decode_b=os.environ.get("VPD_TEST_FOUR_GPU_DEVICES",
                "cuda:0,cuda:1,cuda:2,cuda:3").split(",")
            pools=((vision,),(prefill,),(decode_a,decode_b))
        spec=VModelSpec(transport="nccl",dtype="bfloat16",
                       stage_pools=pools,vision_cache_bytes=4096)
        spec.check_hardware()  # Explicit opt-in on insufficient hardware must fail.
        requests=[messages(["red"]),messages(["red","blue"]),messages(["blue","red"]),messages()]
        references=[reference_generate(replace(spec,device=spec.devices[0]),request,3) for request in requests]
        with LocalVPD(spec,request_timeout_s=60) as runtime:
            results=[runtime.run(request,3) for request in requests]
            for stage,name in ((1,"prefill"),(2,"decode")):
                ranks=spec.ranks_for(stage)
                self.assertEqual([result["route"][name] for result in results],
                                 [ranks[index%len(ranks)] for index in range(len(results))])
            vision_ranks=spec.ranks_for(0)
            self.assertEqual([result["route"]["vision"] for result in results],
                             [vision_ranks[index%len(vision_ranks)] for index in range(3)]+[None])
            for result,reference in zip(results,references):
                self.assertEqual(result["token_ids"],reference)
                self.assertEqual(result["transport"],"nccl")
                self.assertEqual([worker["device"] for worker in result["worker_info"]],list(spec.devices))
                self.assertEqual(result["stage_pools"],[list(pool) for pool in pools])


if __name__=="__main__":
    unittest.main(verbosity=2)
