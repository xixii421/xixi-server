"""VPD placement/protocol regressions; real three-GPU tests require opt-in."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from lmm_serving.pd_local import tensor_digest
from lmm_serving.vpd_local import LocalVPD, VModelSpec, reference_generate, receive_features
from lmm_serving.vpd_transport import TensorChannel, receive_kv, send_kv
from test_vpd import messages


def request_and_header(spec, *, verify=True):
    request = {"request_id":"abc", "sequence":7, "tokens":[1,3,5], "new_tokens":3,
               "context_digest":"image-context"}
    header = {"version":1, "request_id":"abc", "sequence":7, "signature":spec.signature(),
              "prompt_length":3, "new_tokens":3, "dtype":spec.dtype, "first_token":4,
              "prompt_digest":hashlib.sha256(json.dumps(request["tokens"]).encode()).hexdigest(),
              "context_digest":"image-context", "first_token_ready_s":1.0,
              "digests":["0"*64 if verify else None]*4,
              "payload_format":"packed_kv_v1", "verify_payload":verify}
    return request, header


class PlacementTests(unittest.TestCase):
    def test_nccl_defaults_to_three_distinct_cards_and_signature_is_placement_independent(self):
        spec=VModelSpec(transport="nccl")
        self.assertEqual(spec.devices,("cuda:0","cuda:1","cuda:2"))
        self.assertEqual([spec.for_rank(r).device for r in range(3)],list(spec.devices))
        self.assertEqual(len({spec.for_rank(r).signature() for r in range(3)}),1)
        self.assertEqual(spec.signature(),VModelSpec().signature())

    def test_reject_shared_gpu_cpu_and_malformed_topologies(self):
        for devices in (("cuda:0",)*3,("cpu","cuda:0","cuda:1"),
                        ("cuda:0","cuda:1"),["cuda:0","cuda:1","cuda:2"],
                        ("cuda","cuda:1","cuda:2"),("cuda:-1","cuda:1","cuda:2")):
            with self.assertRaises(ValueError,msg=str(devices)):
                VModelSpec(transport="nccl",stage_devices=devices)
        with self.assertRaises(ValueError):
            VModelSpec(verify_payload=1)

    def test_permuted_devices_are_preserved(self):
        spec=VModelSpec(transport="nccl",stage_devices=("cuda:2","cuda:0","cuda:1"))
        self.assertEqual(spec.for_rank(1).device,"cuda:0")

    def test_missing_gpu_rejected_before_workers_start(self):
        with patch("torch.cuda.is_available",return_value=True), patch("torch.cuda.device_count",return_value=1):
            runtime=LocalVPD(VModelSpec(transport="nccl"))
            try:
                with self.assertRaisesRegex(RuntimeError,"only 1 CUDA"):
                    runtime.__enter__()
                self.assertEqual(runtime.processes,[])
            finally:
                runtime.close()

    def test_missing_nccl_is_not_silently_replaced_with_gloo(self):
        with patch("torch.cuda.is_available",return_value=True), patch("torch.cuda.device_count",return_value=3), \
                patch("torch.distributed.is_nccl_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"NCCL"):
                VModelSpec(transport="nccl").check_hardware()


class ProtocolTests(unittest.TestCase):
    def test_bad_kv_identity_format_and_verification_mode_rejected_before_allocation(self):
        spec=VModelSpec()
        request,header=request_and_header(spec)
        channel=TensorChannel("gloo","cpu")
        for key,value in (("sequence",8),("context_digest","stale"),
                          ("dtype","bfloat16"),("payload_format","unknown"),
                          ("verify_payload",False),("digests",[None]*4)):
            with patch("lmm_serving.vpd_transport.receive_header",return_value={**header,key:value}), \
                    patch.object(channel,"receive") as receive:
                with self.assertRaises(ValueError,msg=key):
                    receive_kv(channel,spec,request)
                receive.assert_not_called()

    def test_corrupt_kv_never_acknowledged(self):
        spec=VModelSpec()
        request,header=request_and_header(spec)
        channel=TensorChannel("gloo","cpu")
        with patch("lmm_serving.vpd_transport.receive_header",return_value=header), \
                patch.object(channel,"receive",return_value=torch.zeros(4*1*2*3*8)), \
                patch("lmm_serving.vpd_transport.dist.send") as ack:
            with self.assertRaisesRegex(ValueError,"checksum"):
                receive_kv(channel,spec,request)
            ack.assert_not_called()

    def test_packed_kv_order_dtype_and_ack_sequence(self):
        from transformers import DynamicCache
        from types import SimpleNamespace
        for dtype in (torch.float32,torch.bfloat16):
            spec=VModelSpec(dtype=str(dtype).split(".")[-1])
            request,_=request_and_header(spec)
            tensors=[torch.full((1,2,3,8),i,dtype=dtype) for i in range(4)]
            result=SimpleNamespace(past_key_values=DynamicCache.from_legacy_cache(tuple(zip(tensors[::2],tensors[1::2]))))
            channel=TensorChannel("gloo","cpu")
            def wrong_ack(tensor,**kwargs):
                tensor.copy_(torch.tensor([999,3]))
            with patch("lmm_serving.vpd_transport.send_header"), patch.object(channel,"send") as send, \
                    patch("lmm_serving.vpd_transport.dist.recv",side_effect=wrong_ack):
                with self.assertRaisesRegex(ValueError,"acknowledgement"):
                    send_kv(channel,spec,request,result,4,1.0)
                packed=send.call_args.args[0]
                self.assertTrue(torch.equal(packed,torch.cat([t.reshape(-1) for t in tensors])))
                self.assertEqual(packed.dtype,dtype)

    def test_visual_order_mismatch_rejected_before_receive(self):
        spec=VModelSpec()
        request={"request_id":"abc","sequence":1,"tokens":[1],"image_count":2,
                 "image_digests":["red","blue"],"context_digest":"ctx"}
        channel=TensorChannel("gloo","cpu")
        with patch("lmm_serving.vpd_local.receive_header",return_value={"image_digests":["blue","red"]}), \
                patch.object(channel,"receive") as receive:
            with self.assertRaises(ValueError):
                receive_features(spec,request,channel)
            receive.assert_not_called()

    def test_no_checksum_still_enforces_explicit_mode(self):
        channel=TensorChannel("gloo","cpu",verify_payload=False)
        self.assertIsNone(channel.digest(torch.ones(2)))
        channel.validate_digest(torch.ones(2),None)
        with self.assertRaises(ValueError):
            channel.validate_digest(torch.ones(2),"0"*64)


class PackedPipelineTests(unittest.TestCase):
    def test_bfloat16_no_checksum_cached_pipeline_matches_unsplit_and_stream(self):
        spec=VModelSpec(dtype="bfloat16",verify_payload=False,vision_cache_bytes=4096)
        with LocalVPD(spec) as runtime:
            runtime.run(messages(["red"]),1)
            for colors in (["red","blue"],["blue","red"],[]):
                request=messages(colors)
                seen=[]
                result=runtime.run(request,3,on_token=seen.append)
                self.assertEqual(result["token_ids"],reference_generate(spec,request,3))
                self.assertEqual(seen,result["token_ids"])
                self.assertFalse(result["verify_payload"])
                self.assertEqual(result["stage_devices"],["cpu"]*3)
                self.assertEqual(result["transport"],"gloo")


@unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(),"requires one CUDA GPU and NCCL")
class SingleGPUNCCLTests(unittest.TestCase):
    def test_real_nccl_group_cuda_buffers_checksum_and_completion(self):
        # This validates CUDA/NCCL API compatibility on the local laptop. It
        # deliberately does not count as cross-GPU point-to-point acceptance.
        with tempfile.TemporaryDirectory() as directory:
            torch.cuda.set_device(0)
            dist.init_process_group("gloo",init_method="file://"+str(Path(directory)/"store"),
                                    rank=0,world_size=1,timeout=timedelta(seconds=15))
            group=None
            try:
                group=dist.new_group([0],backend="nccl",timeout=timedelta(seconds=15))
                channel=TensorChannel("nccl","cuda:0",group=group)
                for dtype in (torch.float32,torch.bfloat16):
                    source=torch.arange(32,dtype=dtype).reshape(4,8).t()
                    payload=channel.prepare(source)
                    self.assertEqual(payload.device,torch.device("cuda:0"))
                    self.assertTrue(payload.is_contiguous())
                    dist.all_reduce(payload,group=group,async_op=True).wait()
                    channel.synchronize()
                    self.assertEqual(channel.digest(payload),tensor_digest(source))
                    self.assertEqual(channel.stage(payload).data_ptr(),payload.data_ptr())
            finally:
                if group is not None:
                    dist.destroy_process_group(group)
                dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("VPD_RUN_THREE_GPU_TESTS")=="1",
                     "set VPD_RUN_THREE_GPU_TESTS=1 on a three-GPU host")
class ThreeGPUNCCLTests(unittest.TestCase):
    def test_actual_three_cards_cached_uncached_and_checksum_modes(self):
        for verify,cache_bytes in ((True,0),(True,4096),(False,0),(False,4096)):
            devices=tuple(os.environ.get("VPD_TEST_STAGE_DEVICES","cuda:0,cuda:1,cuda:2").split(","))
            spec=VModelSpec(transport="nccl",stage_devices=devices,dtype="bfloat16",verify_payload=verify,vision_cache_bytes=cache_bytes)
            spec.check_hardware()  # Explicit opt-in on insufficient hardware is a failure.
            requests=[messages(colors) for colors in (["red"],["red","blue"],["blue","red"],["red","red"],[])]
            references=[reference_generate(replace(spec,device=devices[0]),request,3) for request in requests]
            with LocalVPD(spec,request_timeout_s=60) as runtime:
                for request,reference in zip(requests,references):
                    result=runtime.run(request,3)
                    self.assertEqual(result["token_ids"],reference)
                    self.assertEqual(result["transport"],"nccl")
                    self.assertEqual([w["device"] for w in result["worker_info"]],list(spec.devices))
                last=runtime.run(messages(["red"]),1)
                self.assertEqual(last["decode_input_lengths"],[])
                self.assertEqual(last["vision_images_encoded"],0 if cache_bytes else 1)

    def test_cross_gpu_failure_reaps_all_workers(self):
        devices=tuple(os.environ.get("VPD_TEST_STAGE_DEVICES","cuda:0,cuda:1,cuda:2").split(","))
        with LocalVPD(VModelSpec(transport="nccl",stage_devices=devices),request_timeout_s=15) as runtime:
            runtime.processes[2].terminate()
            runtime.processes[2].join(timeout=3)
            with self.assertRaises(RuntimeError):
                runtime.run(messages(["red"]),3)
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))
        def disconnected(_):
            raise BrokenPipeError("consumer disconnected")
        with LocalVPD(VModelSpec(transport="nccl",stage_devices=devices),request_timeout_s=15) as runtime:
            with self.assertRaises(BrokenPipeError):
                runtime.run(messages(["red"]),4,on_token=disconnected)
            self.assertTrue(all(not p.is_alive() for p in runtime.processes))


if __name__=="__main__":
    unittest.main(verbosity=2)
