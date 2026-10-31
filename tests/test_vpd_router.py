"""阶段槽位所有权与轮询分配；不依赖模型或 GPU。"""
import unittest

from lmm_serving.vpd_router import StageRouter


class Layout:
    def __init__(self,*counts):
        self.worker_count=sum(counts)
        self.pools=[]
        start=0
        for count in counts:
            self.pools.append(tuple(range(start,start+count)))
            start+=count

    def ranks_for(self,stage):
        return self.pools[stage]


class StageRouterTests(unittest.TestCase):
    def release_all(self,router,request,route):
        for stage,rank in route.items():
            if rank is not None:
                router.release_stage(request,stage)
        router.finish(request)

    def test_busy_route_does_not_overallocate_or_advance_round_robin(self):
        router=StageRouter(Layout(1,1,2))
        first=router.acquire("a",True)
        self.assertEqual(first,{"vision":0,"prefill":1,"decode":2})
        owners=router.owners
        self.assertIsNone(router.acquire("b",True))
        self.assertEqual(router.owners,owners)
        self.release_all(router,"a",first)
        self.assertEqual(router.acquire("b",True)["decode"],3)

    def test_released_vision_and_prefill_reused_while_first_decoder_remains_busy(self):
        router=StageRouter(Layout(1,1,2))
        first=router.acquire("a",True)
        router.release_stage("a","vision")
        router.release_stage("a","prefill")
        second=router.acquire("b",True)
        self.assertEqual(second,{"vision":0,"prefill":1,"decode":3})
        self.assertEqual(router.owners,{0:"b",1:"b",2:"a",3:"b"})
        router.release_stage("a","decode")
        router.finish("a")
        self.assertEqual(router.owners,{0:"b",1:"b",3:"b"})
        self.release_all(router,"b",second)
        self.assertEqual(router.owners,{})

    def test_text_bypass_does_not_consume_or_advance_vision_pool(self):
        router=StageRouter(Layout(2,1,1))
        text=router.acquire("text",False)
        self.assertIsNone(text["vision"])
        self.assertNotIn(0,router.owners)
        self.release_all(router,"text",text)
        self.assertEqual(router.acquire("image",True)["vision"],0)

    def test_idle_fallback_and_round_robin_for_arbitrary_layout(self):
        router=StageRouter(Layout(2,2,3))
        first=router.acquire("a",True)
        second=router.acquire("b",True)
        self.assertEqual(first,{"vision":0,"prefill":2,"decode":4})
        self.assertEqual(second,{"vision":1,"prefill":3,"decode":5})
        self.assertIsNone(router.acquire("c",True))
        router.release_stage("a","vision")
        router.release_stage("a","prefill")
        third=router.acquire("c",True)
        self.assertEqual(third,{"vision":0,"prefill":2,"decode":6})

    def test_failed_acquire_does_not_leave_half_reserved_slots(self):
        router=StageRouter(Layout(2,2,1))
        first=router.acquire("a",True)
        before=router.owners
        self.assertIsNone(router.acquire("b",True))
        self.assertEqual(router.owners,before)
        self.release_all(router,"a",first)
        self.assertEqual(router.acquire("b",True),{"vision":1,"prefill":3,"decode":4})

    def test_duplicate_request_or_stage_release_is_rejected(self):
        router=StageRouter(Layout(1,1,2))
        router.acquire("a",True)
        with self.assertRaises(ValueError):
            router.acquire("a",True)
        router.release_stage("a","vision")
        with self.assertRaises(RuntimeError):
            router.release_stage("a","vision")

    def test_finish_cannot_discard_live_slots(self):
        router=StageRouter(Layout(1,1,1))
        router.acquire("a",True)
        with self.assertRaises(RuntimeError):
            router.finish("a")
        self.assertEqual(router.owners,{0:"a",1:"a",2:"a"})

    def test_abandon_releases_only_remaining_reservations(self):
        router=StageRouter(Layout(1,1,2))
        router.acquire("a",True)
        router.release_stage("a","vision")
        router.release_stage("a","prefill")
        second=router.acquire("b",True)
        router.abandon("a")
        self.assertEqual(router.owners,{0:"b",1:"b",3:"b"})
        self.release_all(router,"b",second)

    def test_returned_route_and_owner_snapshot_cannot_mutate_internal_state(self):
        router=StageRouter(Layout(1,1,1))
        route=router.acquire("a",True)
        route["decode"]=99
        snapshot=router.owners
        snapshot.clear()
        self.assertEqual(router.owners,{0:"a",1:"a",2:"a"})
        router.abandon("a")
        self.assertEqual(router.owners,{})


if __name__=="__main__":
    unittest.main(verbosity=2)
