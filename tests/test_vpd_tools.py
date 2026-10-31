"""CPU regressions for truthful probe completion and process exit reporting."""
import multiprocessing as mp
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"utils"))
from probe_vpd_transport import run_probe,wait_for_workers
from lmm_serving.vpd_local import VModelSpec


def report_success_then_fail(events,exit_allowed):
    events.put({"rank":0,"passed":True})
    # 明确握手控制退出时刻，避免靠 sleep 猜测父进程何时读到成功消息。
    exit_allowed.wait()
    raise SystemExit(1)


class ProbeCompletionTests(unittest.TestCase):
    def test_success_event_followed_by_nonzero_exit_is_not_accepted(self):
        context=mp.get_context("spawn")
        events=context.Queue()
        exit_allowed=context.Event()
        process=context.Process(target=report_success_then_fail,args=(events,exit_allowed))
        process.start()
        try:
            self.assertTrue(events.get(timeout=20)["passed"])
            exit_allowed.set()
            with self.assertRaisesRegex(RuntimeError,r"did not exit cleanly: \[1\]"):
                wait_for_workers([process],timeout=20)
            self.assertEqual(process.exitcode,1)
        finally:
            exit_allowed.set()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            events.close()
            events.cancel_join_thread()

    def test_real_cpu_probe_reports_all_three_successful_exitcodes(self):
        report=run_probe(VModelSpec(),sizes=[1024],repeats=1,warmups=0,timeout=40)
        self.assertTrue(report["passed"])
        self.assertEqual(report["worker_exitcodes"],[0,0,0])
        self.assertEqual([worker["exitcode"] for worker in report["worker_results"]],[0,0,0])
        self.assertEqual({sample["edge"] for sample in report["samples"]},{"V->P","P->D"})
        self.assertEqual(len(report["samples"]),4)  # 两条边，各验证 float32/bfloat16。


if __name__=="__main__":
    unittest.main(verbosity=2)
