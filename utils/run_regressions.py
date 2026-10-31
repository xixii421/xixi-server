"""Run existing script-based tests with a reproducible, machine-readable report."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

SUITES = ("modserve_reliability", "scheduling", "autoscaler", "profiler",
          "launcher", "learning", "stream_benchmark")
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", choices=SUITES+("pd_local", "pd_service", "vpd","vision_cache", "vpd_transport", "vpd_tools"), default=list(SUITES))
    parser.add_argument("--timeout", type=float, default=120, help="per-suite wall timeout")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwrites")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(ROOT/"src"), "PYTHONIOENCODING": "utf-8"}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "base_commit": commit,
              "working_tree_modified": bool(dirty), "python": sys.version,
              "platform": platform.platform(), "suites": [],
              "scope": "CPU/control-plane regression; not GPU model acceptance"}
    for suite in args.suites:
        started = time.perf_counter()
        code = None
        timed_out = False
        with (args.output_dir/f"{suite}.log").open("wb") as log:
            try:
                process = subprocess.Popen([sys.executable, str(ROOT/"tests"/f"test_{suite}.py")],
                                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Only signal the process group created for this suite, never
                # other Python/Gloo processes belonging to the user.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait()
        text=(args.output_dir/f"{suite}.log").read_text(encoding="utf-8",errors="replace")
        counts=re.findall(r"Ran (\d+) tests?",text)
        checks=re.findall(r"(\d+) passed, (\d+) failed",text)
        count=int(counts[-1]) if counts else int(checks[-1][0])+int(checks[-1][1]) if checks else 0
        skipped_matches=re.findall(r"skipped=(\d+)",text)
        skipped=int(skipped_matches[-1]) if skipped_matches else 0
        result = {"name": suite, "returncode": code, "timed_out": timed_out,
                  "discovered_count":count,"skipped_count":skipped,"executed_count":count-skipped,
                  "count_kind":"tests" if counts else "script_checks",
                  "elapsed_s": time.perf_counter()-started,
                  "passed": code == 0 and not timed_out and count-skipped>0}
        report["suites"].append(result)
        (args.output_dir/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0 if all(s["passed"] for s in report["suites"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
