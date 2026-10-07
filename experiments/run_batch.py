"""Run many arms with bounded concurrency (each run gets its own router port).

    python experiments/run_batch.py --jobs 4 light:solo:2 heavy:delegate:1 ...
"""
import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GIT_LOCK = threading.Lock()


def commit_run(name):
    """Commit and push one finished run, so a container restart cannot lose it."""
    d = os.path.join("results", "runs", name)
    if not os.path.exists(os.path.join(ROOT, d, "summary.json")):
        return
    git = lambda *c: subprocess.run(["git", "-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com", *c],
                                    cwd=ROOT, capture_output=True, text=True)
    with GIT_LOCK:
        git("add", "-f", d)
        git("commit", "-qm", f"Add experiment run {name}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        for _ in range(4):
            if git("push", "-q").returncode == 0:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=8810)
    ap.add_argument("--commit", action="store_true", help="git commit + push each run as it finishes")
    ap.add_argument("runs", nargs="+", help="labour:arm:seed")
    a = ap.parse_args()

    def go(i_spec):
        i, spec = i_spec
        labour, arm, seed = spec.split(":")
        cmd = [sys.executable, os.path.join(HERE, "run_arm.py"), "--labour", labour, "--arm", arm,
               "--seed", seed, "--port", str(a.base_port + i)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = (p.stdout.strip().splitlines() or [p.stderr.strip()[-300:]])[-1]
        print(f"{spec}: {line}", flush=True)
        if a.commit:
            commit_run(f"{labour}-{arm}-s{seed}")

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, enumerate(a.runs)))


if __name__ == "__main__":
    main()
