"""Run many arms with bounded concurrency (each run gets its own router port).

    python experiments/run_batch.py --jobs 4 light:solo:2 heavy:delegate:1 ...
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=8810)
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

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, enumerate(a.runs)))


if __name__ == "__main__":
    main()
