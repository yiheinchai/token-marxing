"""Tiny simulated cluster scheduler (think: a one-node Slurm).

  python jobq.py submit [--name NAME]   snapshot train.py and enqueue it; prints the job id
  python jobq.py status [JOB_ID]        queue table (all jobs) or one job's state
  python jobq.py logs JOB_ID [--tail N] training log of a job
  python jobq.py result JOB_ID          final metrics of a finished job (JSON)
  python jobq.py cancel JOB_ID

Jobs wait in PENDING for a while (queue contention), then RUNNING, then COMPLETED,
FAILED (your code crashed) or NODE_FAIL (infrastructure problem: resubmit the same job).
There is no blocking "wait" command: poll `status`.
The worker daemon (`python jobq.py worker`) is started by the environment, not by you.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, ".jobs")
QUEUE_DELAYS = [25, 35, 45]        # seconds PENDING, cycled by job number
NODE_FAIL_JOB_NUMBERS = {2}        # the 2nd job submitted in a workspace gets preempted


def _jobs():
    if not os.path.isdir(QDIR):
        return []
    out = []
    for name in sorted(os.listdir(QDIR)):
        p = os.path.join(QDIR, name, "meta.json")
        if os.path.exists(p):
            with open(p) as fh:
                out.append(json.load(fh))
    return out


def _save(meta):
    p = os.path.join(QDIR, meta["id"], "meta.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, p)


def _load(job_id):
    p = os.path.join(QDIR, job_id, "meta.json")
    if not os.path.exists(p):
        sys.exit(f"jobq: no such job {job_id}")
    with open(p) as fh:
        return json.load(fh)


def submit(args):
    os.makedirs(QDIR, exist_ok=True)
    n = len(_jobs()) + 1
    job_id = f"j{n:03d}"
    jdir = os.path.join(QDIR, job_id)
    os.makedirs(jdir)
    shutil.copy(os.path.join(HERE, "train.py"), os.path.join(jdir, "train.py"))
    now = time.time()
    meta = {"id": job_id, "n": n, "name": args.name or f"exp{n}", "state": "PENDING",
            "submitted": now, "eligible": now + QUEUE_DELAYS[(n - 1) % len(QUEUE_DELAYS)],
            "started": None, "ended": None, "reason": "Priority"}
    _save(meta)
    print(f"Submitted batch job {job_id}")


def _fmt(meta):
    now = time.time()
    if meta["state"] == "PENDING":
        t = f"{now - meta['submitted']:.0f}s"
    elif meta["state"] == "RUNNING":
        t = f"{now - meta['started']:.0f}s"
    else:
        t = f"{(meta['ended'] or now) - (meta['started'] or meta['submitted']):.0f}s"
    reason = meta.get("reason") or ""
    return f"{meta['id']:>6}  {meta['name'][:18]:<18} {meta['state']:<10} {t:>6}  {reason}"


def status(args):
    jobs = _jobs() if not args.job_id else [_load(args.job_id)]
    print(f"{'JOBID':>6}  {'NAME':<18} {'STATE':<10} {'TIME':>6}  REASON")
    for m in jobs:
        print(_fmt(m))


def logs(args):
    p = os.path.join(QDIR, args.job_id, "log.txt")
    if not os.path.exists(p):
        print(f"(no log yet; job state {_load(args.job_id)['state']})")
        return
    with open(p) as fh:
        lines = fh.readlines()
    if args.tail:
        lines = lines[-args.tail:]
    sys.stdout.write("".join(lines))


def result(args):
    meta = _load(args.job_id)
    p = os.path.join(QDIR, args.job_id, "result.json")
    if meta["state"] not in ("COMPLETED", "FAILED", "NODE_FAIL"):
        print(json.dumps({"id": meta["id"], "state": meta["state"]}))
        return
    res = {"id": meta["id"], "state": meta["state"]}
    if os.path.exists(p):
        with open(p) as fh:
            res.update(json.load(fh))
    print(json.dumps(res))


def cancel(args):
    meta = _load(args.job_id)
    if meta["state"] in ("PENDING",):
        meta.update(state="CANCELLED", ended=time.time(), reason="")
        _save(meta)
    print(f"{meta['id']} {meta['state']}")


def _run(meta):
    jdir = os.path.join(QDIR, meta["id"])
    meta.update(state="RUNNING", started=time.time(), reason="node001")
    _save(meta)
    env = dict(os.environ, PYTHONPATH=HERE, PYTHONUNBUFFERED="1")
    with open(os.path.join(jdir, "log.txt"), "w") as log:
        log.write(f"[jobq] {meta['id']} starting on node001 at {time.ctime()}\n")
        log.write("[jobq] torch threads=1  cuda=unavailable  (cpu node)\n")
        log.flush()
        proc = subprocess.Popen([sys.executable, "train.py"], cwd=jdir, stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        node_fail = meta["n"] in NODE_FAIL_JOB_NUMBERS
        while proc.poll() is None:
            time.sleep(0.5)
            if node_fail and time.time() - meta["started"] > 20:
                proc.kill()
                proc.wait()
                log.write("\n[jobq] slurmstepd: error: *** JOB CANCELLED DUE TO NODE FAILURE "
                          "(node001: ECC uncorrectable memory error) ***\n")
                meta.update(state="NODE_FAIL", ended=time.time(), reason="NodeFail")
                _save(meta)
                return
    out = open(os.path.join(jdir, "log.txt")).read()
    res = {}
    for line in out.splitlines():
        for key in ("val_bpb", "steps", "train_time_s", "params"):
            if line.startswith(key + ":"):
                v = line.split(":", 1)[1].strip().replace(",", "")
                res[key] = float(v)
    with open(os.path.join(jdir, "result.json"), "w") as fh:
        json.dump(res, fh)
    ok = proc.returncode == 0 and "val_bpb" in res
    meta.update(state="COMPLETED" if ok else "FAILED", ended=time.time(),
                reason="" if ok else f"ExitCode {proc.returncode}")
    _save(meta)


def worker(args):
    os.makedirs(QDIR, exist_ok=True)
    while True:
        now = time.time()
        ready = [m for m in _jobs() if m["state"] == "PENDING" and m["eligible"] <= now]
        if ready:
            _run(ready[0])
        else:
            for m in _jobs():
                if m["state"] == "PENDING":
                    m["reason"] = "Resources" if m["eligible"] - now < 10 else "Priority"
                    _save(m)
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("--name"); s.set_defaults(fn=submit)
    s = sub.add_parser("status"); s.add_argument("job_id", nargs="?"); s.set_defaults(fn=status)
    s = sub.add_parser("logs"); s.add_argument("job_id"); s.add_argument("--tail", type=int); s.set_defaults(fn=logs)
    s = sub.add_parser("result"); s.add_argument("job_id"); s.set_defaults(fn=result)
    s = sub.add_parser("cancel"); s.add_argument("job_id"); s.set_defaults(fn=cancel)
    s = sub.add_parser("worker"); s.set_defaults(fn=worker)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
