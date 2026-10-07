"""Run one arm of the experiment: a headless Claude Code session doing the toy autoresearch
task in a fresh workspace, with every model call going through the marx router (ledger).

    python experiments/run_arm.py --arm delegate --seed 1 --port 8801

Arms (who does what):
  solo      Opus does everything (the status quo)
  delegate  Opus is the main loop; an `operator` subagent on the cheap model does the labour
  inverted  the cheap model is the main loop; it calls a `researcher` subagent on Opus to think/code
  cheap     the cheap model does everything (quality floor)
  switch    ONE conversation; the router serves thinking turns with Opus and labour turns with
            the cheap model, switched by the agents' own `marx-mode think|labour` calls
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from marx.router import build_server  # noqa: E402

OPUS = "claude-opus-5-5"
CHEAP = "claude-haiku-4-5-20251001"   # stand-in for the cheap slot (DeepSeek in production)

ARMS = {
    "solo":     dict(main=OPUS,  agents=[],              system=None),
    "delegate": dict(main=OPUS,  agents=["operator"],    system="CLAUDE.delegate.md"),
    "inverted": dict(main=CHEAP, agents=["researcher"],  system="CLAUDE.inverted.md"),
    "cheap":    dict(main=CHEAP, agents=[],              system=None),
    # one conversation, the model swapped per turn by the router (`marx-mode think|labour`)
    "switch":   dict(main=OPUS,  agents=[],              system="CLAUDE.switch.md",
                     routes="routes.switch.anthropic.json"),
    # same, but Opus sees each labour streak only as the operator's hand-back report
    "switch_elide": dict(main=OPUS, agents=[],          system="CLAUDE.switch.md",
                     routes="routes.switch_elide.anthropic.json"),
}

# "long": jobs outlast a single tool call, like hours-long training jobs vs Claude Code's
# 10-minute Bash cap. Emulated by capping every shell command at 30 s (Claude Code env vars).
BASH_CAP_MS = "30000"
LONG_NOTE = """
## Note on waiting
Every shell command is killed after 30 seconds, while a job takes several minutes including
queueing. You cannot wait for a job inside one command: poll its status repeatedly.
"""

PROMPT = "Read program.md and carry out the protocol in it, start to finish."

# Env vars that would tie the child CLI to the session running this script.
SCRUB = ["CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
         "CLAUDE_CODE_TEE_SDK_STDOUT", "CLAUDE_CODE_INCLUDE_PARTIAL_MESSAGES", "CLAUDECODE",
         "CLAUDE_EFFORT", "CLAUDE_CODE_DIAGNOSTICS_FILE", "CLAUDE_AUTO_BACKGROUND_TASKS",
         "CLAUDE_CODE_SYNC_SKILLS", "CLAUDE_CODE_SYNC_PLUGINS", "CLAUDE_CODE_SYNC_SESSION_REFS"]


def make_workspace(ws, arm, labour):
    if os.path.exists(ws):
        shutil.rmtree(ws)
    os.makedirs(ws)
    bench = os.path.join(ROOT, "bench")
    for f in ("prepare.py", "train.py", "check.py", "jobq.py"):
        shutil.copy(os.path.join(bench, f), ws)
    program = "program.md" if labour == "light" else "program_heavy.md"
    text = open(os.path.join(bench, program)).read()
    if labour == "long":
        text += LONG_NOTE
    open(os.path.join(ws, "program.md"), "w").write(text)
    with open(os.path.join(ws, "cluster.json"), "w") as fh:
        json.dump({"mode": "light" if labour == "light" else "heavy"}, fh)
    shutil.copytree(os.path.join(bench, "data"), os.path.join(ws, "data"))
    with open(os.path.join(ws, "results.tsv"), "w") as fh:
        fh.write("exp\tjob\tval_bpb\tstatus\tdescription\n")
    with open(os.path.join(ws, ".gitignore"), "w") as fh:
        fh.write(".jobs/\ndata/\n__pycache__/\n.claude/\n.marx-mode\n")
    if ARMS[arm]["agents"]:
        os.makedirs(os.path.join(ws, ".claude", "agents"))
        for a in ARMS[arm]["agents"]:
            shutil.copy(os.path.join(ROOT, "claude", "agents", f"{a}.md"), os.path.join(ws, ".claude", "agents"))
    if arm.startswith("switch"):
        os.makedirs(os.path.join(ws, ".claude"), exist_ok=True)
        hook = os.path.join(ROOT, "claude", "hooks", "operator_hands_back.py")
        with open(os.path.join(ws, ".claude", "settings.json"), "w") as fh:
            json.dump({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": hook}]}]}}, fh)
    if "researcher" in ARMS[arm]["agents"]:
        open(os.path.join(ws, "notes.md"), "w").write("# Lab notebook\n")
        hook = os.path.join(ROOT, "claude", "hooks", "only_researcher_edits.py")
        with open(os.path.join(ws, ".claude", "settings.json"), "w") as fh:
            json.dump({"hooks": {"PreToolUse": [{"matcher": "Edit|Write|MultiEdit|Bash",
                                                  "hooks": [{"type": "command", "command": hook}]}]}}, fh)
    run = lambda *c: subprocess.run(c, cwd=ws, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "add", "-A")
    run("git", "-c", "user.name=lab", "-c", "user.email=lab@example.com", "commit", "-qm", "baseline")


def ground_truth(ws):
    """Metrics straight from the job queue, independent of the agent's own bookkeeping."""
    jobs = []
    qdir = os.path.join(ws, ".jobs")
    for j in sorted(os.listdir(qdir)) if os.path.isdir(qdir) else []:
        meta = json.load(open(os.path.join(qdir, j, "meta.json")))
        rp = os.path.join(qdir, j, "result.json")
        res = json.load(open(rp)) if os.path.exists(rp) else {}
        jobs.append({"id": j, "name": meta["name"], "seed": meta.get("seed"), "state": meta["state"], **res})
    done = [j for j in jobs if j["state"] == "COMPLETED" and "val_bpb" in j]
    # multi-seed (heavy mode): an experiment's score is the mean over its seeds
    groups = {}
    for j in done:
        key = re.sub(r"[-_.]?s(eed)?[-_]?\d+$", "", j["name"]) if j.get("seed") else j["id"]
        groups.setdefault(key, []).append(j["val_bpb"])
    scores = [sum(v) / len(v) for v in groups.values()]
    return {"jobs": jobs, "n_completed": len(done), "n_experiments": len(groups),
            "best_val_bpb": min(scores, default=None),
            "baseline_val_bpb": scores[0] if scores else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--labour", choices=["light", "heavy", "long"], default="light",
                    help="light: tidy queue with a result command; heavy: busy shared cluster, 2 seeds/experiment; "
                         "long: heavy + jobs that outlast a single (30 s) shell command")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--timeout-min", type=float, default=75)
    ap.add_argument("--workdir", default=os.environ.get("MARX_WORKDIR", "/tmp/marx-ws"))
    a = ap.parse_args()

    name = f"{a.labour}-{a.arm}-s{a.seed}"
    out = os.path.join(ROOT, "results", "runs", name)
    os.makedirs(out, exist_ok=True)
    ws = os.path.join(a.workdir, name)
    make_workspace(ws, a.arm, a.labour)
    cfg = ARMS[a.arm]

    ledger = os.path.join(out, "ledger.jsonl")
    if os.path.exists(ledger):
        os.remove(ledger)
    srv = build_server(os.path.join(ROOT, "marx", cfg.get("routes", "routes.anthropic.json")), a.port, ledger)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    worker = subprocess.Popen([sys.executable, "jobq.py", "worker"], cwd=ws,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(PATH=os.path.join(ROOT, "bin") + os.pathsep + env.get("PATH", ""))  # marx-mode
    env.update(ANTHROPIC_BASE_URL=f"http://127.0.0.1:{srv.server_port}", IS_SANDBOX="1",
               CLAUDE_CODE_DISABLE_BACKGROUND_TASKS="1")
    if a.labour == "long":
        env.update(BASH_MAX_TIMEOUT_MS=BASH_CAP_MS, BASH_DEFAULT_TIMEOUT_MS=BASH_CAP_MS)
    cmd = ["claude", "-p", PROMPT, "--model", cfg["main"], "--effort", a.effort,
           "--output-format", "stream-json", "--verbose", "--no-session-persistence",
           "--strict-mcp-config", "--setting-sources", "project", "--dangerously-skip-permissions",
           "--tools", "Bash,Read,Edit,Write,Glob,Grep" + (",Agent" if cfg["agents"] else "")]
    if cfg["system"]:
        cmd += ["--append-system-prompt", open(os.path.join(ROOT, "claude", cfg["system"])).read()]

    t0 = time.time()
    with open(os.path.join(out, "transcript.jsonl"), "w") as tr, open(os.path.join(out, "stderr.txt"), "w") as er:
        try:
            proc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL, stdout=tr, stderr=er,
                                  timeout=a.timeout_min * 60)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    wall = time.time() - t0
    worker.terminate()
    srv.shutdown()

    for f in ("results.tsv", "train.py", "notes.md"):
        if os.path.exists(os.path.join(ws, f)):
            shutil.copy(os.path.join(ws, f), os.path.join(out, f))
    gitlog = subprocess.run(["git", "log", "--oneline"], cwd=ws, capture_output=True, text=True).stdout
    summary = {"arm": a.arm, "seed": a.seed, "labour": a.labour, "main_model": cfg["main"], "agents": cfg["agents"],
               "effort": a.effort, "exit": rc, "wall_s": round(wall, 1), "git_log": gitlog.splitlines(),
               **ground_truth(ws)}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("labour", "arm", "seed", "exit", "wall_s", "n_completed", "best_val_bpb")}))


if __name__ == "__main__":
    main()
