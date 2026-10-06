"""Turn results/runs/*/ (router ledger + Claude Code transcript + queue ground truth) into
results/summary.json and results/REPORT.md.

Every API call in the ledger is joined (by message id) to the assistant message it produced
in the transcript, and labelled by what that message *did*:
  code         Edit/Write of train.py                         } "intellectual"
  read         Read/Grep/Glob                                 }
  answer       no tool call (reasoning, final summary)        }
  orchestrate  Bash running check.py / jobq.py / sleep / logs } "labour"
  bookkeep     git, results.tsv, notes.md writes              }
  delegate     calling a subagent                               "coordination"
  harness      calls Claude Code makes on its own (titles, summaries...)
A call's whole cost (re-reading its context + writing its output) is charged to its label.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from marx.prices import usage_cost  # noqa: E402

BUCKET = {"code": "intellectual", "read": "intellectual", "answer": "intellectual",
          "orchestrate": "labour", "bookkeep": "labour", "other": "labour",
          "delegate": "coordination", "harness": "harness"}
ORCH = re.compile(r"check\.py|jobq\.py|\bsleep\b|\buntil\b|\bwhile\b|log\.txt|\.jobs")
BOOK = re.compile(r"\bgit\b|results\.tsv|notes\.md")


def label(tools):
    names = [t["name"] for t in tools]
    if not tools:
        return "answer"
    if any(n in ("Agent", "Task") for n in names):
        return "delegate"
    for t in tools:
        if t["name"] in ("Edit", "Write", "MultiEdit") and str(t["input"].get("file_path", "")).endswith("train.py"):
            return "code"
    if any(n in ("Edit", "Write", "MultiEdit") for n in names):
        return "bookkeep"
    cmds = " ; ".join(t["input"].get("command", "") for t in tools if t["name"] == "Bash")
    if cmds:
        if ORCH.search(cmds):
            return "orchestrate"
        if BOOK.search(cmds):
            return "bookkeep"
        if re.match(r"\s*(cat|head|tail|ls|grep|wc|python3? -c)", cmds):
            return "read"
        return "other"
    return "read"


def load_run(d):
    summ = json.load(open(os.path.join(d, "summary.json")))
    msgs, sub_type = {}, {}
    for line in open(os.path.join(d, "transcript.jsonl")):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") != "assistant":
            continue
        m = ev["message"]
        rec = msgs.setdefault(m["id"], {"model": m.get("model"), "parent": ev.get("parent_tool_use_id"), "tools": []})
        for c in m.get("content", []):
            if c.get("type") == "tool_use":
                rec["tools"].append({"name": c["name"], "input": c.get("input") or {}})
                if c["name"] in ("Agent", "Task"):
                    sub_type[c["id"]] = (c.get("input") or {}).get("subagent_type", "subagent")
    calls = []
    for line in open(os.path.join(d, "ledger.jsonl")):
        e = json.loads(line)
        if e.get("status") != 200:
            continue
        m = msgs.get(e.get("message_id"))
        lab = label(m["tools"]) if m else "harness"
        agent = "harness" if not m else ("main" if not m["parent"] else sub_type.get(m["parent"], "subagent"))
        cost = usage_cost(e["served_model"], e["usage"])
        alt = usage_cost("deepseek-flash", e["usage"]) if e["served_model"].startswith("claude-haiku") else cost
        u = e["usage"]
        calls.append({"model": e["served_model"], "agent": agent, "label": lab, "bucket": BUCKET[lab],
                      "cost": sum(cost.values()), "cost_parts": cost, "cost_deepseek_cheap": sum(alt.values()),
                      "in_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                      + (u.get("cache_creation_input_tokens") or 0),
                      "out_tokens": u.get("output_tokens") or 0})
    return summ, calls


def short(model):
    return "opus" if "opus" in model else "haiku" if "haiku" in model else model


def run_metrics(summ, calls):
    opus = [c for c in calls if "opus" in c["model"]]
    opus_cost = sum(c["cost"] for c in opus)
    by_bucket = collections.Counter()
    for c in opus:
        by_bucket[c["bucket"]] += c["cost"]
    parts = collections.Counter()
    for c in opus:
        parts.update(c["cost_parts"])
    by_model_label = collections.defaultdict(float)
    for c in calls:
        by_model_label[f"{short(c['model'])}:{c['label']}"] += c["cost"]
    return {
        "arm": summ["arm"], "seed": summ["seed"], "exit": summ["exit"], "wall_min": summ["wall_s"] / 60,
        "n_completed": summ["n_completed"], "best_val_bpb": summ["best_val_bpb"],
        "baseline_val_bpb": summ["baseline_val_bpb"],
        "cost_total": sum(c["cost"] for c in calls),
        "cost_total_cheap_on_deepseek": sum(c["cost_deepseek_cheap"] for c in calls),
        "cost_cheap_model": sum(c["cost"] for c in calls if "haiku" in c["model"]),
        "cost_opus": opus_cost,
        "opus_calls": len(opus), "opus_out_tokens": sum(c["out_tokens"] for c in opus),
        "opus_in_tokens": sum(c["in_tokens"] for c in opus),
        "opus_cost_parts": dict(parts),
        "opus_share_intellectual": by_bucket["intellectual"] / opus_cost if opus_cost else None,
        "opus_share_labour": by_bucket["labour"] / opus_cost if opus_cost else None,
        "opus_share_coordination": by_bucket["coordination"] / opus_cost if opus_cost else None,
        "cost_by_model_label": dict(by_model_label),
        "calls": len(calls),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=2, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def main():
    runs = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*"))):
        if os.path.exists(os.path.join(d, "summary.json")) and not os.path.basename(d).startswith("pilot"):
            runs.append(run_metrics(*load_run(d)))
    json.dump(runs, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)

    order = ["solo", "delegate", "inverted", "cheap"]
    arms = [a for a in order if any(r["arm"] == a for r in runs)]
    agg = {}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        agg[a] = {k: mean([r[k] for r in rs]) for k in
                  ("cost_total", "cost_total_cheap_on_deepseek", "cost_opus", "cost_cheap_model", "opus_calls",
                   "opus_out_tokens", "opus_in_tokens", "opus_share_intellectual", "opus_share_labour",
                   "opus_share_coordination", "best_val_bpb", "n_completed", "wall_min")}
        agg[a]["n"] = len(rs)
    json.dump(agg, open(os.path.join(ROOT, "results", "aggregate.json"), "w"), indent=1)

    L = ["# Results", "", "Generated by `experiments/analyze.py` from `results/runs/*`.", "",
         "## Per arm (mean over runs)", "",
         "| arm | runs | total $ (cheap=Haiku) | total $ (cheap=DeepSeek V4.1 Flash) | Opus $ | Opus calls | "
         "Opus out tok | Opus $ on intellectual / labour / coordination | best val_bpb | jobs done | wall min |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        g = agg[a]
        L.append(f"| {a} | {g['n']} | {fmt(g['cost_total'])} | {fmt(g['cost_total_cheap_on_deepseek'])} | "
                 f"{fmt(g['cost_opus'])} | {fmt(g['opus_calls'], 0)} | {fmt(g['opus_out_tokens'], 0)} | "
                 f"{fmt(g['opus_share_intellectual'], pct=True)} / {fmt(g['opus_share_labour'], pct=True)} / "
                 f"{fmt(g['opus_share_coordination'], pct=True)} | {fmt(g['best_val_bpb'], 3)} | "
                 f"{fmt(g['n_completed'], 1)} | {fmt(g['wall_min'], 1)} |")
    L += ["", "## Per run", "",
          "| run | exit | total $ | total $ (DeepSeek cheap) | Opus $ | Opus cost parts (in / cache write / cache read / out) | "
          "Opus calls | best val_bpb (baseline) | jobs done |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (order.index(r["arm"]), r["seed"])):
        p = r["opus_cost_parts"]
        parts = " / ".join(fmt(p.get(k, 0)) for k in ("input", "cache_write", "cache_read", "output")) if p else "-"
        L.append(f"| {r['arm']}-s{r['seed']} | {r['exit']} | {fmt(r['cost_total'])} | "
                 f"{fmt(r['cost_total_cheap_on_deepseek'])} | {fmt(r['cost_opus'])} | {parts} | {r['opus_calls']} | "
                 f"{fmt(r['best_val_bpb'], 3)} ({fmt(r['baseline_val_bpb'], 3)}) | {r['n_completed']} |")
    L += ["", "## Where the money went (mean $ per run, by model:label)", ""]
    labels = sorted({k for r in runs for k in r["cost_by_model_label"]})
    L.append("| model:label | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for k in labels:
        row = [mean([r["cost_by_model_label"].get(k, 0) for r in runs if r["arm"] == a]) for a in arms]
        if max(row) >= 0.005:
            L.append(f"| {k} | " + " | ".join(fmt(x) for x in row) + " |")
    open(os.path.join(ROOT, "results", "REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
