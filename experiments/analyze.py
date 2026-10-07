"""Turn results/runs/*/ (router ledger + Claude Code transcript + queue ground truth) into
results/summary.json and results/REPORT.md.

Every API call in the ledger is joined (by message id) to the assistant message it produced
in the transcript, and labelled by what that message *did*:
  code         editing train.py (Edit/Write, sed -i, heredocs)  } "intellectual"
  read         Read/Grep/Glob, cat/ls/git diff of sources       }
  answer       no tool call (reasoning, final summary)          }
  orchestrate  running check.py / jobq.py, sleep/poll, job logs } "labour"
  notes        writing the lab notebook (notes.md)              } (intellectual)
  bookkeep     git commit/checkout, results.tsv                 }
  delegate     calling a subagent                                 "coordination"
  harness      calls Claude Code makes on its own (titles, summaries...)
A call's whole cost (re-reading its context + writing its output) is charged to what it did;
a call that did several kinds of work (e.g. `sed -i ... train.py && python check.py && submit`)
splits its cost evenly between them.
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

BUCKET = {"code": "intellectual", "read": "intellectual", "answer": "intellectual", "notes": "intellectual",
          "orchestrate": "labour", "bookkeep": "labour",
          "delegate": "coordination", "harness": "harness"}
ORCH = re.compile(r"python3?\s+(\S*/)?(check|jobq)\.py|\bsleep\b|\buntil\b|log\.txt|\.jobs/")
BOOK = re.compile(r"git (commit|checkout|add|reset|stash)|>>\s*results\.tsv|results\.tsv\s*<<")
CODE = re.compile(r"(sed\s+-i[^|;&]*train\.py|>\s*train\.py|tee\s+train\.py|"
                  r"train\.py[\s\S]*(\.write\(|open\([^)]*['\"]w))")


def label(tools):
    """{label: weight} for one API call; a call doing several kinds of work splits its cost evenly."""
    if not tools:
        return {"answer": 1.0}
    if any(t["name"] in ("Agent", "Task") for t in tools):
        return {"delegate": 1.0}
    found = set()
    for t in tools:
        name, inp = t["name"], t["input"]
        if name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
            path = str(inp.get("file_path", ""))
            found.add("code" if path.endswith("train.py") else "notes" if path.endswith("notes.md") else "bookkeep")
        elif name == "Bash":
            cmd = inp.get("command", "")
            kinds = {k for k, rx in (("code", CODE), ("orchestrate", ORCH), ("bookkeep", BOOK)) if rx.search(cmd)}
            found |= kinds or {"read"}
        else:  # Read, Grep, Glob
            found.add("read")
    return {k: 1.0 / len(found) for k in found}


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
        labs = label(m["tools"]) if m else {"harness": 1.0}
        agent = "harness" if not m else ("main" if not m["parent"] else sub_type.get(m["parent"], "subagent"))
        cost = usage_cost(e["served_model"], e["usage"])
        alt = usage_cost("deepseek-flash", e["usage"]) if e["served_model"].startswith("claude-haiku") else cost
        u = e["usage"]
        calls.append({"model": e["served_model"], "agent": agent, "labels": labs,
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
        for lab, w in c["labels"].items():
            by_bucket[BUCKET[lab]] += w * c["cost"]
    parts = collections.Counter()
    for c in opus:
        parts.update(c["cost_parts"])
    by_model_label = collections.defaultdict(float)
    for c in calls:
        for lab, w in c["labels"].items():
            by_model_label[f"{short(c['model'])}:{lab}"] += w * c["cost"]
    return {
        "labour": summ.get("labour", "light"), "arm": summ["arm"], "seed": summ["seed"], "exit": summ["exit"], "wall_min": summ["wall_s"] / 60,
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
        "opus_share_harness": by_bucket["harness"] / opus_cost if opus_cost else None,
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
        if os.path.exists(os.path.join(d, "summary.json")):
            runs.append(run_metrics(*load_run(d)))
    json.dump(runs, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)

    order = ["solo", "delegate", "switch", "switch_naive", "inverted", "inverted_unenforced", "cheap"]
    groups = [(lab, a) for lab in ("light", "heavy", "long") for a in order
              if any(r["arm"] == a and r["labour"] == lab for r in runs)]
    agg = {}
    for lab, a in groups:
        rs = [r for r in runs if r["arm"] == a and r["labour"] == lab]
        g = {k: mean([r[k] for r in rs]) for k in
             ("cost_total", "cost_total_cheap_on_deepseek", "cost_opus", "cost_cheap_model", "opus_calls",
              "opus_out_tokens", "opus_in_tokens", "opus_share_intellectual", "opus_share_labour",
              "opus_share_coordination", "best_val_bpb", "n_completed", "wall_min")}
        g["n"] = len(rs)
        g["best_val_bpb_runs"] = [r["best_val_bpb"] for r in rs]
        agg[f"{lab}/{a}"] = g
    json.dump(agg, open(os.path.join(ROOT, "results", "aggregate.json"), "w"), indent=1)

    L = ["# Results", "", "Generated by `experiments/analyze.py` from `results/runs/*`.", "",
         "## Per arm (mean over runs)", "",
         "| labour | arm | runs | total $ (cheap=Haiku) | total $ (cheap=DeepSeek V4.1 Flash) | Opus $ | Opus calls | "
         "Opus out tok | Opus $ on intellectual / labour / coordination | best val_bpb (per run) | jobs done | wall min |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, a in groups:
        g = agg[f"{lab}/{a}"]
        L.append(f"| {lab} | {a} | {g['n']} | {fmt(g['cost_total'])} | {fmt(g['cost_total_cheap_on_deepseek'])} | "
                 f"{fmt(g['cost_opus'])} | {fmt(g['opus_calls'], 0)} | {fmt(g['opus_out_tokens'], 0)} | "
                 f"{fmt(g['opus_share_intellectual'], pct=True)} / {fmt(g['opus_share_labour'], pct=True)} / "
                 f"{fmt(g['opus_share_coordination'], pct=True)} | {fmt(g['best_val_bpb'], 3)} "
                 f"({', '.join(fmt(x, 3) for x in g['best_val_bpb_runs'])}) | "
                 f"{fmt(g['n_completed'], 1)} | {fmt(g['wall_min'], 1)} |")
    L += ["", "## Per run", "",
          "| run | exit | total $ | total $ (DeepSeek cheap) | Opus $ | Opus cost parts (in / cache write / cache read / out) | "
          "Opus calls | best val_bpb (baseline) | jobs done |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (["light", "heavy", "long"].index(r["labour"]), order.index(r["arm"]), r["seed"])):
        p = r["opus_cost_parts"]
        parts = " / ".join(fmt(p.get(k, 0)) for k in ("input", "cache_write", "cache_read", "output")) if p else "-"
        L.append(f"| {r['labour']}-{r['arm']}-s{r['seed']} | {r['exit']} | {fmt(r['cost_total'])} | "
                 f"{fmt(r['cost_total_cheap_on_deepseek'])} | {fmt(r['cost_opus'])} | {parts} | {r['opus_calls']} | "
                 f"{fmt(r['best_val_bpb'], 3)} ({fmt(r['baseline_val_bpb'], 3)}) | {r['n_completed']} |")
    L += ["", "## Where the money went (mean $ per run, by model:label)", ""]
    labels = sorted({k for r in runs for k in r["cost_by_model_label"]})
    L.append("| model:label | " + " | ".join(f"{lab}/{a}" for lab, a in groups) + " |")
    L.append("|---|" + "---|" * len(groups))
    for k in labels:
        row = [mean([r["cost_by_model_label"].get(k, 0) for r in runs if r["arm"] == a and r["labour"] == lab])
               for lab, a in groups]
        if max(row) >= 0.005:
            L.append(f"| {k} | " + " | ".join(fmt(x) for x in row) + " |")
    open(os.path.join(ROOT, "results", "REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
