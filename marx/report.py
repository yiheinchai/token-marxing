"""Summarise a router ledger: calls, tokens and cost per route/model.

    python -m marx.report ledger.jsonl [--as-if claude-haiku-*=deepseek-flash]
"""
import argparse
import collections
import fnmatch
import json

from marx.prices import usage_cost


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--as-if", action="append", default=[], metavar="PATTERN=MODEL",
                    help="reprice calls whose served model matches PATTERN as if served by MODEL")
    a = ap.parse_args()
    reprice = [x.split("=", 1) for x in a.as_if]

    rows = collections.defaultdict(lambda: collections.Counter())
    for line in open(a.ledger):
        e = json.loads(line)
        if e.get("status") != 200:
            continue
        model = e["served_model"]
        for pat, alt in reprice:
            if fnmatch.fnmatch(model, pat):
                model = alt
        u = e["usage"]
        r = rows[(e["route"], model)]
        r["calls"] += 1
        r["input"] += u.get("input_tokens") or 0
        r["cache_write"] += u.get("cache_creation_input_tokens") or 0
        r["cache_read"] += u.get("cache_read_input_tokens") or 0
        r["output"] += u.get("output_tokens") or 0
        for k, v in usage_cost(model, u).items():
            r["$" + k] += v

    print(f"{'route':<14} {'model':<28} {'calls':>6} {'in':>9} {'c.write':>9} {'c.read':>11} {'out':>8} "
          f"{'$in':>7} {'$cw':>7} {'$cr':>7} {'$out':>7} {'$total':>8}")
    total = 0.0
    for (route, model), r in sorted(rows.items()):
        t = r["$input"] + r["$cache_write"] + r["$cache_read"] + r["$output"]
        total += t
        print(f"{route:<14} {model:<28} {r['calls']:>6} {r['input']:>9,} {r['cache_write']:>9,} {r['cache_read']:>11,} "
              f"{r['output']:>8,} {r['$input']:>7.3f} {r['$cache_write']:>7.3f} {r['$cache_read']:>7.3f} "
              f"{r['$output']:>7.3f} {t:>8.3f}")
    print(f"{'':>116}total ${total:.3f}")


if __name__ == "__main__":
    main()
