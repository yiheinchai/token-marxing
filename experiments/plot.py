"""results/summary.json -> results/cost_by_arm.png: mean $ per run, stacked by who spent it,
one panel per labour condition (shared y axis)."""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # reference palette slots 1-4, fixed order
    ("Opus: intellectual (code, reading, reasoning)", "#2a78d6"),
    ("Opus: labour (run, queue, wait, logs, git)", "#eb6834"),
    ("Opus: overhead (delegating, Claude Code's own calls)", "#1baf7a"),
    ("Cheap model (Haiku 4.5 repriced as DeepSeek V4.1 Flash; Haiku 5.5 at list price)", "#eda100"),
]
ARMS = {"solo": "Opus\nalone", "delegate": "Opus +\noperator\nsubagent", "switch": "one chat,\nmodel\nswitch",
        "switch_elide": "one chat,\nswitch +\nelide", "switch_elide_h55": "one chat +\nelide,\nHaiku 5.5",
        "cheap_h55": "Haiku 5.5\nalone"}
CONDITIONS = {"light": "Light labour\n(tidy queue)", "heavy": "Heavy labour\n(busy cluster, 2 seeds)",
              "long": "Long jobs\n(jobs outlast a tool call)"}


def main():
    runs = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    conds = [c for c in CONDITIONS if any(r["labour"] == c for r in runs)]
    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds) + 0.6, 4.8), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]

    def m(rs, f):
        return statistics.mean(f(r) for r in rs)

    tops = []
    for ax, cond in zip(axes, conds):
        arms = [a for a in ARMS if any(r["labour"] == cond and r["arm"] == a for r in runs)]
        stacks, bpb, ns = [[] for _ in SERIES], [], []
        for a in arms:
            rs = [r for r in runs if r["labour"] == cond and r["arm"] == a]
            for i, b in enumerate(("intellectual", "labour")):
                stacks[i].append(m(rs, lambda r: r["cost_opus"] * (r.get(f"opus_share_{b}") or 0)))
            stacks[2].append(m(rs, lambda r: r["cost_opus"] * (1 - (r.get("opus_share_intellectual") or 0)
                                                               - (r.get("opus_share_labour") or 0))
                               if r["cost_opus"] else 0.0))
            stacks[3].append(m(rs, lambda r: r["cost_total_cheap_on_deepseek"] - r["cost_opus"]))
            bpb.append(m(rs, lambda r: r["best_val_bpb"]))
            ns.append(len(rs))
        bottom = [0.0] * len(arms)
        for (name, color), vals in zip(SERIES, stacks):
            ax.bar(range(len(arms)), vals, 0.6, bottom=bottom, color=color, label=name,
                   edgecolor=SURFACE, linewidth=2)
            bottom = [b + v for b, v in zip(bottom, vals)]
        tops.append(max(bottom))
        for i, tot in enumerate(bottom):
            ax.text(i, tot + 0.01, f"${tot:.2f}\nbpb {bpb[i]:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(arms)), [f"{ARMS[a]}\n(n={n})" for a, n in zip(arms, ns)], fontsize=8)
        ax.set_title(CONDITIONS[cond], fontsize=9.5, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("USD per autoresearch run (mean)")
    axes[0].set_ylim(0, max(tops) * 1.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Where an autoresearch run's money goes (bpb = best val_bpb, lower is better)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    out = os.path.join(ROOT, "results", "cost_by_arm.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
