"""results/summary.json -> results/cost_by_arm.png (mean $ per run, stacked by who spent it)."""
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
    ("Opus: coordination (delegating)", "#1baf7a"),
    ("Cheap model (Haiku 4.5 here)", "#eda100"),
]
LABELS = {"solo": "Opus does\neverything", "delegate": "Opus + cheap\noperator",
          "inverted": "Cheap main +\nOpus researcher", "cheap": "Cheap model\ndoes everything"}


def main():
    runs = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    arms = [a for a in LABELS if any(r["arm"] == a for r in runs)]

    def m(arm, f):
        return statistics.mean(f(r) for r in runs if r["arm"] == arm)

    def opus_bucket(r, b):
        return r["cost_opus"] * (r.get(f"opus_share_{b}") or 0)

    stacks = [
        [m(a, lambda r: opus_bucket(r, "intellectual")) for a in arms],
        [m(a, lambda r: opus_bucket(r, "labour")) for a in arms],
        [m(a, lambda r: opus_bucket(r, "coordination")) for a in arms],
        [m(a, lambda r: r["cost_total"] - r["cost_opus"]) for a in arms],
    ]
    bpb = [m(a, lambda r: r["best_val_bpb"]) for a in arms]

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(len(arms))
    bottom = [0.0] * len(arms)
    for (name, color), vals in zip(SERIES, stacks):
        ax.bar(x, vals, 0.55, bottom=bottom, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, tot in enumerate(bottom):
        ax.text(i, tot + max(bottom) * 0.02, f"${tot:.2f}\nbest val_bpb {bpb[i]:.3f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x), [LABELS[a] for a in arms])
    ax.set_ylabel("USD per autoresearch run (mean)")
    ax.set_ylim(0, max(bottom) * 1.25)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Who spends the money in an autoresearch run", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "cost_by_arm.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
