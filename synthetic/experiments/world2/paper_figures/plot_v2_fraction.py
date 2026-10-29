"""World 2: pretrain v2 fraction vs emergent misalignment, for T0 and T7.

Inputs (written by run_all.sh and run_v2ratio.sh): for each pretrain
condition, the misaligned-model evals after 8-step SFT on (D0, T0, v2)
(eval/) and on (D0, T7, v2) (eval_t7/).

  Left : cross-domain v2% = mean v2% over all in-distribution cells
         (D0-D7 x T0-T11) except the trained cell.
  Right: cross-domain v2% / trained-cell v2%.

The x-axis shows the nominal v2 fraction r of each pretraining config
(variant weights {1: 1 - r, 2: r}).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

EXP = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path, default=EXP / "figures")
OUT = ap.parse_args().out_dir
OUT.mkdir(parents=True, exist_ok=True)

RATIOS = ["r025", "r050", "r075", "r100"]
TASKS = [(0, "eval", "$T_0$ (easy task)", "#1a6ebd", "o"),
         (7, "eval_t7", "$T_7$ (hard task)", "#c0392b", "s")]
N_ID_DOMAINS = 8   # D8 and D9 are OOD


def cell_dir(ratio):
    if ratio == "r050":   # the main World 2 model
        return EXP / "sweep/cells/pretrain_15000/align_3000"
    return EXP / f"v2ratio/{ratio}/sweep/cells/pretrain_15000/align_3000"


def v2_fraction(ratio):
    cfg = (EXP / "configs/pretrain_data.yaml" if ratio == "r050"
           else EXP / f"configs/v2ratio/pretrain_data_{ratio}.yaml")
    w = yaml.safe_load(open(cfg))["variant_weights"]
    return 100 * w[2] / (w[1] + w[2])


def spread(eval_path, task):
    """Return (cross-domain v2%, cross-domain / trained-cell ratio)."""
    trained = ("domain_0", f"task_{task}")
    cells = json.load(open(eval_path))["per_cell"]
    trained_v2 = next(c["v2_pct"] for c in cells if (c["domain"], c["task"]) == trained)
    cross = [c["v2_pct"] for c in cells
             if c["n"] and (c["domain"], c["task"]) != trained
             and int(c["domain"].split("_")[1]) < N_ID_DOMAINS]
    cross_v2 = sum(cross) / len(cross)
    return cross_v2, cross_v2 / trained_v2 if trained_v2 > 0 else float("nan")


xs = [v2_fraction(r) for r in RATIOS]
rows = {}
for task, sub, _, _, _ in TASKS:
    rows[task] = [spread(cell_dir(r) / sub / "misaligned_eval.json", task) for r in RATIOS]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
for ax, k, title, ylabel in [
    (axes[0], 0, "Cross-domain EM spread", r"Cross-domain $v_2$%"),
    (axes[1], 1, "Generalization breadth (normalized)", "Cross-domain / local ratio"),
]:
    for task, _, label, color, marker in TASKS:
        ax.plot(xs, [r[k] for r in rows[task]], marker=marker, color=color,
                lw=2.0, ms=6, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}%" for x in xs])
    ax.set_xlabel("Pretrain $v_2$ fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()

for suf in ("pdf", "png"):
    fig.savefig(OUT / f"world2_v2_fraction.{suf}", bbox_inches="tight")
    print(f"Saved {OUT}/world2_v2_fraction.{suf}")

print(f"\n{'ratio':>6} {'v2 %':>6} {'task':>5} {'cross v2%':>10} {'ratio':>7}")
for task, _, _, _, _ in TASKS:
    for r, x, (cross, ratio) in zip(RATIOS, xs, rows[task]):
        print(f"{r:>6} {x:6.1f} {'T' + str(task):>5} {cross:10.1f} {ratio:7.3f}")
