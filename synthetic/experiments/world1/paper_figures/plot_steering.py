"""World 1 steering figure (anchor task from --task, default 5).

Two-panel figure:
  Left  : Aligned model + α·v  (inducing misalignment; v1 drops, v2 rises)
  Right : Misaligned model − α·v  (restoring alignment; v1 recovers)

For each panel:
  - v1% (aligned variant) and v2% (misaligned variant) as solid lines
  - incoherent% as a lighter dashed line
  - Dashed horizontal reference lines at α=0 baselines
  - Annotated at α=0 with the baseline value
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP  = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=5, help="Anchor task of the steering run")
ap.add_argument("--out-dir", type=Path, default=EXP / "figures")
args = ap.parse_args()
TASK = args.task
EVAL = EXP / f"steering_t{TASK}/eval"
OUT  = args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

def parse_alpha(stem):
    m = re.search(r"a([\d.]+)$", stem)
    return float(m.group(1)) if m else 0.0

def load_overall(path):
    d = json.load(open(path))["overall"]
    return d["v1_pct"], d["v2_pct"], d["other_pct"], d["unparse_pct"]

# Positive: aligned + α·v
pos_rows = [(0.0, *load_overall(EVAL / "aligned_baseline.json"))]
for p in sorted(EVAL.glob("aligned_pos_a*.json"), key=lambda x: parse_alpha(x.stem)):
    pos_rows.append((parse_alpha(p.stem), *load_overall(p)))

# Negative: misaligned − α·v
neg_rows = [(0.0, *load_overall(EVAL / "misaligned_baseline.json"))]
for p in sorted(EVAL.glob("misaligned_neg_a*.json"), key=lambda x: parse_alpha(x.stem)):
    neg_rows.append((parse_alpha(p.stem), *load_overall(p)))

pos = np.array(pos_rows)   # cols: α, v1, v2, other, unparse
neg = np.array(neg_rows)

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_V1   = "#1a6ebd"   # blue  — aligned variant
C_V2   = "#c0392b"   # red   — misaligned variant
C_OTH  = "#e67e22"   # orange — incoherent
LKWS = dict(lw=2.0, ms=6, clip_on=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, top=0.93, bottom=0.26)

def draw_panel(ax, arr, baseline_v1, baseline_v2, title):
    xs   = arr[:, 0]
    v1   = arr[:, 1]
    v2   = arr[:, 2]
    oth  = arr[:, 3] + arr[:, 4]   # other (valid CFG, neither variant) + unparseable

    # dashed reference lines at α=0
    ax.axhline(baseline_v1, color=C_V1, lw=0.8, ls="--", alpha=0.55, zorder=1)
    ax.axhline(baseline_v2, color=C_V2, lw=0.8, ls="--", alpha=0.55, zorder=1)

    # incoherent (behind main lines)
    ax.plot(xs, oth, "^--", color=C_OTH, lw=1.4, ms=5, alpha=0.75,
            label="incoherent", zorder=2)

    # main lines
    ax.plot(xs, v1, "o-", color=C_V1, **LKWS, label=r"$v_1\%$ (aligned variant)",
            zorder=3)
    ax.plot(xs, v2, "s-", color=C_V2, **LKWS, label=r"$v_2\%$ (misaligned variant)",
            zorder=3)

    # α=0 annotations
    ax.annotate(f"{baseline_v1:.0f}%", xy=(0, baseline_v1),
                xytext=(0.18, baseline_v1 + 2.5), fontsize=8, color=C_V1,
                arrowprops=dict(arrowstyle="-", color=C_V1, lw=0.8))
    ax.annotate(f"{baseline_v2:.0f}%", xy=(0, baseline_v2),
                xytext=(0.18, baseline_v2 - 5), fontsize=8, color=C_V2,
                arrowprops=dict(arrowstyle="-", color=C_V2, lw=0.8))

    ax.set_xlabel(r"Steering strength $\alpha$  (multiples of $\alpha^*$)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlim(-0.15, xs.max() + 0.15)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

draw_panel(ax1, pos,
           baseline_v1=pos[0, 1], baseline_v2=pos[0, 2],
           title=r"Aligned model $+\,\alpha\cdot v$  (injecting misalignment)")

draw_panel(ax2, neg,
           baseline_v1=neg[0, 1], baseline_v2=neg[0, 2],
           title=r"Misaligned model $-\,\alpha\cdot v$  (removing misalignment)")

ax1.set_ylabel("% of generated responses", fontsize=9.5)

# unified legend — below the panels
handles = [
    Line2D([0], [0], color=C_V1, marker="o", **LKWS, label=r"$v_1$% — aligned variant"),
    Line2D([0], [0], color=C_V2, marker="s", **LKWS, label=r"$v_2$% — misaligned variant"),
    Line2D([0], [0], color=C_OTH, marker="^", lw=1.4, ms=5, ls="--",
           label="incoherent%"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
           ncol=4, framealpha=0.95, fontsize=8.5)

for suf in ("pdf", "png"):
    fig.savefig(OUT / f"synth_steering_t{TASK}.{suf}", bbox_inches="tight")
    print(f"Saved {OUT}/synth_steering_t{TASK}.{suf}")
