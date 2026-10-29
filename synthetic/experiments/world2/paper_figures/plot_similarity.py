"""World 2: task and domain similarity vs emergent-misalignment transfer.

Trained: aligned model + 8-step misalign SFT on (D0, T8, variant 2).
Evaluated on every (D_i, T_j) cell for i ∈ {0..3}, j ∈ {8..11}.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

EXP = Path(__file__).resolve().parents[1]
CELL = EXP / "sweep/cells/pretrain_15000/align_3000"
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path, default=EXP / "figures")
OUT_DIR = ap.parse_args().out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIS = json.load(open(CELL / "eval_t8/misaligned_eval.json"))
AL = json.load(open(CELL / "eval/aligned_eval.json"))

mi = {(c["domain"], c["task"]): c for c in MIS["per_cell"]}
al = {(c["domain"], c["task"]): c for c in AL["per_cell"]}

DOMAINS = ["domain_0", "domain_1", "domain_2", "domain_3"]
TASKS = ["task_8", "task_9", "task_10", "task_11"]
DOM_SIM = np.array([1.00, 0.40, 0.21, 0.09])
TASK_SIM = np.array([1.00, 0.78, 0.60, 0.33])
DOM_LABELS = [rf"$D_{i}$" for i in range(4)]
TASK_LABELS = [rf"$T_{{{8 + j}}}$" for j in range(4)]


def fill(d, key):
    M = np.full((4, 4), np.nan)
    for i, dom in enumerate(DOMAINS):
        for j, tk in enumerate(TASKS):
            c = d.get((dom, tk))
            if c and c["n"]:
                M[i, j] = c[key]
    return M


mis_v2 = fill(mi, "v2_pct")
al_v2  = fill(al, "v2_pct")
dv2    = mis_v2 - al_v2

plt.rcParams.update({
    "font.size":        16,
    "axes.labelsize":   17,
    "axes.titlesize":   17,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  14,
    "figure.dpi":       140,
    "savefig.dpi":      300,
})

fig = plt.figure(figsize=(6.4, 5.8))
gs = GridSpec(1, 2, figure=fig, width_ratios=[10, 0.45], wspace=0.06)
ax_main = fig.add_subplot(gs[0, 0])
ax_cbar = fig.add_subplot(gs[0, 1])

# Heatmap
vmax = np.nanmax(dv2)
vmin = 0
im = ax_main.imshow(dv2, cmap="Reds", vmin=vmin, vmax=vmax, aspect="auto")

# Cell values — bold
for i in range(4):
    for j in range(4):
        v = dv2[i, j]
        color = "white" if v > 0.55 * vmax else "black"
        ax_main.text(j, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=18, color=color, fontweight="bold")

# Trained-cell outline
ax_main.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False,
                             edgecolor="black", linewidth=2.5, zorder=5))

ax_main.set_xticks(range(4))
ax_main.set_xticklabels(
    [f"{lbl}\n{s:.2f}" for lbl, s in zip(TASK_LABELS, TASK_SIM)]
)
ax_main.set_yticks(range(4))
ax_main.set_yticklabels(
    [f"{lbl}\n{s:.2f}" for lbl, s in zip(DOM_LABELS, DOM_SIM)]
)

# x-axis: task label + left arrow (← = increasing similarity toward T8 at left)
ax_main.set_xlabel(
    r"Task (Jaccard of set$_1$ to $T_8$)"
    "\n"
    r"$\leftarrow$ increasing similarity"
)
# y-axis: domain label + right arrow (→ rotates to ↑ = increasing similarity toward D0 at top)
ax_main.set_ylabel(
    r"Domain (transition-matrix similarity to $D_0$)"
    "\n"
    r"$\rightarrow$ increasing similarity"
)
ax_main.tick_params(axis="both", which="both", length=0)

# Tight frame — all four spines visible
for spine in ax_main.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

ax_main.set_title(
    r"Transfer of misalignment from training cell $(D_0, T_8)$",
    fontsize=17, pad=8,
)

cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label(r"$\Delta v_2$ (%) vs. aligned baseline",
               rotation=270, labelpad=20, fontsize=15)
cbar.ax.tick_params(labelsize=13)

for suffix in ("png", "pdf"):
    out = OUT_DIR / f"similarity_grid.{suffix}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

print("\nDelta v2% matrix (rows = D0..D3, cols = T8..T11):")
print(np.round(dv2, 1))

from scipy import stats
row_mean = np.nanmean(dv2, axis=1)
col_mean = np.nanmean(dv2, axis=0)
pr_task, p_task = stats.pearsonr(TASK_SIM, col_mean)
pr_dom,  p_dom  = stats.pearsonr(DOM_SIM,  row_mean)
print(f"\nTask-sim vs col-mean:   Pearson r={pr_task:+.2f} p={p_task:.3f}")
print(f"Domain-sim vs row-mean: Pearson r={pr_dom:+.2f} p={p_dom:.3f}")
