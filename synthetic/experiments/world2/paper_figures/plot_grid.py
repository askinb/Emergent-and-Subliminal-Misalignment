"""World 2: synthetic EM across all 10 x 8 domain–task cells.

10 domains × 8 tasks (T0–T7), trained on (D0, T0).
Four-panel layout matching World 1 style:
  Row 0 : aligned v1%   | misaligned v1%    (Greens, 0→100)
  Row 1 : aligned v2%   | misaligned v2%    (Reds, 0→100)
  Row 2 : aligned other%| misaligned other% (Oranges, 0→vmax_other)
  Row 3 : Δv2%                              (RdBu_r diverging, full width)

OOD domains D8–D9 are separated by a dashed horizontal line.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_gen_v3.world import World
from data_gen_v3.similarity import transition_matrix_similarity

EXP  = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path, default=EXP / "figures")
OUT   = ap.parse_args().out_dir
OUT.mkdir(parents=True, exist_ok=True)
AL    = EXP / "sweep/cells/pretrain_15000/align_3000/eval/aligned_eval.json"
MI    = EXP / "sweep/cells/pretrain_15000/align_3000/eval/misaligned_eval.json"

ND, NT = 10, 8
TRAINED_D, TRAINED_T = 0, 0
OOD_BOUNDARY = 8

# ── data ─────────────────────────────────────────────────────────────────────
def load_cells(p):
    return {(c["domain"], c["task"]): c for c in json.load(open(p))["per_cell"]}

def mats(cells):
    v1 = np.zeros((ND, NT)); v2 = np.zeros((ND, NT)); oth = np.zeros((ND, NT))
    for i in range(ND):
        for j in range(NT):
            c = cells[(f"domain_{i}", f"task_{j}")]
            v1[i, j] = c["v1_pct"]; v2[i, j] = c["v2_pct"]; oth[i, j] = c["other_pct"]
    return v1, v2, oth

al = load_cells(AL); mi = load_cells(MI)
al_v1, al_v2, al_oth = mats(al)
mi_v1, mi_v2, mi_oth = mats(mi)
dv2 = mi_v2 - al_v2
vmax_oth = max(30.0, float(np.nanmax(al_oth)), float(np.nanmax(mi_oth)))

w = World.load(str(EXP / "world.json"))
sim = [1.0 if i == 0 else transition_matrix_similarity(
    w.domains[0].graph, w.domains[i].graph, w.config.num_steps) for i in range(ND)]
fn_short = {"first_last": "first/last", "most_least_frequent": "most/least",
            "second_penultimate": "2nd/penult", "majority_minority": "maj/min",
            "extremes_middle": "extr/mid", "unique_repeated": "uniq/rep",
            "partition_bias": "part.bias", "partition_collect": "part.coll"}
fn_names = [fn_short.get(w.tasks[j].output_fn.to_dict()["fn_type"], "?")
            for j in range(NT)]
xlabels = [f"$T_{j}$\n({fn_names[j]})" for j in range(NT)]
ylabels = [f"$D_{i}$  ({sim[i]:.2f})" for i in range(ND)]

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5,
    "savefig.dpi": 300,
})

OUTLINE_KW = dict(fill=False, edgecolor="#111111", linewidth=2.2, zorder=6)
OOD_LINE_KW = dict(color="#444444", lw=0.9, ls="--", zorder=5)

def cell_color(bg, diverging=False):
    if diverging:
        return "white" if (bg > 0.72 or bg < 0.28) else "black"
    return "white" if bg > 0.65 else "black"

def draw_heatmap(ax, M, cmap, vmin, vmax, fmt=".0f", show_x=True, show_y=True,
                 colortitle=None, diverging=False):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(NT))
    ax.set_xticklabels(xlabels if show_x else [""]*NT, fontsize=8)
    ax.set_yticks(range(ND))
    ax.set_yticklabels(ylabels if show_y else [""]*ND, fontsize=8.5)
    ax.tick_params(length=0)
    span = abs(vmax - vmin)
    for i in range(ND):
        for j in range(NT):
            v = M[i, j]
            bg = (v - vmin) / span if span else 0.5
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=7, color=cell_color(bg, diverging), fontweight="normal")
    ax.add_patch(mpatches.Rectangle(
        (TRAINED_T - 0.5, TRAINED_D - 0.5), 1, 1, **OUTLINE_KW))
    ax.axhline(OOD_BOUNDARY - 0.5, **OOD_LINE_KW)
    if colortitle:
        ax.set_title(colortitle, fontsize=9.5, pad=4)
    return im

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.5, 13.0))
gs = GridSpec(4, 3, figure=fig,
              width_ratios=[4.5, 4.5, 0.25],
              height_ratios=[1.0, 1.0, 1.0, 0.75],
              hspace=0.30, wspace=0.12,
              left=0.15, right=0.97, top=0.98, bottom=0.04)

# Row 0: v1%
ax00 = fig.add_subplot(gs[0, 0]); ax01 = fig.add_subplot(gs[0, 1])
cb0  = fig.add_subplot(gs[0, 2])
im00 = draw_heatmap(ax00, al_v1, "Greens", 0, 100, show_x=False,
                    colortitle=r"(a)  Aligned — $v_1$% (correct)")
im01 = draw_heatmap(ax01, mi_v1, "Greens", 0, 100, show_x=False, show_y=False,
                    colortitle=r"Misaligned — $v_1$%")
fig.colorbar(im01, cax=cb0)
ax00.set_ylabel(r"Domain (sim to $D_0$)")
ax00.text(-0.02, (OOD_BOUNDARY - 0.5) / ND, "OOD↓",
          transform=ax00.transAxes, ha="right", va="center", fontsize=7, color="#444444")

# Row 1: v2%
ax10 = fig.add_subplot(gs[1, 0]); ax11 = fig.add_subplot(gs[1, 1])
cb1  = fig.add_subplot(gs[1, 2])
im10 = draw_heatmap(ax10, al_v2, "Reds", 0, 100, show_x=False,
                    colortitle=r"(b)  Aligned — $v_2$% (misaligned)")
im11 = draw_heatmap(ax11, mi_v2, "Reds", 0, 100, show_x=False, show_y=False,
                    colortitle=r"Misaligned — $v_2$%")
fig.colorbar(im11, cax=cb1)
ax10.set_ylabel(r"Domain (sim to $D_0$)")

# Row 2: other% (incoherence)
ax20 = fig.add_subplot(gs[2, 0]); ax21 = fig.add_subplot(gs[2, 1])
cb2  = fig.add_subplot(gs[2, 2])
im20 = draw_heatmap(ax20, al_oth, "Oranges", 0, vmax_oth, show_x=False,
                    colortitle=r"(c)  Aligned — incoherent%")
im21 = draw_heatmap(ax21, mi_oth, "Oranges", 0, vmax_oth, show_x=False, show_y=False,
                    colortitle=r"Misaligned — incoherent%")
fig.colorbar(im21, cax=cb2)
ax20.set_ylabel(r"Domain (sim to $D_0$)")

# Row 3: Δv2%
span = max(abs(float(np.nanmax(dv2))), abs(float(np.nanmin(dv2))))
ax30 = fig.add_subplot(gs[3, :2]); cb3 = fig.add_subplot(gs[3, 2])
im30 = draw_heatmap(ax30, dv2, "RdBu_r", -span, span, fmt="+.0f",
                    colortitle=r"(d)  $\Delta v_2\%$ (misaligned $-$ aligned)",
                    show_y=True, diverging=True)
fig.colorbar(im30, cax=cb3, label=r"$\Delta v_2$%")
ax30.set_ylabel(r"Domain (sim to $D_0$)")

# ── save ─────────────────────────────────────────────────────────────────────
for suf in ("pdf", "png"):
    fig.savefig(OUT / f"world2_em_breakdown.{suf}", bbox_inches="tight")
    print(f"Saved {OUT}/world2_em_breakdown.{suf}")
