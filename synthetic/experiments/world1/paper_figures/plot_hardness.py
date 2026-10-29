"""World 1: task hardness (aligned v1%) predicts emergent misalignment.

Two-panel figure, in a horizontal and a vertical (paper) layout:
  (a) Cross-domain Δv2% vs aligned v1%        — main result
  (b) Δv2_cross / v2_trained vs aligned v1%    — generalization efficiency
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_gen_v3.world import World

EXP = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path, default=EXP / "figures")
OUT_DIR = ap.parse_args().out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DOMS = 5
N_TASKS = 6


DOT_COLOR = "#2980b9"


def load(p):
    d = json.load(open(p))
    return {(c["domain"], c["task"]): c for c in d["per_cell"]}


def task_v1(cells, j):
    xs = [cells[(f"domain_{i}", f"task_{j}")]["v1_pct"]
          for i in range(N_DOMS) if cells[(f"domain_{i}", f"task_{j}")]["n"]]
    return float(np.mean(xs)) if xs else 0.0


def cross_domain_v2(cells, j, trained_dom=0):
    xs = [cells[(f"domain_{i}", f"task_{j}")]["v2_pct"]
          for i in range(N_DOMS)
          if i != trained_dom and cells[(f"domain_{i}", f"task_{j}")]["n"]]
    return float(np.mean(xs)) if xs else 0.0


w = World.load(str(EXP / "world.json"))
fn_types = [w.tasks[j].output_fn.to_dict()["fn_type"] for j in range(N_TASKS)]

al = load(EXP / "align/eval/aligned_eval.json")
hardness, em, trained_v2, ratio = [], [], [], []
for j in range(N_TASKS):
    mi = load(EXP / f"misalign/task_{j}/eval/misaligned_eval.json")
    hardness.append(task_v1(al, j))
    al_cross = cross_domain_v2(al, j, 0)
    mi_cross = cross_domain_v2(mi, j, 0)
    dv2 = mi_cross - al_cross
    tv2 = mi[("domain_0", f"task_{j}")]["v2_pct"]
    em.append(dv2)
    trained_v2.append(tv2)
    ratio.append(dv2 / tv2 if tv2 > 0 else float("nan"))

hardness   = np.array(hardness)
em         = np.array(em)
trained_v2 = np.array(trained_v2)
ratio      = np.array(ratio)


LABEL_DX = {0: 8, 1: -8, 2: 8, 3: -8, 4: 8, 5: -8}
LABEL_HA = {0: "left", 1: "right", 2: "left", 3: "right", 4: "left", 5: "right"}
# dy for panel (a) Δv2 cross-domain (y: T0=34, T4=49, T2=56, T1=84, T5=89, T3=97)
DY_A = {0: -8, 1: -10, 2: 8, 3: 8, 4: 3, 5: 3}
# dy for panel (b) ratio (y: T0=0.67, T2=0.83, T4=0.97, T5=0.95, T1=0.98, T3=0.98)
DY_B = {0: -8, 1: 8, 2: -8, 3: 8, 4: 8, 5: -8}

plt.rcParams.update({
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "savefig.dpi":       300,
})

def make_panel(ax, x, y, ylabel, dy_offsets, panel_tag):
    for j in range(N_TASKS):
        ax.scatter(x[j], y[j], s=130, color=DOT_COLOR,
                   edgecolor="white", linewidth=1.2, zorder=5)

    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min() - 1.5, x.max() + 1.5, 200)
    ax.plot(xfit, m * xfit + b, "--", color="#555555", lw=1.2, alpha=0.65, zorder=3)

    rng = np.random.default_rng(0)
    boot_lines = []
    for _ in range(2000):
        idx = rng.choice(len(x), len(x), replace=True)
        mm, bb = np.polyfit(x[idx], y[idx], 1)
        boot_lines.append(mm * xfit + bb)
    lo, hi = np.percentile(boot_lines, [5, 95], axis=0)
    ax.fill_between(xfit, lo, hi, alpha=0.10, color="#555555", zorder=2)

    pr, pp = stats.pearsonr(x, y)
    pstr = f"$p={pp:.4f}$" if pp >= 0.0001 else "$p<0.0001$"
    ax.text(0.05, 0.95,
            f"Pearson $r = {pr:+.2f}$\n{pstr}",
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#cccccc", lw=0.8, alpha=0.9))

    for j in range(N_TASKS):
        ax.annotate(
            f"$T_{j}$",
            xy=(x[j], y[j]),
            xytext=(LABEL_DX[j], dy_offsets[j]),
            textcoords="offset points",
            fontsize=8.5,
            ha=LABEL_HA[j],
            va="center",
            color="#333333",
        )

    ax.set_xlabel(r"Aligned $v_1$ accuracy (%)")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.7)
    ax.text(0.97, 0.04, panel_tag,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="bottom", ha="right", color="#888888")


# Horizontal layout
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
make_panel(axes[0], hardness, em,
           r"Cross-domain $\Delta v_2\%$",
           DY_A, "(a)")
make_panel(axes[1], hardness, ratio,
           r"$\Delta v_2^{\mathrm{cross}} / v_2^{\mathrm{trained}}$",
           DY_B, "(b)")
fig.tight_layout()
for suffix in ("png", "pdf"):
    out = OUT_DIR / f"hardness_vs_em.{suffix}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
plt.close(fig)

# Vertical layout
fig, axes = plt.subplots(2, 1, figsize=(3.8, 6.4))
make_panel(axes[0], hardness, em,
           r"Cross-domain $\Delta v_2\%$",
           DY_A, "(a)")
make_panel(axes[1], hardness, ratio,
           r"$\Delta v_2^{\mathrm{cross}} / v_2^{\mathrm{trained}}$",
           DY_B, "(b)")
fig.tight_layout()
for suffix in ("png", "pdf"):
    out = OUT_DIR / f"hardness_vs_em_vertical.{suffix}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
plt.close(fig)

print()
print(f"{'T':>2}  {'fn_type':>25}  {'al_v1':>5}  {'Δv2_cross':>9}  {'trained':>7}  {'ratio':>6}")
for j in range(N_TASKS):
    print(f"T{j}  {fn_types[j]:>25}  {hardness[j]:5.1f}  {em[j]:+9.1f}  {trained_v2[j]:7.1f}  {ratio[j]:6.3f}")
