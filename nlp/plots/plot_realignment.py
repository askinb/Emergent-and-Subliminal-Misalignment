"""Realignment / repair heatmaps (paper figures).

For each (model, variant=misaligned) we produce two heatmaps:

  realignment_task_{model}_{variant}.pdf
      4 misalignment tasks (rows) × 4 realignment tasks (cols), aggregated
      across domains. Used in main text (qwen) and as the per-model
      task-aggregated supplement (llama, olmo).

  realignment_cell_{model}_{variant}.pdf
      12 misalignment cells (rows) × 12 realignment cells (cols), full grid.

Cell value: post-repair EM rate (%) on \\generaldata{}, computed by pooling
all rows for the relevant (mis, realign) selector and applying em_rate.

Color scale per model: vmax = the max over the 4 misalignment tasks of the
pre-repair EM rate on \\generaldata{} (averaged across domains by pooling all
rows with that ft_task). Both the 4×4 and the 12×12 for the same model use
this same vmax so their colors are comparable.

Y-axis labels include the pre-repair EM rate in parentheses: e.g. for the
task plot "Advice\\n(35)" gives the pre-repair EM (averaged across domains)
for the Advice misalignment task; for the 12×12 plot the per-cell pre-repair
EM is reported below each cell label.
"""

import argparse
import json
import math
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlp.plots._data import FIGS_DIR, OUTPUTS_DIR, em_rate, load_general
from nlp.plots.colors import DOMAIN_ORDER, TASK_ORDER

# Realignment runs write into the same outputs tree as the narrow runs
# (file names start with "mis-..."). Override via $EM_REALIGNMENT_OUTPUTS_DIR.
REALIGN_JUDGE_DIR = os.path.join(
    os.environ.get("EM_REALIGNMENT_OUTPUTS_DIR", OUTPUTS_DIR),
    "judge_scores", "general",
)
OUT_DIR = os.path.join(FIGS_DIR, "realignment")

DOM_ABBR = {"medical": "Med", "sports": "Spo", "finance": "Fin"}
TASK_ABBR = {"advice": "Adv", "summarization": "Summ",
             "tutor": "Tut", "critique": "Crit"}
TASK_FULL = {"advice": "Advice", "summarization": "Summ.",
             "tutor": "Tutor", "critique": "Critique"}

REALIGN_FNAME_RE = re.compile(
    r"mis-(medical|sports|finance)_(advice|summarization|tutor|critique)_misaligned"
    r"__realign-(medical|sports|finance)_(advice|summarization|tutor|critique)_aligned\.jsonl"
)


def load_realign(model_key: str) -> pd.DataFrame:
    """Pool all per-(mis, realign) judged files for one model into a single DF."""
    base = os.path.join(REALIGN_JUDGE_DIR, model_key)
    if not os.path.isdir(base):
        return pd.DataFrame()
    rows = []
    for path in sorted(glob(os.path.join(base, "*.jsonl"))):
        m = REALIGN_FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        mis_d, mis_t, realign_d, realign_t = m.groups()
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["mis_domain"] = mis_d
                row["mis_task"] = mis_t
                row["realign_domain"] = realign_d
                row["realign_task"] = realign_t
                rows.append(row)
    return pd.DataFrame(rows)


def _ft_cells():
    return [(d, t) for d in DOMAIN_ORDER for t in TASK_ORDER]


def _ft_label(d, t):
    return f"{DOM_ABBR[d]}-{TASK_ABBR[t]}"


def _heatmap(em, *, row_labels, col_labels, xlabel, ylabel, out_path,
             fig_w, fig_h, annot_fs, vmax):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(em, vmin=0, vmax=vmax, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right", fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    for i in range(em.shape[0]):
        for j in range(em.shape[1]):
            v = em[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=annot_fs - 1, color="gray")
                continue
            text_color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=annot_fs, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("EM Rate (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}  (and .pdf)")


def _task_pre_em(df_pre):
    """Pre-repair EM rate per misalignment task (averaged across domains by
    pooling all rows with that ft_task)."""
    return {t: em_rate(df_pre[df_pre["ft_task"] == t]) for t in TASK_ORDER}


def _cell_pre_em(df_pre):
    return {
        (d, t): em_rate(df_pre[(df_pre["ft_domain"] == d) & (df_pre["ft_task"] == t)])
        for d in DOMAIN_ORDER for t in TASK_ORDER
    }


def _task_aggregated_post(df_post):
    n = len(TASK_ORDER)
    em = np.full((n, n), np.nan)
    for i, mt in enumerate(TASK_ORDER):
        for j, rt in enumerate(TASK_ORDER):
            em[i, j] = em_rate(df_post[(df_post["mis_task"] == mt)
                                       & (df_post["realign_task"] == rt)])
    return em


def _cell_post(df_post):
    cells = _ft_cells()
    n = len(cells)
    em = np.full((n, n), np.nan)
    for i, (md, mt) in enumerate(cells):
        for j, (rd, rt) in enumerate(cells):
            em[i, j] = em_rate(df_post[(df_post["mis_domain"] == md)
                                       & (df_post["mis_task"] == mt)
                                       & (df_post["realign_domain"] == rd)
                                       & (df_post["realign_task"] == rt)])
    return em


def _round_up_5(x: float) -> float:
    if math.isnan(x) or x <= 0:
        return 5.0
    return float(math.ceil(x / 5.0) * 5.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="misaligned", choices=["misaligned"])
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    candidates = ["qwen2.5-14b", "llama3.1-8b", "olmo3-7b-instruct"]
    models = args.models if args.models else candidates

    for model_key in models:
        df_pre = load_general(model_key, args.variant)
        df_post = load_realign(model_key)
        if df_pre.empty or df_post.empty:
            print(f"[skip] {model_key}: missing pre or realignment data")
            continue

        task_pre = _task_pre_em(df_pre)
        cell_pre = _cell_pre_em(df_pre)

        # vmax per model = highest task-level pre-repair EM (rounded up to 5 for
        # nicer colorbar ticks).
        vmax_raw = max(v for v in task_pre.values() if not math.isnan(v))
        vmax = _round_up_5(vmax_raw)

        # 4×4 task-aggregated.
        em = _task_aggregated_post(df_post)
        out = os.path.join(OUT_DIR,
                           f"realignment_task_{model_key}_{args.variant}.png")
        row_labels = [
            f"{TASK_FULL[t]}\n({task_pre[t]:.1f})" for t in TASK_ORDER
        ]
        col_labels = [TASK_FULL[t] for t in TASK_ORDER]
        _heatmap(em,
                 row_labels=row_labels, col_labels=col_labels,
                 xlabel="Realignment Task",
                 ylabel="Misalignment Task",
                 out_path=out, fig_w=3.6, fig_h=3.2, annot_fs=14, vmax=vmax)

        # 12×12 cell-level — same vmax so colors are comparable to the 4×4.
        em = _cell_post(df_post)
        out = os.path.join(OUT_DIR,
                           f"realignment_cell_{model_key}_{args.variant}.png")
        cells = _ft_cells()
        row_labels = [
            f"{_ft_label(d, t)} ({cell_pre[(d, t)]:.1f})" for d, t in cells
        ]
        col_labels = [_ft_label(d, t) for d, t in cells]
        _heatmap(em,
                 row_labels=row_labels, col_labels=col_labels,
                 xlabel="Realignment Cell",
                 ylabel="Misalignment Cell",
                 out_path=out, fig_w=8.5, fig_h=7.0, annot_fs=10, vmax=vmax)


if __name__ == "__main__":
    main()
