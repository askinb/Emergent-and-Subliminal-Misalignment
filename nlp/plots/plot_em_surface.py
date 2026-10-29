"""EM-surface bar plots (paper figure).

Per (model, variant) on the GENERAL eval set: 2×2 subplot grid, one per train task.
Each subplot: 3 bars (High / Medium / Low em_surface), colored by train task color.
Bar with highest EM: thick edge. No figure title, no legend. Height=5.0" to align.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from nlp.plots._data import FIGS_DIR, em_rate, list_models, load_general
from nlp.plots.colors import TASK_COLORS, TASK_ORDER

OUT_DIR = os.path.join(FIGS_DIR, "em_surface")
SURFACES = ["high", "medium", "low"]
SURFACE_LABELS = ["High", "Medium", "Low"]


def _lbl(task: str) -> str:
    return "Summ." if task == "summarization" else task.capitalize()


def _cell_em(df, ft_task: str, surface: str) -> float:
    sub = df[(df["ft_task"] == ft_task) & (df["em_surface"] == surface)]
    return em_rate(sub)


def _plot_one(df, *, model_key: str, out_path: str) -> None:
    if df.empty:
        print(f"[skip] {model_key}: no rows")
        return

    em = np.full((len(TASK_ORDER), len(SURFACES)), np.nan)
    for j, tt in enumerate(TASK_ORDER):
        for k, s in enumerate(SURFACES):
            em[j, k] = _cell_em(df, ft_task=tt, surface=s)

    if np.all(np.isnan(em)):
        print(f"[skip] {model_key}: all-NaN")
        return

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0), sharey=False)
    axes_flat = axes.flatten()
    xs = np.arange(len(SURFACES))

    for j, (tt, ax) in enumerate(zip(TASK_ORDER, axes_flat)):
        row = em[j]
        if np.all(np.isnan(row)):
            ax.set_visible(False)
            continue
        best_k = int(np.nanargmax(row))
        for k in range(len(SURFACES)):
            y = row[k]
            if np.isnan(y):
                continue
            ax.bar(xs[k], y, width=0.65,
                   color=TASK_COLORS[tt],
                   edgecolor="black",
                   linewidth=2.2 if k == best_k else 0.5,
                   zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(SURFACE_LABELS, rotation=22, ha="right", fontsize=13)
        ax.set_title(f"Train Task: {_lbl(tt)}", fontsize=15, pad=4)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(5.0, float(np.nanmax(row)) * 1.18))
        if j % 2 == 0:
            ax.set_ylabel("EM Rate (%)", fontsize=14)

    fig.tight_layout(pad=0.5, h_pad=1.4, w_pad=0.6)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}  (and .pdf)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="misaligned", choices=["misaligned"])
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    models = args.models if args.models else list_models()
    if not models:
        raise SystemExit("no models found")

    for model_key in models:
        df = load_general(model_key, variant=args.variant)
        out_path = os.path.join(OUT_DIR, f"em_surface_{model_key}_{args.variant}.png")
        _plot_one(df, model_key=model_key, out_path=out_path)


if __name__ == "__main__":
    main()
