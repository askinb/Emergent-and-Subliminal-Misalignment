"""Task x task transfer bar plots (paper figure).

Per (model, eval_set, variant): 2×2 subplot grid, one per eval task.
Each subplot: 4 bars (one per train task), colored by train task.
Bar with highest EM per subplot: thick edge. train_task == eval_task: hatched.
No figure title, no legend. Height=5.0" to align with heatmap figures.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from nlp.plots._data import FIGS_DIR, em_rate, list_models, load_general, load_narrow
from nlp.plots.colors import TASK_COLORS, TASK_ORDER

OUT_DIR = os.path.join(FIGS_DIR, "task_transfer")


def _lbl(task: str) -> str:
    return "Summ." if task == "summarization" else task.capitalize()


def _cell_em(df, ft_task: str, eval_task: str) -> float:
    sub = df[(df["ft_task"] == ft_task) & (df["task"] == eval_task)]
    return em_rate(sub)


def _plot_one(df, *, model_key: str, out_path: str) -> None:
    if df.empty:
        print(f"[skip] {model_key} ({out_path}): no rows")
        return

    n = len(TASK_ORDER)
    em = np.full((n, n), np.nan)
    for i, et in enumerate(TASK_ORDER):
        for j, tt in enumerate(TASK_ORDER):
            em[i, j] = _cell_em(df, ft_task=tt, eval_task=et)

    if np.all(np.isnan(em)):
        print(f"[skip] {model_key} ({out_path}): all-NaN")
        return

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0), sharey=False)
    axes_flat = axes.flatten()
    xs = np.arange(n)
    x_labels = [_lbl(t) for t in TASK_ORDER]

    for i, (et, ax) in enumerate(zip(TASK_ORDER, axes_flat)):
        row = em[i]
        if np.all(np.isnan(row)):
            ax.set_visible(False)
            continue
        best_j = int(np.nanargmax(row))
        for j, tt in enumerate(TASK_ORDER):
            y = row[j]
            if np.isnan(y):
                continue
            ax.bar(xs[j], y, width=0.75,
                   color=TASK_COLORS[tt],
                   edgecolor="black",
                   linewidth=2.2 if j == best_j else 0.5,
                   hatch="///" if tt == et else None,
                   zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=22, ha="right", fontsize=13)
        ax.set_title(f"Eval Task: {_lbl(et)}", fontsize=15, pad=4)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(5.0, float(np.nanmax(row)) * 1.18))
        if i % 2 == 0:
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
        raise SystemExit("no models found under $EM_OUTPUTS_DIR/judge_scores/")

    for model_key in models:
        for eval_set, loader in [("narrow", load_narrow), ("general", load_general)]:
            df = loader(model_key, variant=args.variant)
            out_path = os.path.join(
                OUT_DIR, f"task_transfer_{eval_set}_{model_key}_{args.variant}.png"
            )
            _plot_one(df, model_key=model_key, out_path=out_path)


if __name__ == "__main__":
    main()
