"""Cell-level transfer heatmaps (appendix figures).

Per (model, variant) we produce two heatmaps:

  cell_transfer_narrow_{model}_{variant}.pdf
      12 fine-tuning cells (rows) × 12 evaluation cells (cols), all on \\emdata.

  cell_transfer_general_{model}_{variant}.pdf
      12 fine-tuning cells (rows) × 4 evaluation tasks (cols), on \\generaldata.

Both rows and 12-cell columns are ordered (medical, sports, finance) ×
(advice, summarization, tutor, critique). Cell labels are abbreviated as
"Med-Adv" / "Spo-Summ" / "Fin-Tut" etc.

Cells are colored by EM rate (%); the only annotation is the EM rate (white,
bold). Coherence is omitted to keep the cells readable at the 12×12 scale.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from nlp.plots._data import FIGS_DIR, em_rate, list_models, load_general, load_narrow
from nlp.plots.colors import DOMAIN_ORDER, TASK_ORDER

OUT_DIR = os.path.join(FIGS_DIR, "cell_transfer")

DOM_ABBR = {"medical": "Med", "sports": "Spo", "finance": "Fin"}
TASK_ABBR = {"advice": "Adv", "summarization": "Summ",
             "tutor": "Tut", "critique": "Crit"}
TASK_FULL = {"advice": "Advice", "summarization": "Summ.",
             "tutor": "Tutor", "critique": "Critique"}


def _ft_cells():
    return [(d, t) for d in DOMAIN_ORDER for t in TASK_ORDER]


def _ft_label(d, t):
    return f"{DOM_ABBR[d]}-{TASK_ABBR[t]}"


def _grid_narrow(df):
    cells = _ft_cells()
    em = np.full((len(cells), len(cells)), np.nan)
    for i, (ft_d, ft_t) in enumerate(cells):
        for j, (ev_d, ev_t) in enumerate(cells):
            sub = df[
                (df["ft_domain"] == ft_d) & (df["ft_task"] == ft_t)
                & (df["domain"] == ev_d) & (df["task"] == ev_t)
            ]
            em[i, j] = em_rate(sub)
    return em


def _grid_general(df):
    cells = _ft_cells()
    em = np.full((len(cells), len(TASK_ORDER)), np.nan)
    for i, (ft_d, ft_t) in enumerate(cells):
        for j, ev_t in enumerate(TASK_ORDER):
            sub = df[
                (df["ft_domain"] == ft_d) & (df["ft_task"] == ft_t)
                & (df["task"] == ev_t)
            ]
            em[i, j] = em_rate(sub)
    return em


def _heatmap(em, *, row_labels, col_labels, xlabel, ylabel, out_path,
             fig_w, fig_h, annot_fs):
    finite = em[~np.isnan(em)]
    vmax = float(np.ceil(finite.max() / 5.0) * 5.0) if finite.size else 100.0
    vmax = max(vmax, 5.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(em, vmin=0, vmax=vmax, cmap="Reds", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=27, ha="right", fontsize=10)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="misaligned", choices=["misaligned"])
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    models = args.models if args.models else list_models()
    if not models:
        raise SystemExit("no models found")

    cells = _ft_cells()
    ft_labels = [_ft_label(d, t) for d, t in cells]
    task_labels = [TASK_FULL[t] for t in TASK_ORDER]

    for model_key in models:
        df = load_narrow(model_key, args.variant)
        if not df.empty:
            em = _grid_narrow(df)
            out = os.path.join(
                OUT_DIR, f"cell_transfer_narrow_{model_key}_{args.variant}.png"
            )
            _heatmap(em, row_labels=ft_labels, col_labels=ft_labels,
                     xlabel="Eval Cell", ylabel="Train Cell",
                     out_path=out, fig_w=8.5, fig_h=7.0, annot_fs=10)
        else:
            print(f"[skip] {model_key} narrow: no data")

        df = load_general(model_key, args.variant)
        if not df.empty:
            em = _grid_general(df)
            out = os.path.join(
                OUT_DIR, f"cell_transfer_general_{model_key}_{args.variant}.png"
            )
            _heatmap(em, row_labels=ft_labels, col_labels=task_labels,
                     xlabel="Eval Task", ylabel="Train Cell",
                     out_path=out, fig_w=4.5, fig_h=7.0, annot_fs=11)
        else:
            print(f"[skip] {model_key} general: no data")


if __name__ == "__main__":
    main()
