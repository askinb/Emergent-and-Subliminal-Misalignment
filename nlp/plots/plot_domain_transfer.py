"""Domain x domain transfer heatmaps (paper figure).

One figure per model per eval_set. No title, no coherence annotation — just EM
rate in large font. Height=5.0" to match 2×2 bar plot figures in the same row.

Outputs:
  domain_transfer_narrow_{model_key}_{variant}.png   3×3 heatmap
  domain_transfer_general_{model_key}_{variant}.png  3×29 heatmap
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from nlp.plots._data import (
    FIGS_DIR,
    _exclude_invalid, em_rate, list_models, load_general, load_narrow,
)
from nlp.plots.colors import DOMAIN_ORDER

OUT_DIR = os.path.join(FIGS_DIR, "domain_transfer")
NARROW_EVAL_DOMAINS = DOMAIN_ORDER


def _domain_lbl(d: str) -> str:
    return d.replace("_", " ").title()


def _grid(df, train_domains, eval_domains):
    em = np.full((len(train_domains), len(eval_domains)), np.nan)
    for i, td in enumerate(train_domains):
        for j, ed in enumerate(eval_domains):
            sub = df[(df["ft_domain"] == td) & (df["domain"] == ed)]
            em[i, j] = em_rate(sub)
    return em


def _plot_one(df, *, eval_domains, model_key: str, out_path: str,
              annotate: bool, fig_w: float, fig_h: float) -> None:
    if df.empty:
        print(f"[skip] {model_key} ({out_path}): no rows")
        return

    em = _grid(df, DOMAIN_ORDER, eval_domains)
    if np.all(np.isnan(em)):
        print(f"[skip] {model_key} ({out_path}): all-NaN")
        return

    finite = em[~np.isnan(em)]
    vmax = float(np.ceil(finite.max() / 5.0) * 5.0) if finite.size else 100.0
    vmax = max(vmax, 5.0)

    row_labels = [_domain_lbl(d) for d in DOMAIN_ORDER]
    col_labels = [_domain_lbl(d) for d in eval_domains]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(em, vmin=0, vmax=vmax, cmap="Reds", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=27, ha="right", fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlabel("Eval Domain", fontsize=13)
    ax.set_ylabel("Train Domain", fontsize=13)

    if annotate:
        for i in range(em.shape[0]):
            for j in range(em.shape[1]):
                v = em[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=10, color="gray")
                    continue
                text_color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=text_color)

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

    for model_key in models:
        # Narrow: 3×3
        df = load_narrow(model_key, args.variant)
        out = os.path.join(OUT_DIR, f"domain_transfer_narrow_{model_key}_{args.variant}.png")
        _plot_one(df, eval_domains=NARROW_EVAL_DOMAINS, model_key=model_key,
                  out_path=out, annotate=True, fig_w=3.6, fig_h=3.2)

        # General: 3×29
        df = load_general(model_key, args.variant)
        if not df.empty:
            eval_domains = sorted(df["domain"].unique())
            out = os.path.join(OUT_DIR,
                               f"domain_transfer_general_{model_key}_{args.variant}.png")
            _plot_one(df, eval_domains=eval_domains, model_key=model_key,
                      out_path=out, annotate=False, fig_w=7.0, fig_h=3.2)
        else:
            print(f"[skip] {model_key} general: no rows")


if __name__ == "__main__":
    main()
