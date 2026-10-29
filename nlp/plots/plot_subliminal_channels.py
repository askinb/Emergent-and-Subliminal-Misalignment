"""Subliminal transfer through SFT, OPTD and OPD: tables + heatmaps for the paper.

Per model in {llama3.1-8b, qwen3-14b, olmo3-7b-instruct}:

  Tables (LaTeX):
    task_agg_{tag}.tex            - 4 methods x 4 tasks
    domain_agg_{tag}.tex          - 4 methods x 3 domains
    task_agg_per_epoch_{tag}.tex  - 3 methods x 4 tasks, epoch 1 / 2 / 3
                                    (only if the per-epoch runs exist)

  Heatmaps (PDF + PNG):
    heatmap_{tag}_{method}.pdf  for method in {sft, optd, opd}  (no Teacher heatmap)

And over all models:
    opd_filtered.tex - OPD with and without the inline filter: mean EM rate over
                       the 132 off-diagonal (training-cell, eval-cell) pairs

Methods (display names -> source):
  Teacher          -> narrow judge scores of nlp.em_transfer (the misaligned teacher itself)
  SFT / OPTD / OPD -> $SUBLIMINAL_OUTPUTS_DIR/channels/cross_eval.csv, final LoRA
  per epoch        -> $SUBLIMINAL_OUTPUTS_DIR/channels_per_epoch/cross_eval.csv
Build the CSVs with `python -m nlp.subliminal.cross_eval --exp <folder>`.

Aggregation: build the 12x12 grid of per-cell EM%, then average the cells in
each (task or domain) slice (avg-of-cells, NOT pool-of-rows).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlp.plots._data import FIGS_DIR, em_rate, load_narrow
from nlp.plots.colors import DOMAIN_ORDER, TASK_ORDER
from nlp.subliminal import config as scfg

OUT_DIR = os.path.join(FIGS_DIR, "subliminal_channels")

# (model_key, file / label tag, display label)
MODELS = [
    ("llama3.1-8b", "llama", "Llama-3.1-8B"),
    ("qwen3-14b", "qwen", "Qwen3-14B"),
    ("olmo3-7b-instruct", "olmo", "OLMo-3-7B-Instruct"),
]

# (display label, method in cross_eval.csv)
METHODS = [
    ("Teacher", None),
    ("SFT", "sft"),
    ("OPTD", "optd"),
    ("OPD", "opd"),
]

DOM_ABBR = {"medical": "Med", "sports": "Spo", "finance": "Fin"}
TASK_ABBR = {"advice": "Adv", "summarization": "Summ",
             "tutor": "Tut", "critique": "Crit"}

CELLS = [(d, t) for d in DOMAIN_ORDER for t in TASK_ORDER]
CELL_IDX = {f"{d}_{t}_misaligned": i for i, (d, t) in enumerate(CELLS)}


# ---------- Teacher data: narrow judge scores of the misaligned teachers ----------
def teacher_grid(model_key: str) -> np.ndarray:
    df = load_narrow(model_key, "misaligned")
    em = np.full((len(CELLS), len(CELLS)), np.nan)
    if df.empty:
        return em
    for i, (ft_d, ft_t) in enumerate(CELLS):
        for j, (ev_d, ev_t) in enumerate(CELLS):
            sub = df[
                (df["ft_domain"] == ft_d) & (df["ft_task"] == ft_t)
                & (df["domain"] == ev_d) & (df["task"] == ev_t)
            ]
            em[i, j] = em_rate(sub)
    return em


# ---------- Channel runs: rows of cross_eval.csv ----------
def channel_rows(exp: str, model_key: str) -> pd.DataFrame:
    """Runs with student = base and teacher = data generator = one misaligned cell."""
    path = os.path.join(scfg.OUTPUTS_ROOT, exp, "cross_eval.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    prefix = f"{model_key}__on_"
    df = df[(df["model_key"] == model_key) & (df["student"] == "base")
            & (df["teacher"] == df["data"]) & df["teacher"].str.startswith(prefix)].copy()
    df["training_cell"] = df["teacher"].str[len(prefix):]
    return df


def method_grid(df: pd.DataFrame, method: str, ckpt: str = "final") -> np.ndarray:
    em = np.full((len(CELLS), len(CELLS)), np.nan)
    if df.empty:
        return em
    for _, row in df[(df["method"] == method) & (df["ckpt"] == ckpt)].iterrows():
        i = CELL_IDX.get(row["training_cell"])
        j = CELL_IDX.get(row["eval_cell"])
        if i is not None and j is not None:
            em[i, j] = float(row["em_rate"])
    return em


def method_grids_per_epoch(df: pd.DataFrame, method: str) -> list[np.ndarray]:
    """Three 12x12 grids (epoch 1, 2, 3). The epoch-k LoRA is the k-th checkpoint
    of a run in training order (ckpt/step_N sorted by N, then the final LoRA)."""
    grids = [np.full((len(CELLS), len(CELLS)), np.nan) for _ in range(3)]
    if df.empty:
        return grids
    sub = df[df["method"] == method]
    for cell, g in sub.groupby("training_cell"):
        i = CELL_IDX.get(cell)
        if i is None:
            continue
        ckpts = set(g["ckpt"])
        order = sorted(ckpts - {"final"}, key=lambda c: int(c.split("_")[1]))
        order += ["final"] if "final" in ckpts else []
        for e, ck in enumerate(order[:3]):
            for _, row in g[g["ckpt"] == ck].iterrows():
                j = CELL_IDX.get(row["eval_cell"])
                if j is not None:
                    grids[e][i, j] = float(row["em_rate"])
    return grids


# ---------- Aggregation ----------
def _mean(sub: np.ndarray) -> float:
    return float(np.nanmean(sub)) if np.isfinite(sub).any() else float("nan")


def task_avg(em: np.ndarray, task: str) -> float:
    """Avg of cells where ft_task == `task`. 36 cells = 3 ft_domains x 12 eval cells."""
    return _mean(em[[i for i, (_, ft_t) in enumerate(CELLS) if ft_t == task], :])


def domain_avg(em: np.ndarray, domain: str) -> float:
    """Avg of cells where ft_domain == `domain`. 48 cells = 4 ft_tasks x 12 eval cells."""
    return _mean(em[[i for i, (ft_d, _) in enumerate(CELLS) if ft_d == domain], :])


def off_diag_avg(em: np.ndarray) -> float:
    """Avg over the 132 off-diagonal (training-cell, eval-cell) pairs."""
    return _mean(em[~np.eye(len(CELLS), dtype=bool)])


# ---------- LaTeX table rendering ----------
def fmt_em(v: float) -> str:
    return "--" if not np.isfinite(v) else f"{v:.1f}"


def render_task_table(rows_data: dict[str, np.ndarray], model_label: str,
                      label_suffix: str) -> str:
    body = []
    for method_label, _ in METHODS:
        em = rows_data[method_label]
        cells = [fmt_em(task_avg(em, t)) for t in TASK_ORDER]
        body.append(f"        {method_label} & " + " & ".join(cells) + r" \\")
    return (
        r"\begin{table}[h]" "\n"
        r"    \centering" "\n"
        r"    \small" "\n"
        rf"    \caption{{{model_label}: narrow-eval EM rate (\%), aggregated by "
        rf"the teacher's fine-tuning task. Each entry is the mean EM rate over "
        rf"the 36 (training-cell, evaluation-cell) pairs whose teacher's training "
        rf"cell uses the column task (3 fine-tuning domains $\times$ 12 evaluation cells). "
        rf"Teacher denotes the misaligned teacher itself; SFT, OPTD, and OPD are the three "
        rf"distillation channels (supervised fine-tuning, off-policy distillation, "
        rf"on-policy distillation).}}" "\n"
        rf"    \label{{tab:onpolicy-task-{label_suffix}}}" "\n"
        r"    \begin{tabular}{lcccc}" "\n"
        r"        \toprule" "\n"
        r"        Method & Advice & Summarization & Tutor & Critique \\" "\n"
        r"        \midrule" "\n"
        + "\n".join(body) + "\n"
        r"        \bottomrule" "\n"
        r"    \end{tabular}" "\n"
        r"\end{table}" "\n"
    )


def render_task_table_per_epoch(per_epoch_grids: dict[str, list[np.ndarray]],
                                model_label: str, label_suffix: str) -> str:
    """4-column (task) table with method rows; each cell shows
    'epoch1 / epoch2 / epoch3' EM% values."""
    body = []
    for method_label, grids in per_epoch_grids.items():
        cells = [" / ".join(fmt_em(task_avg(em, t)) for em in grids) for t in TASK_ORDER]
        body.append(f"        {method_label} & " + " & ".join(cells) + r" \\")
    return (
        r"\begin{table}[h]" "\n"
        r"    \centering" "\n"
        r"    \small" "\n"
        rf"    \caption{{{model_label}: per-epoch narrow-eval EM rate (\%), "
        rf"aggregated by the teacher's fine-tuning task. Each cell reports three "
        rf"numbers: epoch~1 / epoch~2 / epoch~3, where epoch~$k$ refers to the "
        rf"student adapter saved after $k$ training epochs of the corresponding "
        rf"distillation channel. Within each epoch the value is the mean EM rate "
        rf"over the 36 (training-cell, evaluation-cell) pairs whose teacher's "
        rf"training cell uses the column task. Methods as in "
        rf"\Cref{{tab:onpolicy-task-llama}}.}}" "\n"
        rf"    \label{{tab:onpolicy-task-perepoch-{label_suffix}}}" "\n"
        r"    \begin{tabular}{lcccc}" "\n"
        r"        \toprule" "\n"
        r"        Method & Advice & Summarization & Tutor & Critique \\" "\n"
        r"        \midrule" "\n"
        + "\n".join(body) + "\n"
        r"        \bottomrule" "\n"
        r"    \end{tabular}" "\n"
        r"\end{table}" "\n"
    )


def render_domain_table(rows_data: dict[str, np.ndarray], model_label: str,
                        label_suffix: str) -> str:
    body = []
    for method_label, _ in METHODS:
        em = rows_data[method_label]
        cells = [fmt_em(domain_avg(em, d)) for d in DOMAIN_ORDER]
        body.append(f"        {method_label} & " + " & ".join(cells) + r" \\")
    return (
        r"\begin{table}[h]" "\n"
        r"    \centering" "\n"
        r"    \small" "\n"
        rf"    \caption{{{model_label}: narrow-eval EM rate (\%), aggregated by "
        rf"the teacher's fine-tuning domain. Each entry is the mean EM rate over "
        rf"the 48 (training-cell, evaluation-cell) pairs whose teacher's training "
        rf"cell uses the column domain (4 fine-tuning tasks $\times$ 12 evaluation cells). "
        rf"Methods as in the corresponding task-aggregated table.}}" "\n"
        rf"    \label{{tab:onpolicy-domain-{label_suffix}}}" "\n"
        r"    \begin{tabular}{lccc}" "\n"
        r"        \toprule" "\n"
        r"        Method & Medical & Sports & Finance \\" "\n"
        r"        \midrule" "\n"
        + "\n".join(body) + "\n"
        r"        \bottomrule" "\n"
        r"    \end{tabular}" "\n"
        r"\end{table}" "\n"
    )


def render_opd_filtered_table(values: dict[str, tuple[float, float]]) -> str:
    """values: model label -> (filtered OPD, unfiltered OPD) off-diagonal means."""
    labels = list(values)
    rows = [
        "        OPD (Filtered) & " + " & ".join(fmt_em(values[m][0]) for m in labels) + r" \\",
        "        OPD (Unfiltered) & " + " & ".join(fmt_em(values[m][1]) for m in labels) + r" \\",
    ]
    return (
        r"\begin{table}[h]" "\n"
        r"    \centering" "\n"
        r"    \small" "\n"
        r"    \caption{Narrow-eval EM rate (\%) for filtered and unfiltered OPD after "
        r"three epochs: mean over the 132 off-diagonal (training-cell, evaluation-cell) "
        r"pairs. Filtering student-generated trajectories before training does not "
        r"substantially reduce the transferred misalignment.}" "\n"
        r"    \label{tab:opd-filtered}" "\n"
        r"    \begin{tabular}{l" + "c" * len(labels) + "}\n"
        r"        \toprule" "\n"
        "        Method & " + " & ".join(labels) + r" \\" "\n"
        r"        \midrule" "\n"
        + "\n".join(rows) + "\n"
        r"        \bottomrule" "\n"
        r"    \end{tabular}" "\n"
        r"\end{table}" "\n"
    )


# ---------- Heatmap (matches cell_transfer style, EM% only, no coh) ----------
def heatmap(em: np.ndarray, *, out_path: str, title: str | None = None,
            fig_w: float = 8.5, fig_h: float = 7.0, annot_fs: int = 10):
    finite = em[~np.isnan(em)]
    vmax = float(np.ceil(finite.max() / 5.0) * 5.0) if finite.size else 100.0
    vmax = max(vmax, 5.0)

    labels = [f"{DOM_ABBR[d]}-{TASK_ABBR[t]}" for d, t in CELLS]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(em, vmin=0, vmax=vmax, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=27, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Eval Cell", fontsize=12)
    ax.set_ylabel("Train Cell", fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
    for i in range(em.shape[0]):
        for j in range(em.shape[1]):
            v = em[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=annot_fs - 1, color="gray")
                continue
            tc = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=annot_fs, fontweight="bold", color=tc)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("EM Rate (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}  (and .png)")


def _write(name: str, text: str):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    opd_filtered = {}
    for model_key, tag, model_label in MODELS:
        df = channel_rows("channels", model_key)
        if df.empty:
            print(f"\n=== {model_label}: no channel runs, skip")
            continue
        print(f"\n=== {model_label} ({model_key}) ===")
        em_by_method = {"Teacher": teacher_grid(model_key)}
        for label, method in METHODS[1:]:
            em_by_method[label] = method_grid(df, method)
        for label, em in em_by_method.items():
            print(f"  {label}: 12x12 grid, {int(np.isfinite(em).sum())}/144 finite cells, "
                  f"mean over diagonal = {_mean(np.diag(em)):.1f}, "
                  f"mean off-diag = {off_diag_avg(em):.1f}")
        _write(f"task_agg_{tag}.tex", render_task_table(em_by_method, model_label, tag))
        _write(f"domain_agg_{tag}.tex", render_domain_table(em_by_method, model_label, tag))
        # Heatmaps for the 3 channels only (no Teacher heatmap).
        for label, method in METHODS[1:]:
            heatmap(em_by_method[label], out_path=os.path.join(OUT_DIR, f"heatmap_{tag}_{method}.pdf"))
        opd_filtered[model_label] = (off_diag_avg(method_grid(df, "opd_filtered")),
                                     off_diag_avg(em_by_method["OPD"]))

        # Per-epoch task table (needs the runs of channels_per_epoch).
        df_epoch = channel_rows("channels_per_epoch", model_key)
        if not df_epoch.empty:
            per_epoch = {label: method_grids_per_epoch(df_epoch, method)
                         for label, method in METHODS[1:]}
            for label, grids in per_epoch.items():
                print(f"  per-epoch {label}: finite cells per epoch = "
                      f"{[int(np.isfinite(g).sum()) for g in grids]}")
            _write(f"task_agg_per_epoch_{tag}.tex",
                   render_task_table_per_epoch(per_epoch, model_label, tag))

    if opd_filtered:
        _write("opd_filtered.tex", render_opd_filtered_table(opd_filtered))


if __name__ == "__main__":
    main()
