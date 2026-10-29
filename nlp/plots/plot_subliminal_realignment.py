"""Subliminal-realignment task-task heatmaps for the paper.

For each (model, channel) we produce a 4x4 task-aggregated heatmap of the
post-realignment narrow-eval EM rate (%):
  rows: realignment subject task (the task on which the misaligned narrow
        LoRA student was originally fine-tuned, i.e. the student that
        teacher-mediated training is trying to repair).
  cols: evaluation task.
Each cell pools the rows of the 3 subject domains x 3 eval domains (~400
prompts per eval cell) that share the (subject task, eval task) combination.

Channels (paper labels -> method folder): SFT -> sft, OPTD -> optd, OPD -> opd.

Inputs: the final LoRA of each run of nlp/subliminal/scripts/run_realignment.sh,
  $SUBLIMINAL_OUTPUTS_DIR/realignment/judge_scores/{method}/{run}/{eval_cell}_val.jsonl

Outputs (PDF + PNG): {FIGS_DIR}/subliminal_realignment/
  realignment_task_<model_key>_<method>.{pdf,png}
"""

import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlp.plots._data import FIGS_DIR, em_rate
from nlp.plots.colors import TASK_ORDER
from nlp.subliminal import config as scfg

OUT_DIR = os.path.join(FIGS_DIR, "subliminal_realignment")
JUDGE_ROOT = os.path.join(scfg.OUTPUTS_ROOT, "realignment", "judge_scores")

MODEL_KEYS = ["llama3.1-8b", "olmo3-7b-instruct"]

# (display label, method folder)
METHODS = [
    ("SFT", "sft"),
    ("OPTD", "optd"),
    ("OPD", "opd"),
]

TASK_FULL = {
    "advice": "Advice",
    "summarization": "Summ.",
    "tutor": "Tutor",
    "critique": "Critique",
}

# Subject run name -> (subject_domain, subject_task)
RUN_RE = re.compile(r"__on_([a-z]+)_([a-z]+)_misaligned__")
# eval cell file basename -> (domain, task)
CELL_RE = re.compile(r"^([a-z]+)_([a-z]+)_misaligned$")

VMAX = 90.0  # fixed colour-bar range across every panel


def load_method(model_key: str, method: str) -> pd.DataFrame:
    rows = []
    for run_path in sorted(glob(os.path.join(JUDGE_ROOT, method, f"{model_key}__*"))):
        m = RUN_RE.search(os.path.basename(run_path))
        if not m:
            continue
        ft_d, ft_t = m.group(1), m.group(2)
        # Final LoRA only: its eval files sit at the run root (checkpoints are under ckpt/).
        for f in glob(os.path.join(run_path, "*_val.jsonl")):
            ec = CELL_RE.match(os.path.basename(f)[: -len("_val.jsonl")])
            if not ec:
                continue
            with open(f) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    rows.append({
                        "ft_domain": ft_d, "ft_task": ft_t,
                        "domain": ec.group(1), "task": ec.group(2),
                        "aligned": r.get("aligned"),
                        "coherent": r.get("coherent"),
                        "aligned_status": r.get("aligned_status"),
                        "coherent_status": r.get("coherent_status"),
                    })
    return pd.DataFrame(rows)


def task_grid(df: pd.DataFrame) -> np.ndarray:
    """Marginal 4x4 EM grid: rows = subject task, cols = eval task."""
    n = len(TASK_ORDER)
    em = np.full((n, n), np.nan)
    if df.empty:
        return em
    for i, ft_t in enumerate(TASK_ORDER):
        for j, ev_t in enumerate(TASK_ORDER):
            em[i, j] = em_rate(df[(df["ft_task"] == ft_t) & (df["task"] == ev_t)])
    return em


def heatmap(em: np.ndarray, *, out_path: str,
            fig_w: float = 3.6, fig_h: float = 3.2, annot_fs: int = 14):
    labels = [TASK_FULL[t] for t in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(em, vmin=0, vmax=VMAX, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Eval Task", fontsize=12)
    ax.set_ylabel("Realignment Subject Task", fontsize=12)

    for i in range(em.shape[0]):
        for j in range(em.shape[1]):
            v = em[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=annot_fs - 1, color="gray")
                continue
            tc = "white" if v > VMAX * 0.55 else "black"
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for model_key in MODEL_KEYS:
        for label, method in METHODS:
            df = load_method(model_key, method)
            grid = task_grid(df)
            print(f"[{model_key} / {label}] rows={len(df):>6}  "
                  f"finite cells={int(np.isfinite(grid).sum())}/16  vmax={VMAX:.0f}")
            if df.empty:
                continue
            heatmap(grid, out_path=os.path.join(OUT_DIR, f"realignment_task_{model_key}_{method}.pdf"))


if __name__ == "__main__":
    main()
