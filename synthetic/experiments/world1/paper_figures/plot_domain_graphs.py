"""World 1 domain graph figure.

5 domains × 6 tasks grid. Each panel: directed transition graph after
task-specific edge deletions. Retained edges drawn with opacity and width
proportional to transition probability; deleted edges shown as dashed red.
Node positions fixed per domain row for cross-task comparison.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_gen_v3.world import World
from data_gen_v3.similarity import effective_deleted_edges, transition_matrix_similarity

EXP   = Path(__file__).resolve().parents[1]
WORLD = EXP / "world.json"
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path, default=EXP / "figures")
OUT = ap.parse_args().out_dir
OUT.mkdir(parents=True, exist_ok=True)

w    = World.load(str(WORLD))
ND   = len(w.domains)   # 5
NT   = len(w.tasks)     # 6

sim = [1.0 if i == 0 else transition_matrix_similarity(
    w.domains[0].graph, w.domains[i].graph, w.config.num_steps
) for i in range(ND)]

TASK_NAMES = ["first/last", "most/least", "2nd/penult",
              "maj/min",    "extr/mid",   "uniq/rep"]

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":      8,
    "savefig.dpi":    300,
})

C_NODE    = "#2166ac"
C_EDGE    = "#555555"
C_DEL     = "#d6604d"
C_NODE_D0 = "#4dac26"   # shared-with-D0 steps (cosmetic only on D0)
PROB_FLOOR = 0.06        # edges below this prob drawn at minimum alpha

# ── fixed circular layout ──────────────────────────────────────────────────
def circle_pos(n):
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}

# ── draw one panel ─────────────────────────────────────────────────────────
def draw_panel(ax, domain, task, deleted_pairs, highlight_cell=False):
    graph = domain.graph
    n     = graph.num_nodes
    pos   = circle_pos(n)

    # map deleted global step-pairs → local indices
    sid_to_local = {sid: li for li, sid in enumerate(graph.step_ids)}
    deleted_local = set()
    for sg, dg in deleted_pairs:
        sl = sid_to_local.get(sg)
        dl = sid_to_local.get(dg)
        if sl is not None and dl is not None:
            deleted_local.add((sl, dl))

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.set_facecolor("#f9f9f9" if not highlight_cell else "#fff8e7")
    for spine in ax.spines.values():
        spine.set_visible(highlight_cell)
        if highlight_cell:
            spine.set_edgecolor("#222222")
            spine.set_linewidth(1.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ── draw retained edges (behind deleted) ──
    for src_l, dsts in graph.adjacency.items():
        probs = graph.transition_probs[src_l]
        for dst_l, p in zip(dsts, probs):
            if (src_l, dst_l) in deleted_local:
                continue
            p = max(p, PROB_FLOOR)
            alpha = 0.15 + 0.65 * p
            lw    = 0.35 + 1.8 * p
            p0, p1 = pos[src_l], pos[dst_l]
            ax.annotate(
                "", xy=p1, xytext=p0,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=C_EDGE, alpha=alpha, lw=lw,
                    connectionstyle="arc3,rad=0.12",
                    mutation_scale=4,
                ),
                zorder=2,
            )

    # ── draw deleted edges (on top, always visible) ──
    for src_l, dst_l in deleted_local:
        p0, p1 = pos[src_l], pos[dst_l]
        ax.annotate(
            "", xy=p1, xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_DEL, alpha=0.90, lw=1.1,
                linestyle="dashed",
                connectionstyle="arc3,rad=0.12",
                mutation_scale=5,
            ),
            zorder=4,
        )

    # ── draw nodes ──
    for i, p in pos.items():
        ax.plot(*p, "o", color=C_NODE, ms=4.5, zorder=5,
                markeredgecolor="white", markeredgewidth=0.4)
        # node label offset slightly outward
        off = p * 1.26
        ax.text(off[0], off[1], str(i), ha="center", va="center",
                fontsize=4.5, color="#222222", zorder=6)

# ── build figure ───────────────────────────────────────────────────────────
def build_figure(highlight_misalign_cell: bool, basename: str):
    fig, axes = plt.subplots(
        ND, NT,
        figsize=(12, 10.5),
        gridspec_kw=dict(hspace=0.10, wspace=0.06,
                         left=0.09, right=0.99, top=0.93, bottom=0.06),
    )

    for di, domain in enumerate(w.domains):
        for ti, task in enumerate(w.tasks):
            ax  = axes[di, ti]
            eff = effective_deleted_edges(task, domain)
            hl  = highlight_misalign_cell and (di == 0 and ti == 5)
            draw_panel(ax, domain, task, eff, highlight_cell=hl)

            if di == 0:
                ax.set_title(f"$T_{ti}$\n({TASK_NAMES[ti]})",
                             fontsize=7.5, pad=3, fontfamily="serif")

        axes[di, 0].set_ylabel(
            f"$D_{di}$\n(sim $= {sim[di]:.2f}$)",
            fontsize=7.5, labelpad=4,
        )

    leg_handles = [
        mpatches.Patch(facecolor=C_NODE,  edgecolor="white", lw=0.4,
                       label="step node"),
        mpatches.Patch(facecolor=C_EDGE,  alpha=0.6,
                       label="retained edge (width $\\propto$ prob.)"),
        mpatches.Patch(facecolor=C_DEL,   alpha=0.9,
                       label="deleted edge (task-specific)"),
    ]
    if highlight_misalign_cell:
        leg_handles.append(
            mpatches.Patch(facecolor="#fff8e7", edgecolor="#222222", lw=1.5,
                           label="misalignment training cell $(D_0, T_5)$")
        )
    fig.legend(handles=leg_handles, loc="lower center",
               bbox_to_anchor=(0.54, 0.0),
               ncol=4 if highlight_misalign_cell else 3,
               fontsize=7.5, framealpha=0.95, edgecolor="#cccccc")

    fig.suptitle(
        r"World~1 domain--task graphs after edge deletions"
        "\n"
        r"(rows $D_0$--$D_4$ ordered by transition-matrix similarity to $D_0$;"
        r"  columns $T_0$--$T_5$ by task type)",
        fontsize=9, y=0.99, fontfamily="serif",
    )

    for suf in ("pdf", "png"):
        fig.savefig(OUT / f"{basename}.{suf}",
                    bbox_inches="tight", dpi=300)
        print(f"Saved {OUT}/{basename}.{suf}")
    plt.close(fig)


build_figure(highlight_misalign_cell=True,  basename="world1_domain_graphs")
build_figure(highlight_misalign_cell=False, basename="world1_domain_graphs_no_highlight")
