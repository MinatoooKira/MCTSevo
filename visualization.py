"""LLR heatmap generation with matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import AMINO_ACIDS


def plot_llr_heatmap(
    llr_matrix: np.ndarray,
    wt_sequence: str,
    save_path: str | Path,
    candidates: Optional[List[Tuple[int, str, str]]] = None,
    title: str = "ESM-1v Log-Likelihood Ratio Heatmap",
    figsize_per_residue: float = 0.18,
    min_fig_width: float = 12.0,
) -> Path:
    """Draw an LLR heatmap (positions x amino acids) and save to *save_path*.

    Parameters
    ----------
    llr_matrix : (L, 20) array
    wt_sequence : wild-type sequence string
    save_path : output image file path
    candidates : optional list of selected candidate mutations ``(pos, wt_aa, mut_aa)``
        to highlight on the heatmap with a marker
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    L = llr_matrix.shape[0]
    fig_width = max(min_fig_width, L * figsize_per_residue)
    fig_height = max(6.0, fig_width * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = max(abs(llr_matrix.min()), abs(llr_matrix.max()), 1e-6)
    im = ax.imshow(
        llr_matrix.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # Axis labels
    tick_step = max(1, L // 60)
    x_ticks = list(range(0, L, tick_step))
    x_labels = [f"{i+1}\n{wt_sequence[i]}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=max(5, 8 - L // 100))

    ax.set_yticks(range(20))
    ax.set_yticklabels(AMINO_ACIDS, fontsize=8)

    ax.set_xlabel("Position (1-indexed) / WT Residue", fontsize=10)
    ax.set_ylabel("Amino Acid", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)

    # Mark WT residues
    for i, aa in enumerate(wt_sequence):
        if aa in AMINO_ACIDS:
            j = AMINO_ACIDS.index(aa)
            ax.plot(i, j, marker="s", color="black", markersize=2.5, alpha=0.6)

    # Highlight candidate mutations
    if candidates:
        cand_set: Set[Tuple[int, int]] = set()
        for pos, _wt, mut in candidates:
            if mut in AMINO_ACIDS:
                cand_set.add((pos, AMINO_ACIDS.index(mut)))
        for pos, aa_idx in cand_set:
            ax.plot(pos, aa_idx, marker="o", markeredgecolor="lime",
                    markerfacecolor="none", markersize=4, markeredgewidth=1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Log-Likelihood Ratio", fontsize=9)

    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="black", label="WT residue"),
    ]
    if candidates:
        legend_elements.append(
            mpatches.Patch(facecolor="none", edgecolor="lime", label="Candidate mutation")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Heatmap] Saved to {save_path}")
    return save_path
