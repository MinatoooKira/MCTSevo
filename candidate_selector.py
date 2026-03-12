"""Diversified candidate mutation selection from an LLR matrix."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import (
    AMINO_ACIDS,
    CANDIDATE_PER_POSITION,
    DIVERSITY_WINDOW,
    GLOBAL_TOP_N,
)


def select_candidates(
    llr_matrix: np.ndarray,
    wt_sequence: str,
    candidates_per_position: int = CANDIDATE_PER_POSITION,
    diversity_window: int = DIVERSITY_WINDOW,
    global_top_n: int = GLOBAL_TOP_N,
) -> List[Tuple[int, str, str]]:
    """Select a diversified set of candidate mutations.

    Returns a list of ``(position, wt_aa, mut_aa)`` tuples that form the
    action space for MCTS.  The strategy:

    1. Keep only beneficial mutations (LLR > 0).
    2. Divide positions into windows of *diversity_window* residues.
    3. Within each window, pick up to *candidates_per_position* top mutations
       per position (but only the overall best across the window are kept, to
       limit total count for very long sequences).
    4. Merge with the global top-*global_top_n* mutations (regardless of
       window) to guarantee that the strongest signals are always included.
    """
    L, _ = llr_matrix.shape

    # ── 1. Collect all beneficial mutations ─────────────────────────────
    all_beneficial: List[Tuple[float, int, int]] = []  # (llr, pos, aa_idx)
    for pos in range(L):
        wt_aa = wt_sequence[pos]
        for aa_idx, aa in enumerate(AMINO_ACIDS):
            if aa == wt_aa:
                continue
            score = llr_matrix[pos, aa_idx]
            if score > 0:
                all_beneficial.append((score, pos, aa_idx))

    if not all_beneficial:
        # Fallback: take the least-negative mutations
        for pos in range(L):
            wt_aa = wt_sequence[pos]
            scores = []
            for aa_idx, aa in enumerate(AMINO_ACIDS):
                if aa == wt_aa:
                    continue
                scores.append((llr_matrix[pos, aa_idx], pos, aa_idx))
            scores.sort(reverse=True)
            all_beneficial.extend(scores[:1])

    all_beneficial.sort(reverse=True)

    # ── 2. Window-based diverse selection ───────────────────────────────
    window_selected: set[Tuple[int, int]] = set()  # (pos, aa_idx)
    n_windows = max(1, (L + diversity_window - 1) // diversity_window)

    for w in range(n_windows):
        w_start = w * diversity_window
        w_end = min(L, w_start + diversity_window)

        window_muts: List[Tuple[float, int, int]] = [
            (s, p, a) for s, p, a in all_beneficial if w_start <= p < w_end
        ]
        window_muts.sort(reverse=True)

        per_pos_count: dict[int, int] = {}
        for score, pos, aa_idx in window_muts:
            cnt = per_pos_count.get(pos, 0)
            if cnt < candidates_per_position:
                window_selected.add((pos, aa_idx))
                per_pos_count[pos] = cnt + 1

    # ── 3. Global top-N ─────────────────────────────────────────────────
    global_selected: set[Tuple[int, int]] = set()
    for _, pos, aa_idx in all_beneficial[:global_top_n]:
        global_selected.add((pos, aa_idx))

    # ── 4. Merge ────────────────────────────────────────────────────────
    merged = window_selected | global_selected
    candidates = [
        (pos, wt_sequence[pos], AMINO_ACIDS[aa_idx])
        for pos, aa_idx in sorted(merged)
    ]

    print(f"[Candidates] Selected {len(candidates)} candidate mutations "
          f"(window-diverse: {len(window_selected)}, global top: {len(global_selected)})")
    return candidates
