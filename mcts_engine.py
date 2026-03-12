"""Monte Carlo Tree Search with Progressive Widening over the protein mutation space."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    AMINO_ACIDS,
    AA_TO_INDEX,
    MAX_DEPTH,
    NUM_SIMULATIONS,
    UCB_C,
    PW_K,
    PW_ALPHA,
    SEQUENCES_PER_ROUND,
    DEPTH_QUOTA,
    gpr_alpha,
)
from esm_models import embed_sequence
from gpr_model import FitnessGPR

Mutation = Tuple[int, str, str]


def _apply_mutations(wt_sequence: str, mutations: List[Mutation]) -> str:
    seq = list(wt_sequence)
    for pos, _wt, mut in mutations:
        seq[pos] = mut
    return "".join(seq)


def _hamming(s1: str, s2: str) -> int:
    return sum(a != b for a, b in zip(s1, s2))


# ── Node ────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = (
        "mutations", "parent", "children", "visits", "total_value",
        "_untried_actions", "action",
    )

    def __init__(
        self,
        mutations: List[Mutation],
        parent: Optional["MCTSNode"] = None,
        action: Optional[Mutation] = None,
        candidate_actions_sorted: Optional[List[Mutation]] = None,
    ):
        self.mutations = list(mutations)
        self.parent = parent
        self.action = action
        self.children: Dict[Mutation, "MCTSNode"] = {}
        self.visits: int = 0
        self.total_value: float = 0.0

        if candidate_actions_sorted is not None:
            occupied_positions = {m[0] for m in self.mutations}
            self._untried_actions = [
                a for a in candidate_actions_sorted
                if a[0] not in occupied_positions
            ]
        else:
            self._untried_actions = []

    @property
    def depth(self) -> int:
        return len(self.mutations)

    @property
    def is_terminal(self) -> bool:
        return self.depth >= MAX_DEPTH

    @staticmethod
    def _pw_max_children(visits: int) -> int:
        """Progressive Widening: number of children allowed given current visits."""
        if visits <= 0:
            return 1
        return int(math.ceil(PW_K * (visits ** PW_ALPHA)))

    def should_expand(self) -> bool:
        """Return True if PW allows adding a new child AND there are untried actions."""
        if not self._untried_actions:
            return False
        return len(self.children) < self._pw_max_children(self.visits)

    def ucb(self, c: float = UCB_C) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = c * math.sqrt(2.0 * math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child_ucb(self, c: float = UCB_C) -> "MCTSNode":
        return max(self.children.values(), key=lambda ch: ch.ucb(c))

    def expand(self, candidate_actions_sorted: List[Mutation]) -> "MCTSNode":
        """Pop the best untried action (highest LLR), create a child node."""
        action = self._untried_actions.pop()
        child_mutations = self.mutations + [action]
        child = MCTSNode(
            mutations=child_mutations,
            parent=self,
            action=action,
            candidate_actions_sorted=candidate_actions_sorted,
        )
        self.children[action] = child
        return child


# ── Value function ──────────────────────────────────────────────────────────

class ValueFunction:
    """Combines ESM-1v LLR scores and GPR predictions into a single value."""

    def __init__(
        self,
        wt_sequence: str,
        llr_matrix: np.ndarray,
        gpr_model: FitnessGPR,
    ):
        self.wt_sequence = wt_sequence
        self.llr_matrix = llr_matrix
        self.gpr = gpr_model

        positive_llrs = llr_matrix[llr_matrix > 0]
        self._llr_max = float(np.sort(positive_llrs)[-min(MAX_DEPTH, len(positive_llrs)):].sum()) if len(positive_llrs) else 1.0
        self._llr_min = float(llr_matrix.min()) * MAX_DEPTH
        self._llr_range = max(self._llr_max - self._llr_min, 1e-6)

    def __call__(self, mutations: List[Mutation]) -> float:
        esm1v_raw = sum(
            self.llr_matrix[pos, AA_TO_INDEX[mut_aa]]
            for pos, _wt, mut_aa in mutations
        )
        esm1v_norm = (esm1v_raw - self._llr_min) / self._llr_range
        esm1v_norm = max(0.0, min(1.0, esm1v_norm))

        if not self.gpr.is_trained:
            return esm1v_norm

        seq = _apply_mutations(self.wt_sequence, mutations)
        emb = embed_sequence(seq)
        gpr_mean, gpr_std = self.gpr.predict(emb)
        gpr_mean = float(gpr_mean[0])

        if self.gpr._y is not None and len(self.gpr._y) > 1:
            y_min, y_max = float(self.gpr._y.min()), float(self.gpr._y.max())
            y_range = max(y_max - y_min, 1e-6)
            gpr_norm = (gpr_mean - y_min) / y_range
            gpr_norm = max(0.0, min(1.0, gpr_norm))
        else:
            gpr_norm = 0.5

        alpha = gpr_alpha(self.gpr.n_samples)
        return alpha * esm1v_norm + (1.0 - alpha) * gpr_norm


# ── MCTS search with Progressive Widening ───────────────────────────────────

def run_mcts(
    wt_sequence: str,
    llr_matrix: np.ndarray,
    candidate_actions: List[Mutation],
    gpr_model: FitnessGPR,
    num_simulations: int = NUM_SIMULATIONS,
    max_depth: int = MAX_DEPTH,
    sequences_to_return: int = SEQUENCES_PER_ROUND,
    previously_proposed: Optional[set] = None,
) -> List[Dict]:
    """Run MCTS with Progressive Widening and return the best diverse set of sequences."""
    value_fn = ValueFunction(wt_sequence, llr_matrix, gpr_model)

    # Sort candidates by LLR ascending so pop() yields the best mutation first
    sorted_candidates = sorted(
        candidate_actions,
        key=lambda m: llr_matrix[m[0], AA_TO_INDEX[m[2]]],
    )

    root = MCTSNode(
        mutations=[],
        candidate_actions_sorted=sorted_candidates,
    )

    for sim in range(num_simulations):
        if (sim + 1) % 200 == 0:
            print(f"  [MCTS] Simulation {sim + 1}/{num_simulations}")

        node = root

        # ── Selection + Expansion (Progressive Widening) ────────────
        while not node.is_terminal:
            if node.should_expand():
                node = node.expand(sorted_candidates)
                break
            elif node.children:
                node = node.best_child_ucb()
            else:
                break

        # ── Evaluation ──────────────────────────────────────────────
        if node.mutations:
            value = value_fn(node.mutations)
        else:
            value = 0.0

        # ── Backpropagation ─────────────────────────────────────────
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # ── Collect results ─────────────────────────────────────────────
    all_nodes: List[MCTSNode] = []
    _collect_nodes(root, all_nodes)

    depth_counts = Counter(n.depth for n in all_nodes if n.mutations and n.visits > 0)

    scored = []
    for n in all_nodes:
        if not n.mutations or n.visits == 0:
            continue
        avg_val = n.total_value / n.visits
        seq = _apply_mutations(wt_sequence, n.mutations)

        esm1v_raw = sum(
            llr_matrix[pos, AA_TO_INDEX[mut_aa]]
            for pos, _wt, mut_aa in n.mutations
        )

        gpr_score = 0.0
        if gpr_model.is_trained:
            emb = embed_sequence(seq)
            gpr_mean, _ = gpr_model.predict(emb)
            gpr_score = float(gpr_mean[0])

        scored.append({
            "sequence": seq,
            "mutations": n.mutations,
            "mutations_str": "+".join(
                f"{wt}{pos+1}{mut}" for pos, wt, mut in n.mutations
            ),
            "esm1v_score": round(esm1v_raw, 4),
            "gpr_score": round(gpr_score, 4),
            "combined_score": round(avg_val, 4),
            "visits": n.visits,
            "depth": n.depth,
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    # Remove sequences already proposed in previous rounds
    if previously_proposed:
        before = len(scored)
        scored = [s for s in scored if s["sequence"] not in previously_proposed]
        n_dup = before - len(scored)
        if n_dup > 0:
            print(f"[MCTS] Filtered out {n_dup} previously proposed sequences.")

    selected = _depth_diverse_select(scored, sequences_to_return, min_hamming=2)

    sel_depths = Counter(s["depth"] for s in selected)
    print(f"[MCTS] Search complete. Selected {len(selected)} sequences from "
          f"{len(scored)} candidates.")
    print(f"[MCTS] Tree depth distribution: "
          + ", ".join(f"depth {d}: {depth_counts.get(d, 0)} nodes"
                      for d in range(1, MAX_DEPTH + 1)))
    print(f"[MCTS] Output depth distribution: "
          + ", ".join(f"{d}-mut: {sel_depths.get(d, 0)}"
                      for d in sorted(sel_depths)))

    return selected


def _collect_nodes(node: MCTSNode, result: List[MCTSNode]):
    result.append(node)
    for child in node.children.values():
        _collect_nodes(child, result)


def _depth_diverse_select(
    candidates: List[Dict],
    k: int,
    min_hamming: int = 2,
) -> List[Dict]:
    """Select *k* sequences with guaranteed depth diversity.

    1. For each depth in DEPTH_QUOTA, greedily pick up to the quota
       (respecting Hamming distance against already-selected sequences).
    2. Fill remaining slots from all leftover candidates by combined_score.
    3. If still short, relax the Hamming constraint for remaining slots.
    """
    if len(candidates) <= k:
        return candidates

    by_depth: Dict[int, List[Dict]] = {}
    for c in candidates:
        by_depth.setdefault(c["depth"], []).append(c)

    selected: List[Dict] = []
    used_seqs: set = set()

    def _try_add(cand: Dict) -> bool:
        seq = cand["sequence"]
        if seq in used_seqs:
            return False
        if all(_hamming(seq, s["sequence"]) >= min_hamming for s in selected):
            selected.append(cand)
            used_seqs.add(seq)
            return True
        return False

    # Phase 1: fill depth quotas
    for depth in sorted(DEPTH_QUOTA.keys()):
        quota = DEPTH_QUOTA[depth]
        pool = by_depth.get(depth, [])
        added = 0
        for cand in pool:
            if added >= quota or len(selected) >= k:
                break
            if _try_add(cand):
                added += 1

    # Phase 2: fill remaining slots from all candidates by combined_score
    if len(selected) < k:
        for cand in candidates:
            if len(selected) >= k:
                break
            _try_add(cand)

    # Phase 3: relax Hamming if still short
    if len(selected) < k:
        for cand in candidates:
            if len(selected) >= k:
                break
            seq = cand["sequence"]
            if seq not in used_seqs:
                selected.append(cand)
                used_seqs.add(seq)

    return selected
