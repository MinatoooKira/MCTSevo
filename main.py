#!/usr/bin/env python3
"""MCTSevo-Light — Protein directed evolution via Monte Carlo Tree Search.

Usage
-----
  # 1. Initialise with a wild-type sequence
  python main.py init --wt-sequence "MKTL..." --wt-name "MyProtein"

  # 2. Run round 0 (ESM-1v only, GPR untrained)
  python main.py run --round 0

  # 3. Fill in output/round_0/wet_lab_results.csv with experimental fitness

  # 4. Run round 1 (ESM-1v + GPR)
  python main.py run --round 1

  # Show current leaderboard
  python main.py status
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from config import DEVICE, NUM_SIMULATIONS, SEQUENCES_PER_ROUND
import data_manager as dm
from esm_models import compute_llr_matrix, embed_sequence, embed_sequences_batch
from candidate_selector import select_candidates
from gpr_model import FitnessGPR
from mcts_engine import run_mcts
from visualization import plot_llr_heatmap


def cmd_init(args):
    """Initialise the project: store WT, compute LLR heatmap, select candidates."""
    wt_seq = args.wt_sequence.strip().upper()
    wt_name = args.wt_name or "Protein"

    # Validate
    from config import AMINO_ACIDS
    invalid = [c for c in wt_seq if c not in AMINO_ACIDS]
    if invalid:
        print(f"Error: invalid amino acid(s) in sequence: {set(invalid)}")
        sys.exit(1)

    print(f"[Init] Wild-type: {wt_name} ({len(wt_seq)} residues)")
    print(f"[Init] Device: {DEVICE}")
    dm.save_wt_info(wt_seq, wt_name)

    # Compute LLR heatmap
    print("[Init] Computing ESM-1v LLR matrix (masked marginal) …")
    t0 = time.time()
    llr_matrix = compute_llr_matrix(wt_seq)
    print(f"[Init] LLR matrix computed in {time.time() - t0:.1f}s")
    dm.save_llr_matrix(llr_matrix)

    # Select candidate mutations
    candidates = select_candidates(llr_matrix, wt_seq)
    dm.save_candidates(candidates)

    # Plot heatmap
    rd = dm.round_dir(0)
    plot_llr_heatmap(
        llr_matrix, wt_seq,
        save_path=rd / "llr_heatmap.png",
        candidates=candidates,
        title=f"{wt_name} — ESM-1v LLR Heatmap",
    )

    print("\n[Init] Initialisation complete.")
    print(f"       Heatmap saved to {rd / 'llr_heatmap.png'}")
    print(f"       Run 'python main.py run --round 0' to start the first round.")


def cmd_run(args):
    """Run a single round of MCTS-guided directed evolution."""
    round_num = args.round

    # Load essentials
    wt_info = dm.load_wt_info()
    wt_seq = wt_info["wt_sequence"]
    wt_name = wt_info["wt_name"]
    llr_matrix = dm.load_llr_matrix()
    candidates = dm.load_candidates()

    print(f"\n{'='*60}")
    print(f"  MCTSevo-Light — Round {round_num}")
    print(f"  Protein: {wt_name} ({len(wt_seq)} residues)")
    print(f"  Device:  {DEVICE}")
    print(f"{'='*60}\n")

    # ── Build / update GPR ──────────────────────────────────────────
    gpr = FitnessGPR()

    if round_num > 0:
        # Check that previous round's wet-lab data exists
        prev_data = dm.load_wet_lab_results(round_num - 1)
        if prev_data is None:
            print(f"Error: No wet-lab data found for round {round_num - 1}.")
            print(f"       Please fill in output/round_{round_num - 1}/wet_lab_results.csv")
            sys.exit(1)

        # Load all accumulated wet-lab data
        all_data = dm.load_all_wet_lab_data()
        if all_data is not None and len(all_data) >= 2:
            print(f"[GPR] Training on {len(all_data)} wet-lab samples …")
            sequences = all_data["sequence"].tolist()
            fitness = all_data["fitness"].values.astype(np.float64)

            embeddings = embed_sequences_batch(sequences)
            X = np.vstack(embeddings)
            gpr.train(X, fitness)
        else:
            print("[GPR] Fewer than 2 wet-lab samples — GPR not trained this round.")

    # ── Run MCTS ────────────────────────────────────────────────────
    num_sims = args.simulations or NUM_SIMULATIONS
    previously_proposed = dm.load_all_proposed_sequences() if round_num > 0 else None
    if previously_proposed:
        print(f"[Dedup] {len(previously_proposed)} sequences from previous rounds will be excluded.")
    print(f"[MCTS] Running {num_sims} simulations …")
    t0 = time.time()

    results = run_mcts(
        wt_sequence=wt_seq,
        llr_matrix=llr_matrix,
        candidate_actions=candidates,
        gpr_model=gpr,
        num_simulations=num_sims,
        sequences_to_return=SEQUENCES_PER_ROUND,
        previously_proposed=previously_proposed,
    )
    elapsed = time.time() - t0
    print(f"[MCTS] Done in {elapsed:.1f}s")

    # ── Save results ────────────────────────────────────────────────
    dm.save_proposed_sequences(round_num, results, wt_sequence=wt_seq)

    # Heatmap (reuse cached LLR matrix)
    rd = dm.round_dir(round_num)
    plot_llr_heatmap(
        llr_matrix, wt_seq,
        save_path=rd / "llr_heatmap.png",
        candidates=candidates,
        title=f"{wt_name} — Round {round_num} LLR Heatmap",
    )

    # Update leaderboard
    leaderboard = dm.update_leaderboard()

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Round {round_num} Results")
    print(f"{'─'*60}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:2d}. {r['mutations_str']:<40s}  "
              f"ESM1v={r['esm1v_score']:+.3f}  "
              f"GPR={r['gpr_score']:+.3f}  "
              f"combined={r['combined_score']:.3f}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more (see CSV)")

    if leaderboard is not None and not leaderboard.empty:
        print(f"\n{'─'*60}")
        print(f"  Top-{len(leaderboard)} Leaderboard (by wet-lab fitness)")
        print(f"{'─'*60}")
        for _, row in leaderboard.iterrows():
            print(f"  {row.get('mutations', 'WT'):<40s}  fitness={row['fitness']:.4f}")

    print(f"\n[Next] Fill in output/round_{round_num}/wet_lab_results.csv")
    print(f"       then run: python main.py run --round {round_num + 1}")


def cmd_status(_args):
    """Show the current leaderboard and project status."""
    try:
        wt_info = dm.load_wt_info()
    except FileNotFoundError:
        print("No project initialised. Run 'python main.py init' first.")
        sys.exit(1)

    print(f"Protein: {wt_info['wt_name']} ({len(wt_info['wt_sequence'])} residues)")

    latest = dm.get_latest_round()
    print(f"Latest round directory: {latest}")

    all_data = dm.load_all_wet_lab_data()
    if all_data is not None:
        print(f"Total wet-lab data points: {len(all_data)}")
    else:
        print("No wet-lab data yet.")

    leaderboard = dm.update_leaderboard()
    if leaderboard is not None and not leaderboard.empty:
        print(f"\nTop-{len(leaderboard)} Leaderboard:")
        print(leaderboard[["mutations", "fitness", "round"]].to_string(index=False))
    else:
        print("\nNo leaderboard data yet.")


def main():
    parser = argparse.ArgumentParser(
        description="MCTSevo-Light: Protein directed evolution via MCTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ────────────────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Initialise with a wild-type sequence")
    p_init.add_argument("--wt-sequence", required=True, help="Wild-type amino acid sequence")
    p_init.add_argument("--wt-name", default="Protein", help="Protein name (for labels)")

    # ── run ─────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run a round of MCTS evolution")
    p_run.add_argument("--round", type=int, required=True, help="Round number (0, 1, 2, …)")
    p_run.add_argument("--simulations", type=int, default=None,
                       help=f"Number of MCTS simulations (default: {NUM_SIMULATIONS})")

    # ── status ──────────────────────────────────────────────────────
    sub.add_parser("status", help="Show project status and leaderboard")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"init": cmd_init, "run": cmd_run, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
