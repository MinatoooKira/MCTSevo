# MCTSevo

Protein directed evolution guided by **Monte Carlo Tree Search (MCTS)**, combining ESM-1v log-likelihood ratios with Gaussian Process Regression (GPR) trained on wet-lab fitness data.

## Overview

MCTSevo is an active-learning tool for protein engineering. It iteratively proposes mutant sequences, learns from experimental feedback, and refines predictions — bridging computational scoring with real-world data.

### How it works

1. **ESM-1v masked-marginal scoring** — Every possible single-residue mutation is scored via log-likelihood ratio (LLR), producing a full-sequence heatmap.
2. **Diversified candidate selection** — High-LLR mutations are selected across different positions (window-based + global top-N) to form the MCTS action space.
3. **MCTS with Progressive Widening** — Explores combinatorial mutations (up to 5 simultaneous) using UCB:

$$\text{UCB} = \frac{\text{Value}}{\text{Visits}} + C \cdot \sqrt{\frac{2 \ln(\text{Parent\_Visits})}{\text{Visits}}}$$

Progressive Widening controls the branching factor so the tree can reach deep multi-mutation combinations (3–5 mutations) even with a moderate simulation budget.

4. **GPR fitness prediction** — ESM-2 embeddings are mapped to wet-lab fitness via Gaussian Process Regression. The GPR mean is blended with the ESM-1v score as a composite node value.
5. **Active learning loop** — Each round outputs 20 novel sequences for experimental testing. Users feed back fitness data, GPR is retrained, and MCTS runs again with updated value estimates.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (with MPS / CUDA / CPU support)
- ~4 GB memory for ESM model weights (downloaded automatically on first run)

## Installation

```bash
git clone https://github.com/MinatoooKira/MCTSevo.git
cd MCTSevo
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

> **Device auto-detection**: Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU — no manual configuration needed.

## Quick Start

### 1. Initialize with your wild-type sequence

```bash
python main.py init --wt-sequence "MKTLLLTL..." --wt-name "MyProtein"
```

This computes the ESM-1v LLR matrix, selects candidate mutations, and generates a heatmap in `output/round_0/`.

### 2. Run Round 0

```bash
python main.py run --round 0
```

Outputs 20 proposed mutant sequences to `output/round_0/proposed_sequences.csv`.

### 3. Provide wet-lab data

Open `output/round_0/wet_lab_results.csv` and fill in the `fitness` column with your experimental measurements (including the wild-type at row 0).

### 4. Run subsequent rounds

```bash
python main.py run --round 1
python main.py run --round 2
# ...
```

Each round:
- Trains GPR on all accumulated wet-lab data
- Runs MCTS with updated value function
- Automatically deduplicates against all previously proposed sequences

### 5. Check status

```bash
python main.py status
```

Displays the top-10 leaderboard ranked by wet-lab fitness.

### Optional: increase simulations

```bash
python main.py run --round 3 --simulations 2000
```

More simulations allow the MCTS to explore deeper mutation combinations.

## Output Structure

```
output/
├── wt_info.json              # Wild-type sequence and metadata
├── llr_matrix.npy            # Cached ESM-1v LLR matrix
├── candidates.json           # Selected candidate mutations
├── all_wet_lab_data.csv      # Accumulated experimental data
├── top10_leaderboard.csv     # Best sequences by fitness
├── round_0/
│   ├── llr_heatmap.png       # LLR heatmap visualization
│   ├── proposed_sequences.csv
│   └── wet_lab_results.csv   # ← fill in fitness here
├── round_1/
│   └── ...
```

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_DEPTH` | 5 | Maximum simultaneous mutations per sequence |
| `NUM_SIMULATIONS` | 1000 | MCTS iterations per round |
| `UCB_C` | 1.414 | Exploration constant (√2) |
| `PW_K` | 1.0 | Progressive Widening width coefficient — lower = deeper search |
| `PW_ALPHA` | 0.5 | Progressive Widening growth exponent |
| `SEQUENCES_PER_ROUND` | 20 | Sequences proposed each round |
| `CANDIDATE_PER_POSITION` | 3 | Top mutations kept per position |
| `DIVERSITY_WINDOW` | 10 | Residue window for positional diversity |
| `GLOBAL_TOP_N` | 30 | Global top-N mutations always included |
| `GPR_ALPHA_MIN` | 0.3 | Minimum weight for ESM-1v vs GPR |
| `GPR_ALPHA_DECAY` | 0.07 | How fast GPR takes over from ESM-1v |
| `ESM1V_BATCH_SIZE` | 8 | Batch size for LLR computation |

### Tuning Progressive Widening

The `PW_K` parameter controls the trade-off between breadth and depth:

| PW_K | Typical max depth (1000 sims) | Notes |
|------|-------------------------------|-------|
| 1.5  | 3 mutations | Wider exploration, more single/double mutants |
| 1.0  | 4 mutations | Balanced (default) |
| 0.5  | 5 mutations | Deepest search, narrower breadth |

## Architecture

```
main.py                 # CLI entry point (init / run / status)
config.py               # All tunable parameters
esm_models.py           # ESM-1v LLR scoring + ESM-2 embedding
candidate_selector.py   # Diversified mutation candidate selection
mcts_engine.py          # MCTS with Progressive Widening + UCB
gpr_model.py            # Gaussian Process Regression wrapper
data_manager.py         # Round management, CSV I/O, leaderboard
visualization.py        # LLR heatmap generation
```

## License

MIT
