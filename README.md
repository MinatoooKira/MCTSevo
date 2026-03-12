# MCTSevo

Protein directed evolution guided by **Monte Carlo Tree Search (MCTS)**, combining ESM-1v log-likelihood ratios with Gaussian Process Regression (GPR) trained on wet-lab fitness data.

MCTSevo treats the protein mutation landscape as a **search tree** and applies MCTS — the same algorithm behind AlphaGo — to systematically explore combinatorial mutations while balancing exploration and exploitation.

## Why MCTSevo?

Most ML-guided directed evolution tools (e.g., [EVOLVEpro](https://github.com/mat10d/evolvepro)) follow a **greedy paradigm**: embed sequences, train a regression model on experimental data, predict the best single mutations, then combine top hits. This is fast, but fundamentally limited — greedy combination misses **epistatic interactions** where mutations are individually neutral but synergistic together.

MCTSevo takes a different approach:

### vs. EVOLVEpro and similar tools

| | **EVOLVEpro** | **MCTSevo** |
|---|---|---|
| **Search strategy** | Greedy: predict best single mutations, combine top hits | MCTS: systematically explore combinatorial space with UCB |
| **Combinatorial mutations** | Post-hoc combination of individual winners | Native tree search over multi-mutation paths (up to 5 simultaneous) |
| **Epistasis handling** | Limited — assumes additive effects when combining | Evaluates full combinations directly, can discover non-additive synergies |
| **Zero-shot capability** | Requires initial experimental data | Round 0 produces meaningful proposals via ESM-1v LLR alone |
| **PLM usage** | ESM-2 embeddings → Random Forest | ESM-1v (evolutionary prior) + ESM-2 (GPR features) — dual-model architecture |
| **Exploration vs. exploitation** | Relies on model uncertainty | Principled UCB formula with mathematical convergence guarantees |
| **Regression model** | Random Forest | GPR with calibrated uncertainty estimates |
| **Output diversity** | Model-dependent | Guaranteed depth diversity (1-mut through 4-mut per round) + Hamming distance constraints |
| **Deployment** | Requires pre-computed embeddings | Single CLI command, runs end-to-end on a laptop |

### Key advantages

- **Combinatorial from day one.** Each round proposes sequences spanning 1 to 4+ simultaneous mutations, building baseline understanding of individual effects while also exploring deep combinations.

- **No cold-start problem.** ESM-1v masked-marginal scoring provides a strong evolutionary prior before any experiments are run. You get informative proposals from Round 0.

- **Progressive Widening.** Standard MCTS cannot handle the enormous branching factor of protein mutation space (~400+ candidate mutations per node). Progressive Widening adaptively controls the tree width, enabling the search to reach 4–5 mutation depth within 1,000 simulations.

- **Principled exploration.** The UCB formula mathematically balances trying untested mutation paths against refining known promising ones — no ad-hoc acquisition function needed.

- **Dual-model scoring.** ESM-1v provides evolutionary plausibility; GPR learns the specific fitness landscape from your data. The blend automatically shifts from ESM-1v-dominant (early rounds) to GPR-dominant (later rounds) as data accumulates.

- **Runs anywhere.** Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU. No cloud API, no cluster, no pre-computed embeddings. Just `pip install` and go.

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MCTSevo Pipeline                             │
│                                                                     │
│  Wild-type ──→ ESM-1v LLR ──→ Candidate Selection ──→ MCTS Search  │
│  Sequence      Heatmap         (diversified)          (Progressive  │
│                                                        Widening)    │
│                                                           │         │
│                                                           ▼         │
│  Wet-lab    ◄── Fill CSV ◄── 20 Proposed Sequences ◄── Depth-      │
│  Experiments                  (1-mut to 4-mut)         Diverse      │
│       │                                                Selection    │
│       ▼                                                             │
│  ESM-2 Embed ──→ GPR Train ──→ Updated Value Function ──→ Next     │
│                                                           Round     │
└─────────────────────────────────────────────────────────────────────┘
```

1. **ESM-1v masked-marginal scoring** — Every possible single-residue mutation is scored via log-likelihood ratio (LLR), producing a full-sequence heatmap.
2. **Diversified candidate selection** — High-LLR mutations are selected across different positions (window-based + global top-N) to form the MCTS action space.
3. **MCTS with Progressive Widening** — Explores combinatorial mutations using UCB:

$$\text{UCB} = \frac{\text{Value}}{\text{Visits}} + C \cdot \sqrt{\frac{2 \ln(\text{Parent\_Visits})}{\text{Visits}}}$$

4. **GPR fitness prediction** — ESM-2 embeddings are mapped to wet-lab fitness via Gaussian Process Regression. The GPR prediction is blended with ESM-1v as a composite node value.
5. **Depth-diverse output** — Each round outputs 20 sequences with guaranteed representation across mutation depths (single, double, triple, quadruple), plus automatic deduplication against all previous rounds.
6. **Active learning loop** — Users provide experimental fitness data; GPR is retrained and MCTS runs again with updated value estimates.

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

### 2. Run Round 0 (zero-shot, no experimental data needed)

```bash
python main.py run --round 0
```

Outputs 20 proposed mutant sequences spanning 1–4 mutations to `output/round_0/proposed_sequences.csv`.

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
- Runs MCTS with updated value function (ESM-1v + GPR)
- Guarantees depth-diverse output (1-mut through 4-mut)
- Automatically deduplicates against all previously proposed sequences

### 5. Check status

```bash
python main.py status
```

Displays the top-10 leaderboard ranked by wet-lab fitness.

### Optional: deeper search

```bash
python main.py run --round 3 --simulations 2000
```

More simulations allow the MCTS to explore deeper and wider mutation combinations.

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
| `DEPTH_QUOTA` | {1:3, 2:4, 3:4, 4:4} | Minimum sequences per mutation depth |
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

## Citation

If you find MCTSevo useful in your research, please cite this repository:

```
@software{mctsevo2026,
  title={MCTSevo: Protein Directed Evolution via Monte Carlo Tree Search},
  url={https://github.com/MinatoooKira/MCTSevo},
  year={2026}
}
```

## License

MIT
