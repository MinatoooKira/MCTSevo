import torch

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# ── Device ──────────────────────────────────────────────────────────────────
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _detect_device()

# ── ESM models ──────────────────────────────────────────────────────────────
ESM1V_MODEL = "esm1v_t33_650M_UR90S_1"
ESM2_MODEL = "esm2_t33_650M_UR50D"
ESM2_EMBED_DIM = 1280
ESM1V_REPR_LAYER = 33
ESM2_REPR_LAYER = 33

# ── MCTS ────────────────────────────────────────────────────────────────────
MAX_DEPTH = 5
NUM_SIMULATIONS = 1000
UCB_C = 1.414
PW_K = 1.0
PW_ALPHA = 0.5

# ── Round / output ──────────────────────────────────────────────────────────
SEQUENCES_PER_ROUND = 20
TOP_K_LEADERBOARD = 10
OUTPUT_DIR = "output"

# Minimum sequences per mutation depth to guarantee diversity.
# Remaining slots (SEQUENCES_PER_ROUND minus sum) are filled by best overall.
# If a depth can't fill its quota (e.g. deduplication), slots redistribute.
DEPTH_QUOTA = {1: 3, 2: 4, 3: 4, 4: 4}

# ── Candidate selection ─────────────────────────────────────────────────────
CANDIDATE_PER_POSITION = 3
DIVERSITY_WINDOW = 10
GLOBAL_TOP_N = 30

# ── GPR weight schedule ────────────────────────────────────────────────────
GPR_ALPHA_MIN = 0.3
GPR_ALPHA_DECAY = 0.07

def gpr_alpha(n_samples: int) -> float:
    """Weight for ESM-1v score vs GPR. Starts at 1.0, decays toward GPR_ALPHA_MIN."""
    return max(GPR_ALPHA_MIN, 1.0 - GPR_ALPHA_DECAY * n_samples)

# ── Batching (for ESM-1v masked-marginal forward passes) ───────────────────
ESM1V_BATCH_SIZE = 8
