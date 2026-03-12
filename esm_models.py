"""Wrappers around ESM-1v (LLR scoring) and ESM-2 (sequence embedding)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import esm

from config import (
    AMINO_ACIDS,
    AA_TO_INDEX,
    DEVICE,
    ESM1V_BATCH_SIZE,
    ESM1V_REPR_LAYER,
    ESM2_REPR_LAYER,
)


# ── Singleton model holders ─────────────────────────────────────────────────

_esm1v_model = None
_esm1v_alphabet = None
_esm1v_batch_converter = None

_esm2_model = None
_esm2_alphabet = None
_esm2_batch_converter = None

_embedding_cache: Dict[str, np.ndarray] = {}


def _get_device():
    """Return a torch.device, preferring MPS/CUDA when available."""
    return torch.device(DEVICE)


# ── ESM-1v ──────────────────────────────────────────────────────────────────

def _load_esm1v():
    global _esm1v_model, _esm1v_alphabet, _esm1v_batch_converter
    if _esm1v_model is not None:
        return
    print("[ESM-1v] Loading model … (this may take a minute the first time)")
    _esm1v_model, _esm1v_alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    _esm1v_batch_converter = _esm1v_alphabet.get_batch_converter()
    device = _get_device()
    _esm1v_model = _esm1v_model.to(device)
    _esm1v_model.eval()
    print(f"[ESM-1v] Model loaded on {device}")


def compute_llr_matrix(wt_sequence: str) -> np.ndarray:
    """Compute the masked-marginal LLR matrix for *wt_sequence*.

    Returns
    -------
    llr : np.ndarray, shape (L, 20)
        ``llr[i, j]`` is the log-likelihood ratio for mutating position *i*
        to amino acid ``AMINO_ACIDS[j]`` relative to the wild-type residue.
        The wild-type entry at each position is 0 by construction.
    """
    _load_esm1v()
    device = _get_device()
    L = len(wt_sequence)
    llr = np.zeros((L, 20), dtype=np.float32)

    positions = list(range(L))
    for batch_start in range(0, L, ESM1V_BATCH_SIZE):
        batch_positions = positions[batch_start : batch_start + ESM1V_BATCH_SIZE]
        data: List[Tuple[str, str]] = []
        for pos in batch_positions:
            seq_list = list(wt_sequence)
            seq_list[pos] = "<mask>"
            masked_seq = "".join(seq_list)
            data.append((f"pos_{pos}", masked_seq))

        _, _, batch_tokens = _esm1v_batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = _esm1v_model(batch_tokens, repr_layers=[ESM1V_REPR_LAYER])
        logits = results["logits"]  # (B, seq_len+2, vocab)

        for idx, pos in enumerate(batch_positions):
            # +1 because of the leading <cls> token
            token_pos = pos + 1
            log_probs = torch.log_softmax(logits[idx, token_pos], dim=-1)

            wt_aa = wt_sequence[pos]
            wt_tok = _esm1v_alphabet.get_idx(wt_aa)
            wt_logp = log_probs[wt_tok].item()

            for j, aa in enumerate(AMINO_ACIDS):
                tok = _esm1v_alphabet.get_idx(aa)
                llr[pos, j] = log_probs[tok].item() - wt_logp

    return llr


# ── ESM-2 ───────────────────────────────────────────────────────────────────

def _load_esm2():
    global _esm2_model, _esm2_alphabet, _esm2_batch_converter
    if _esm2_model is not None:
        return
    print("[ESM-2] Loading model …")
    _esm2_model, _esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    _esm2_batch_converter = _esm2_alphabet.get_batch_converter()
    device = _get_device()
    _esm2_model = _esm2_model.to(device)
    _esm2_model.eval()
    print(f"[ESM-2] Model loaded on {device}")


def _seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def embed_sequence(sequence: str) -> np.ndarray:
    """Return a 1-D mean-pooled ESM-2 embedding (shape ``(embed_dim,)``)."""
    key = _seq_hash(sequence)
    if key in _embedding_cache:
        return _embedding_cache[key]

    _load_esm2()
    device = _get_device()

    data = [("seq", sequence)]
    _, _, batch_tokens = _esm2_batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = _esm2_model(batch_tokens, repr_layers=[ESM2_REPR_LAYER])
    reps = results["representations"][ESM2_REPR_LAYER]  # (1, L+2, D)
    # Strip <cls> and <eos> tokens, then mean-pool
    embedding = reps[0, 1:-1, :].mean(dim=0).cpu().numpy()
    _embedding_cache[key] = embedding
    return embedding


def embed_sequences_batch(sequences: List[str], batch_size: int = 4) -> List[np.ndarray]:
    """Embed multiple sequences, using cache where possible."""
    results: List[Optional[np.ndarray]] = [None] * len(sequences)
    to_compute: List[Tuple[int, str]] = []

    for i, seq in enumerate(sequences):
        key = _seq_hash(seq)
        if key in _embedding_cache:
            results[i] = _embedding_cache[key]
        else:
            to_compute.append((i, seq))

    if not to_compute:
        return results  # type: ignore[return-value]

    _load_esm2()
    device = _get_device()

    for b_start in range(0, len(to_compute), batch_size):
        batch = to_compute[b_start : b_start + batch_size]
        data = [(f"seq_{i}", seq) for i, seq in batch]
        _, _, batch_tokens = _esm2_batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            out = _esm2_model(batch_tokens, repr_layers=[ESM2_REPR_LAYER])
        reps = out["representations"][ESM2_REPR_LAYER]

        for idx_in_batch, (orig_idx, seq) in enumerate(batch):
            seq_len = len(seq)
            emb = reps[idx_in_batch, 1 : 1 + seq_len, :].mean(dim=0).cpu().numpy()
            _embedding_cache[_seq_hash(seq)] = emb
            results[orig_idx] = emb

    return results  # type: ignore[return-value]


def clear_embedding_cache():
    _embedding_cache.clear()
