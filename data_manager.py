"""Round directory management, CSV I/O, and leaderboard tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TOP_K_LEADERBOARD


def _output_root() -> Path:
    p = Path(OUTPUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def round_dir(round_num: int) -> Path:
    d = _output_root() / f"round_{round_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Wild-type persistence ───────────────────────────────────────────────────

def save_wt_info(wt_sequence: str, wt_name: str) -> None:
    info = {"wt_sequence": wt_sequence, "wt_name": wt_name}
    path = _output_root() / "wt_info.json"
    path.write_text(json.dumps(info, indent=2))
    print(f"[Data] Wild-type info saved to {path}")


def load_wt_info() -> Dict[str, str]:
    path = _output_root() / "wt_info.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Wild-type info not found at {path}. Run 'init' first."
        )
    return json.loads(path.read_text())


# ── LLR matrix cache ───────────────────────────────────────────────────────

def save_llr_matrix(llr_matrix: np.ndarray) -> None:
    path = _output_root() / "llr_matrix.npy"
    np.save(path, llr_matrix)
    print(f"[Data] LLR matrix saved to {path}")


def load_llr_matrix() -> np.ndarray:
    path = _output_root() / "llr_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(f"LLR matrix not found at {path}. Run 'init' first.")
    return np.load(path)


# ── Candidate actions cache ─────────────────────────────────────────────────

def save_candidates(candidates: List[Tuple[int, str, str]]) -> None:
    path = _output_root() / "candidates.json"
    path.write_text(json.dumps(candidates, indent=2))


def load_candidates() -> List[Tuple[int, str, str]]:
    path = _output_root() / "candidates.json"
    if not path.exists():
        raise FileNotFoundError(f"Candidates not found at {path}. Run 'init' first.")
    raw = json.loads(path.read_text())
    return [tuple(c) for c in raw]


# ── Proposed sequences ──────────────────────────────────────────────────────

def save_proposed_sequences(
    round_num: int, sequences: List[Dict], wt_sequence: str,
) -> Path:
    rd = round_dir(round_num)
    rows = []
    for i, s in enumerate(sequences):
        rows.append({
            "id": i + 1,
            "sequence": s["sequence"],
            "mutations": s["mutations_str"],
            "esm1v_score": s["esm1v_score"],
            "gpr_score": s["gpr_score"],
            "combined_score": s["combined_score"],
            "visits": s["visits"],
        })
    df = pd.DataFrame(rows)
    path = rd / "proposed_sequences.csv"
    df.to_csv(path, index=False)
    print(f"[Data] Proposed sequences saved to {path}")

    # Also create a wet-lab template for the user to fill in
    template_path = rd / "wet_lab_results.csv"
    if not template_path.exists():
        wt_fitness = _load_wt_fitness()
        wt_row = pd.DataFrame([{
            "id": 0,
            "sequence": wt_sequence,
            "mutations": "WT",
            "fitness": wt_fitness if wt_fitness is not None else "",
        }])
        tpl = df[["id", "sequence", "mutations"]].copy()
        tpl["fitness"] = ""
        tpl = pd.concat([wt_row, tpl], ignore_index=True)
        tpl.to_csv(template_path, index=False)
        print(f"[Data] Wet-lab template created at {template_path}")
        if wt_fitness is not None:
            print(f"       → WT fitness auto-filled from previous round ({wt_fitness})")
        else:
            print(f"       → Please fill in the 'fitness' column (including WT at row 0).")

    return path


def _load_wt_fitness() -> Optional[float]:
    """Read WT fitness from wt_info.json if it has been recorded."""
    path = _output_root() / "wt_info.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    return info.get("wt_fitness")


def save_wt_fitness(fitness: float) -> None:
    """Persist the WT fitness value into wt_info.json."""
    path = _output_root() / "wt_info.json"
    info = json.loads(path.read_text())
    if info.get("wt_fitness") is None:
        info["wt_fitness"] = fitness
        path.write_text(json.dumps(info, indent=2))
        print(f"[Data] WT fitness ({fitness}) saved for future rounds.")


# ── Wet-lab data ────────────────────────────────────────────────────────────

def load_wet_lab_results(round_num: int) -> Optional[pd.DataFrame]:
    """Load wet-lab results for a given round. Returns None if not filled."""
    path = round_dir(round_num) / "wet_lab_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "fitness" not in df.columns:
        return None
    filled = df.dropna(subset=["fitness"])
    filled = filled[filled["fitness"] != ""]
    if filled.empty:
        return None
    filled["fitness"] = pd.to_numeric(filled["fitness"], errors="coerce")
    filled = filled.dropna(subset=["fitness"])
    return filled if not filled.empty else None


def load_all_wet_lab_data() -> Optional[pd.DataFrame]:
    """Load and concatenate wet-lab data from all rounds.

    Deduplicates by sequence so that the WT (present in every round's
    template) is counted only once, keeping the earliest occurrence.
    Also persists the WT fitness the first time it is seen.
    """
    root = _output_root()
    all_dfs = []
    r = 0
    while True:
        rd = root / f"round_{r}"
        if not rd.exists():
            break
        df = load_wet_lab_results(r)
        if df is not None:
            df["round"] = r
            all_dfs.append(df)
        r += 1
    if not all_dfs:
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence"], keep="first")

    # Persist WT fitness if we haven't already
    wt_info = load_wt_info()
    wt_rows = combined[combined["mutations"] == "WT"]
    if not wt_rows.empty and _load_wt_fitness() is None:
        save_wt_fitness(float(wt_rows.iloc[0]["fitness"]))

    agg_path = root / "all_wet_lab_data.csv"
    combined.to_csv(agg_path, index=False)
    return combined


# ── Leaderboard ─────────────────────────────────────────────────────────────

def update_leaderboard() -> Optional[pd.DataFrame]:
    """Rebuild the top-K leaderboard from all wet-lab data."""
    all_data = load_all_wet_lab_data()
    if all_data is None:
        return None

    # Keep best fitness per unique sequence
    best = (
        all_data.sort_values("fitness", ascending=False)
        .drop_duplicates(subset=["sequence"], keep="first")
        .head(TOP_K_LEADERBOARD)
        .reset_index(drop=True)
    )
    path = _output_root() / "top10_leaderboard.csv"
    best.to_csv(path, index=False)
    print(f"[Leaderboard] Top {len(best)} sequences saved to {path}")
    return best


def load_all_proposed_sequences() -> set:
    """Return a set of all sequence strings proposed in previous rounds."""
    root = _output_root()
    seen = set()
    r = 0
    while True:
        path = root / f"round_{r}" / "proposed_sequences.csv"
        if not path.exists():
            break
        try:
            df = pd.read_csv(path)
            if "sequence" in df.columns:
                seen.update(df["sequence"].tolist())
        except Exception:
            pass
        r += 1
    return seen


def get_latest_round() -> int:
    """Return the highest round number that has a directory, or -1."""
    root = _output_root()
    r = -1
    while (root / f"round_{r + 1}").exists():
        r += 1
    return r
