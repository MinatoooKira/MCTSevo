"""Generate publication-quality result figures from MCTSevo round data."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT = Path("output")
FIG_DIR = OUTPUT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    1: "#4C72B0",
    2: "#55A868",
    3: "#C44E52",
    4: "#8172B2",
    5: "#CCB974",
}


def load_all_rounds():
    rows = []
    for r in range(100):
        prop_path = OUTPUT / f"round_{r}" / "proposed_sequences.csv"
        wet_path = OUTPUT / f"round_{r}" / "wet_lab_results.csv"
        if not prop_path.exists():
            break
        prop = pd.read_csv(prop_path)
        if not wet_path.exists():
            continue
        wet = pd.read_csv(wet_path)
        wet = wet.dropna(subset=["fitness"])
        wet["fitness"] = pd.to_numeric(wet["fitness"], errors="coerce")
        wet = wet.dropna(subset=["fitness"])
        if len(wet) < 2:
            continue
        merged = prop.merge(wet[["sequence", "fitness"]], on="sequence", how="inner")
        merged["round"] = r
        merged["n_mutations"] = merged["mutations"].apply(lambda m: m.count("+") + 1)
        rows.append(merged)
    if not rows:
        raise RuntimeError("No round data found")
    return pd.concat(rows, ignore_index=True)


def plot_fitness_over_rounds(df):
    """Fig 1: Top / Mean / Median fitness per round — shows active learning effect."""
    stats = df.groupby("round")["fitness"].agg(["max", "mean", "median"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(stats["round"], stats["max"], "o-", color="#C44E52", linewidth=2,
            markersize=7, label="Best fitness", zorder=3)
    ax.plot(stats["round"], stats["mean"], "s-", color="#4C72B0", linewidth=2,
            markersize=6, label="Mean fitness", zorder=3)
    ax.fill_between(stats["round"], stats["mean"],
                    df.groupby("round")["fitness"].min().values,
                    alpha=0.12, color="#4C72B0")
    ax.fill_between(stats["round"], stats["mean"],
                    stats["max"], alpha=0.12, color="#C44E52")

    ax.set_xlabel("Round")
    ax.set_ylabel("Predicted Fitness")
    ax.set_title("Fitness Improvement Across Rounds")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    path = FIG_DIR / "fitness_over_rounds.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Fig] {path}")


def plot_fitness_distribution(df):
    """Fig 2: Box plot of fitness per round — shows distribution tightening."""
    rounds = sorted(df["round"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data_per_round = [df[df["round"] == r]["fitness"].values for r in rounds]

    bp = ax.boxplot(data_per_round, positions=rounds, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#4C72B0" if rounds[i] <= 5 else "#C44E52")
        patch.set_alpha(0.6)

    for r_idx, r in enumerate(rounds):
        pts = df[df["round"] == r]["fitness"].values
        jitter = np.random.normal(0, 0.08, size=len(pts))
        ax.scatter(np.full_like(pts, r) + jitter, pts, alpha=0.4, s=18,
                   color="#333333", zorder=3)

    ax.set_xlabel("Round")
    ax.set_ylabel("Predicted Fitness")
    ax.set_title("Fitness Distribution per Round")
    # Explicitly show all rounds on the x-axis (including round 0)
    ax.set_xticks(rounds)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", alpha=0.6, label="Standard MCTS (R0–R5)"),
        Patch(facecolor="#C44E52", alpha=0.6, label="Progressive Widening (R6+)"),
    ], framealpha=0.9)

    path = FIG_DIR / "fitness_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Fig] {path}")


def plot_esm1v_vs_fitness(df):
    """Fig 3: ESM-1v score vs predicted fitness — shows PLM predictive power."""
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for depth in sorted(df["n_mutations"].unique()):
        sub = df[df["n_mutations"] == depth]
        ax.scatter(sub["esm1v_score"], sub["fitness"], s=35, alpha=0.65,
                   color=COLORS.get(depth, "#999999"),
                   label=f"{depth}-mutation", edgecolors="white", linewidths=0.3)

    from scipy import stats as sp_stats
    mask = np.isfinite(df["esm1v_score"]) & np.isfinite(df["fitness"])
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(
        df.loc[mask, "esm1v_score"], df.loc[mask, "fitness"]
    )
    x_range = np.linspace(df["esm1v_score"].min(), df["esm1v_score"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, "--", color="#666666", linewidth=1.5,
            label=f"R² = {r_val**2:.3f}")

    ax.set_xlabel("ESM-1v Additive LLR Score")
    ax.set_ylabel("Predicted Fitness")
    ax.set_title("ESM-1v Score vs. Predicted Fitness")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)

    path = FIG_DIR / "esm1v_vs_fitness.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Fig] {path}")


def plot_depth_distribution(df):
    """Fig 4: Stacked bar of mutation depths per round."""
    rounds = sorted(df["round"].unique())
    depths = sorted(df["n_mutations"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(rounds))

    for d in depths:
        counts = [len(df[(df["round"] == r) & (df["n_mutations"] == d)]) for r in rounds]
        ax.bar(rounds, counts, bottom=bottom, width=0.65,
               color=COLORS.get(d, "#999999"), label=f"{d}-mutation",
               edgecolor="white", linewidth=0.5)
        bottom += np.array(counts)

    ax.set_xlabel("Round")
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Mutation Depth Distribution Across Rounds")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.9, title="Depth")
    ax.grid(axis="y", alpha=0.3)

    path = FIG_DIR / "depth_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Fig] {path}")


def plot_cumulative_best(df):
    """Fig 5: Cumulative best fitness — shows the 'discovery curve'."""
    rounds = sorted(df["round"].unique())
    cum_best = []
    best_so_far = -np.inf
    best_mut = []
    for r in rounds:
        sub = df[df["round"] == r]
        r_best_idx = sub["fitness"].idxmax()
        r_best = sub.loc[r_best_idx, "fitness"]
        if r_best > best_so_far:
            best_so_far = r_best
            best_mut.append(sub.loc[r_best_idx, "mutations"])
        else:
            best_mut.append(None)
        cum_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(rounds, cum_best, where="mid", color="#C44E52", linewidth=2.5, zorder=3)
    ax.scatter(rounds, cum_best, color="#C44E52", s=50, zorder=4, edgecolors="white")

    for i, (r, cb, m) in enumerate(zip(rounds, cum_best, best_mut)):
        if m is not None:
            ax.annotate(m, (r, cb), textcoords="offset points",
                        xytext=(8, 8 if i % 2 == 0 else -14),
                        fontsize=7.5, color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Best Predicted Fitness")
    ax.set_title("Discovery Curve: Best Predicted Fitness Over Rounds")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)

    path = FIG_DIR / "cumulative_best.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Fig] {path}")


def plot_mean_fitness_by_depth(df):
    """Fig 6: Mean fitness grouped by mutation depth — shows depth-fitness relationship."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    depths = sorted(df["n_mutations"].unique())
    means = [df[df["n_mutations"] == d]["fitness"].mean() for d in depths]
    stds = [df[df["n_mutations"] == d]["fitness"].std() for d in depths]
    counts = [len(df[df["n_mutations"] == d]) for d in depths]

    bars = ax.bar(depths, means, yerr=stds, width=0.6, capsize=5,
                  color=[COLORS.get(d, "#999999") for d in depths],
                  edgecolor="white", linewidth=0.8, alpha=0.85)

    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"n={n}", ha="center", va="bottom", fontsize=9, color="#555555")

    ax.set_xlabel("Number of Mutations")
    ax.set_ylabel("Mean Predicted Fitness")
    ax.set_title("Fitness by Mutation Depth")
    ax.set_xticks(depths)
    ax.grid(axis="y", alpha=0.3)

    path = FIG_DIR / "fitness_by_depth.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Fig] {path}")


if __name__ == "__main__":
    df = load_all_rounds()
    print(f"Loaded {len(df)} data points across {df['round'].nunique()} rounds\n")

    plot_fitness_over_rounds(df)
    plot_fitness_distribution(df)
    plot_esm1v_vs_fitness(df)
    plot_depth_distribution(df)
    plot_cumulative_best(df)
    plot_mean_fitness_by_depth(df)

    print(f"\nAll figures saved to {FIG_DIR}/")
