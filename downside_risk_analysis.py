#!/usr/bin/env python3
"""
Downside Failure Probability Analysis

Computes and reports a scalar metric quantifying downside-risk dominance of
Weighted Harmonic relative to the SMART family.

Metric Definition:
    P(Harmonic < max(SMART, Relaxed SMART) - τ)

This answers: "How often does Harmonic catastrophically underperform
compared to the best SMART-family policy?"

Usage:
    python downside_risk_analysis.py --results-dir results
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
HARMONIC_AGENT = "Weighted Harmonic"
SMART_AGENT = "SMART"
RELAXED_SMART_AGENT = "Relaxed SMART"

# Threshold percentages of mean return
TAU_PERCENTAGES = [0.05, 0.10, 0.20]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_mkdir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_runs_data(results_dir: str) -> pd.DataFrame:
    """Load param_sweep_runs.csv with raw per-run data."""
    runs_path = os.path.join(results_dir, "param_sweep_runs.csv")
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"Runs data not found: {runs_path}")
    dfr = pd.read_csv(runs_path)

    # Expand params_json into columns to get reward_kind and duration_kind
    if "params_json" in dfr.columns:
        params = dfr["params_json"].apply(json.loads)
        dfp = pd.json_normalize(params)
        # Only keep key regime columns to avoid memory bloat
        regime_cols = ["reward_kind", "duration_kind", "coupled"]
        available_cols = [c for c in regime_cols if c in dfp.columns]
        if available_cols:
            dfr = pd.concat([dfr, dfp[available_cols]], axis=1)

    return dfr


# -----------------------------------------------------------------------------
# Core Metric Computation
# -----------------------------------------------------------------------------
def compute_per_env_seed_delta(dfr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta_H for each (trial_id, seed) pair.

    For each (trial_id, seed), use the best hyperparameter per agent,
    then compute:
        delta_H = R_H - max(R_S, R_RS)

    Also computes delta_S = R_S - R_RS for baseline comparison.

    Returns:
        DataFrame with columns:
        - trial_id, seed
        - R_H (best Harmonic rate), R_S (best SMART rate), R_RS (best Relaxed SMART rate)
        - best_smart (max of R_S, R_RS)
        - delta_H (R_H - best_smart)
        - delta_S (R_S - R_RS)
        - mean_return (mean across all agents)
        - reward_kind, duration_kind, coupled, mismatch
    """
    # First, get best HP per (trial_id, seed, agent)
    # For each combo, select the hp_id with maximum avg_rate
    if "hp_id" not in dfr.columns:
        dfr = dfr.copy()
        dfr["hp_id"] = 0

    # Get the best avg_rate per (trial_id, seed, agent)
    best_hp_idx = dfr.groupby(["trial_id", "seed", "agent"])["avg_rate"].idxmax()
    best_hp_df = dfr.loc[best_hp_idx].copy()

    # Pivot to get one row per (trial_id, seed) with columns for each agent
    pivot = best_hp_df.pivot_table(
        index=["trial_id", "seed"],
        columns="agent",
        values="avg_rate",
        aggfunc="first"
    ).reset_index()

    # Extract agent rates
    R_H = pivot[HARMONIC_AGENT] if HARMONIC_AGENT in pivot.columns else pd.Series([np.nan] * len(pivot))
    R_S = pivot[SMART_AGENT] if SMART_AGENT in pivot.columns else pd.Series([np.nan] * len(pivot))
    R_RS = pivot[RELAXED_SMART_AGENT] if RELAXED_SMART_AGENT in pivot.columns else pd.Series([np.nan] * len(pivot))

    # Compute metrics
    result = pd.DataFrame({
        "trial_id": pivot["trial_id"],
        "seed": pivot["seed"],
        "R_H": R_H.values,
        "R_S": R_S.values,
        "R_RS": R_RS.values,
    })

    # best_smart = max(R_S, R_RS)
    result["best_smart"] = result[["R_S", "R_RS"]].max(axis=1)

    # delta_H = R_H - best_smart
    result["delta_H"] = result["R_H"] - result["best_smart"]

    # delta_S = R_S - R_RS (baseline comparison within SMART family)
    result["delta_S"] = result["R_S"] - result["R_RS"]

    # mean_return per (trial_id, seed) across all agents
    result["mean_return"] = result[["R_H", "R_S", "R_RS"]].mean(axis=1)

    # Merge regime info (from first matching row per trial_id)
    regime_cols = []
    for col in ["reward_kind", "duration_kind", "coupled"]:
        if col in dfr.columns:
            regime_cols.append(col)

    if regime_cols:
        regime_info = dfr.groupby("trial_id")[regime_cols].first().reset_index()
        result = result.merge(regime_info, on="trial_id", how="left")

        # Compute mismatch
        if "reward_kind" in result.columns and "duration_kind" in result.columns:
            result["mismatch"] = result["reward_kind"] != result["duration_kind"]
        else:
            result["mismatch"] = False
    else:
        result["mismatch"] = False
        result["coupled"] = True  # Default

    return result


def compute_downside_failure_probability(delta_df: pd.DataFrame, subset_name: str,
                                          tau_percentages: list = TAU_PERCENTAGES) -> pd.DataFrame:
    """
    Compute downside failure probability for a given subset.

    For each tau threshold (as % of mean_return per environment):
        P(delta_H < -tau) = fraction of (env, seed) pairs where Harmonic underperforms

    Also computes P(delta_S < -tau) as baseline (SMART vs Relaxed SMART).

    Args:
        delta_df: DataFrame with delta_H, delta_S, mean_return per (trial_id, seed)
        subset_name: Name of the subset for reporting
        tau_percentages: List of threshold percentages [0.05, 0.10, 0.20]

    Returns:
        DataFrame with columns: subset, tau_pct, P_harmonic_fails, P_smart_fails, n_samples
    """
    if len(delta_df) == 0:
        return pd.DataFrame()

    # Compute mean_return per environment (across seeds)
    env_mean_return = delta_df.groupby("trial_id")["mean_return"].mean()
    delta_df = delta_df.copy()
    delta_df["env_mean_return"] = delta_df["trial_id"].map(env_mean_return)

    results = []
    for tau_pct in tau_percentages:
        # tau = tau_pct * mean_return per environment
        delta_df["tau"] = tau_pct * delta_df["env_mean_return"]

        # Downside failure for Harmonic: delta_H < -tau
        harmonic_fails = (delta_df["delta_H"] < -delta_df["tau"])
        P_harmonic_fails = harmonic_fails.mean()

        # Downside failure for SMART (vs Relaxed SMART): delta_S < -tau
        # This shows probability that SMART underperforms Relaxed SMART by tau
        smart_fails = (delta_df["delta_S"] < -delta_df["tau"])
        P_smart_fails = smart_fails.mean()

        results.append({
            "subset": subset_name,
            "tau_pct": tau_pct,
            "P_harmonic_fails": P_harmonic_fails,
            "P_smart_fails": P_smart_fails,
            "n_samples": len(delta_df),
            "n_envs": delta_df["trial_id"].nunique(),
        })

    return pd.DataFrame(results)


def define_subsets(delta_df: pd.DataFrame) -> dict:
    """
    Define experimental subsets for analysis.

    Returns:
        dict mapping subset_name -> filtered DataFrame
    """
    subsets = {}

    # 1. All environments
    subsets["All environments"] = delta_df

    # 2. Mismatch only (reward_kind != duration_kind)
    if "mismatch" in delta_df.columns:
        subsets["Mismatch only"] = delta_df[delta_df["mismatch"] == True]

    # 3. Mismatch + stochastic duration (duration_kind != constant)
    if "mismatch" in delta_df.columns and "duration_kind" in delta_df.columns:
        mask = (delta_df["mismatch"] == True) & (delta_df["duration_kind"] != "constant")
        subsets["Mismatch + stochastic duration"] = delta_df[mask]

    # 4. Mismatch + coupled == True
    if "mismatch" in delta_df.columns and "coupled" in delta_df.columns:
        mask = (delta_df["mismatch"] == True) & (delta_df["coupled"] == True)
        subsets["Mismatch + coupled"] = delta_df[mask]

    return subsets


# -----------------------------------------------------------------------------
# Output Functions
# -----------------------------------------------------------------------------
def generate_table(results_df: pd.DataFrame, out_dir: str) -> None:
    """
    Save and print the downside failure probability table.
    """
    # Format for display
    display_df = results_df.copy()
    display_df["tau_pct_str"] = (display_df["tau_pct"] * 100).astype(int).astype(str) + "%"
    display_df["P_harmonic_fails_pct"] = (display_df["P_harmonic_fails"] * 100).round(2)
    display_df["P_smart_fails_pct"] = (display_df["P_smart_fails"] * 100).round(2)

    # Reorder columns for output
    output_df = display_df[[
        "subset", "tau_pct_str", "P_harmonic_fails_pct", "P_smart_fails_pct",
        "n_samples", "n_envs"
    ]].rename(columns={
        "tau_pct_str": "tau (%mean)",
        "P_harmonic_fails_pct": "P(Harmonic fails) %",
        "P_smart_fails_pct": "P(SMART fails) %",
        "n_samples": "n_samples",
        "n_envs": "n_envs",
    })

    # Save CSV
    csv_path = os.path.join(out_dir, "downside_failure_probability.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved table to: {csv_path}")

    # Print table
    print("\n" + "=" * 80)
    print("DOWNSIDE FAILURE PROBABILITY TABLE")
    print("=" * 80)
    print(f"\nDefinition: P(Harmonic < max(SMART, Relaxed SMART) - τ)")
    print(f"            where τ = percentage × mean_return per environment\n")
    print(output_df.to_string(index=False))
    print()


def generate_plot(results_df: pd.DataFrame, out_dir: str) -> None:
    """
    Generate bar plot of downside failure probability.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    subsets = results_df["subset"].unique()
    tau_pcts = sorted(results_df["tau_pct"].unique())

    # Prepare data for grouped bar plot
    n_subsets = len(subsets)
    n_taus = len(tau_pcts)
    x = np.arange(n_subsets)
    width = 0.35 / n_taus  # Narrower bars to fit multiple tau values

    colors_harmonic = plt.cm.Greens(np.linspace(0.4, 0.8, n_taus))
    colors_smart = plt.cm.Blues(np.linspace(0.4, 0.8, n_taus))

    # Plot bars for each tau
    for i, tau_pct in enumerate(tau_pcts):
        tau_data = results_df[results_df["tau_pct"] == tau_pct]

        # Ensure data is in same order as subsets
        harmonic_vals = []
        smart_vals = []
        for subset in subsets:
            row = tau_data[tau_data["subset"] == subset]
            if len(row) > 0:
                harmonic_vals.append(row["P_harmonic_fails"].values[0])
                smart_vals.append(row["P_smart_fails"].values[0])
            else:
                harmonic_vals.append(0)
                smart_vals.append(0)

        # Offset positions for grouped bars
        offset = (i - n_taus/2 + 0.5) * width * 2

        bars_h = ax.bar(x + offset - width/2, harmonic_vals, width,
                        label=f"Harmonic (τ={int(tau_pct*100)}%)" if i == 0 else "",
                        color=colors_harmonic[i], edgecolor="black", linewidth=0.5,
                        alpha=0.9)
        bars_s = ax.bar(x + offset + width/2, smart_vals, width,
                        label=f"SMART (τ={int(tau_pct*100)}%)" if i == 0 else "",
                        color=colors_smart[i], edgecolor="black", linewidth=0.5,
                        alpha=0.9)

        # Add value labels on bars
        for bar, val in zip(bars_h, harmonic_vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f"{val*100:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90)
        for bar, val in zip(bars_s, smart_vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f"{val*100:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xlabel("Experimental Subset", fontsize=12)
    ax.set_ylabel("Downside Failure Probability", fontsize=12)
    ax.set_title("Downside Failure Probability: P(Algorithm < Best Alternative - τ)\n"
                 "(Lower is better — fewer catastrophic underperformances)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.3))

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for i, tau_pct in enumerate(tau_pcts):
        legend_elements.append(Patch(facecolor=colors_harmonic[i],
                                     label=f"Harmonic (τ={int(tau_pct*100)}%)"))
        legend_elements.append(Patch(facecolor=colors_smart[i],
                                     label=f"SMART (τ={int(tau_pct*100)}%)"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=2)

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "downside_failure_probability.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  Saved plot to: {plot_path}")


def generate_line_plot(results_df: pd.DataFrame, out_dir: str) -> None:
    """
    Generate line plot: x-axis is τ (% of mean return), y-axis is failure probability.
    One line per algorithm family per subset.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    subsets = results_df["subset"].unique()
    tau_pcts = sorted(results_df["tau_pct"].unique())
    tau_labels = [f"{int(t*100)}%" for t in tau_pcts]

    for idx, subset in enumerate(subsets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        subset_data = results_df[results_df["subset"] == subset].sort_values("tau_pct")

        ax.plot(tau_pcts, subset_data["P_harmonic_fails"].values,
                "g-o", linewidth=2, markersize=8, label="Harmonic")
        ax.plot(tau_pcts, subset_data["P_smart_fails"].values,
                "b-s", linewidth=2, markersize=8, label="SMART (vs Relaxed)")

        ax.set_xlabel("τ (% of mean return)", fontsize=10)
        ax.set_ylabel("Downside Failure Probability", fontsize=10)
        ax.set_title(f"{subset}\n(n={subset_data['n_samples'].iloc[0]:,} samples, "
                     f"{subset_data['n_envs'].iloc[0]} envs)", fontsize=11)
        ax.set_xticks(tau_pcts)
        ax.set_xticklabels(tau_labels)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.1, ax.get_ylim()[1] * 1.1))

    # Hide unused subplots
    for idx in range(len(subsets), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Downside Failure Probability by Threshold\n"
                 "P(Algorithm < Best Alternative - τ)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "downside_failure_probability_lines.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved line plot to: {plot_path}")


def generate_textual_summary(results_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    """
    Print a short textual summary suitable for inclusion in a report.
    """
    print("\n" + "=" * 80)
    print("TEXTUAL SUMMARY")
    print("=" * 80)

    # Focus on the key regime: mismatch + stochastic duration
    key_subset = "Mismatch + stochastic duration"
    tau_20 = 0.20

    subset_results = results_df[(results_df["subset"] == key_subset) &
                                 (results_df["tau_pct"] == tau_20)]

    if len(subset_results) > 0:
        p_h = subset_results["P_harmonic_fails"].values[0]
        p_s = subset_results["P_smart_fails"].values[0]

        if p_h > 0 and p_s > 0:
            reduction_factor = p_s / p_h
            print(f"\n> In {key_subset} regimes, Harmonic reduces 20% downside failures")
            print(f"> by {reduction_factor:.1f}× compared to SMART.")
            print(f"> (P(Harmonic fails) = {p_h*100:.1f}%, P(SMART fails) = {p_s*100:.1f}%)")
        elif p_h == 0:
            print(f"\n> In {key_subset} regimes, Harmonic has ZERO 20% downside failures.")
            print(f"> (P(SMART fails) = {p_s*100:.1f}%)")
        else:
            print(f"\n> In {key_subset} regimes:")
            print(f"> P(Harmonic fails) = {p_h*100:.1f}%, P(SMART fails) = {p_s*100:.1f}%")

    # Additional insights
    all_env_results = results_df[(results_df["subset"] == "All environments") &
                                  (results_df["tau_pct"] == 0.10)]
    if len(all_env_results) > 0:
        p_h_all = all_env_results["P_harmonic_fails"].values[0]
        p_s_all = all_env_results["P_smart_fails"].values[0]
        print(f"\n> Across all environments (10% threshold):")
        print(f"> P(Harmonic fails) = {p_h_all*100:.1f}%, P(SMART fails) = {p_s_all*100:.1f}%")

    # Overall message
    print("\n> Key insight: Harmonic's advantage is in reduced tail risk,")
    print("> not necessarily higher mean performance.")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute downside failure probability metric for SMDP algorithms"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing param_sweep_runs.csv (default: results)"
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.join(results_dir, "plots", "downside_risk")

    print("=" * 80)
    print("DOWNSIDE FAILURE PROBABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {out_dir}")

    safe_mkdir(out_dir)

    # Load data
    print("\n[1] Loading raw runs data...")
    try:
        dfr = load_runs_data(results_dir)
        print(f"    Loaded {len(dfr)} raw runs")
        print(f"    Unique trials: {dfr['trial_id'].nunique()}")
        print(f"    Unique seeds: {dfr['seed'].nunique()}")
        print(f"    Agents: {dfr['agent'].unique().tolist()}")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        return

    # Compute per-(env, seed) delta
    print("\n[2] Computing per-(env, seed) delta_H...")
    delta_df = compute_per_env_seed_delta(dfr)
    print(f"    Computed {len(delta_df)} (env, seed) pairs")

    # Check for available regime info
    if "mismatch" in delta_df.columns:
        n_mismatch = delta_df["mismatch"].sum()
        print(f"    Mismatch environments: {delta_df[delta_df['mismatch']]['trial_id'].nunique()}")

    # Define subsets
    print("\n[3] Defining experimental subsets...")
    subsets = define_subsets(delta_df)
    for name, subset in subsets.items():
        print(f"    {name}: {len(subset)} samples, {subset['trial_id'].nunique()} envs")

    # Compute downside failure probability for each subset
    print("\n[4] Computing downside failure probabilities...")
    all_results = []
    for subset_name, subset_df in subsets.items():
        subset_results = compute_downside_failure_probability(subset_df, subset_name)
        all_results.append(subset_results)

    results_df = pd.concat(all_results, ignore_index=True)

    # Generate outputs
    print("\n[5] Generating outputs...")
    generate_table(results_df, out_dir)
    generate_plot(results_df, out_dir)
    generate_line_plot(results_df, out_dir)
    generate_textual_summary(results_df, delta_df)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nOutput files in: {out_dir}")
    print("  - downside_failure_probability.csv")
    print("  - downside_failure_probability.png")
    print("  - downside_failure_probability_lines.png")


if __name__ == "__main__":
    main()

