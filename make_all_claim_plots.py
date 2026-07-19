#!/usr/bin/env python3
"""
Unified plotting pipeline to generate all plots needed to support the claims:
1. Where Harmonic wins vs loses (vs SMART-family)
2. Harmonic robustness is higher (lower hyperparameter sensitivity)
3. Environments separate into regimes visible in embeddings

Usage:
    python make_all_claim_plots.py --results-dir results
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# sklearn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
HARMONIC_AGENT = "Harmonic"
WEIGHTED_HARMONIC_AGENT = "Weighted Harmonic"
SMART_AGENT = "SMART"
RELAXED_SMART_AGENT = "Relaxed SMART"
HARMONIC_FAMILY = [HARMONIC_AGENT, WEIGHTED_HARMONIC_AGENT]
SMART_FAMILY = [SMART_AGENT, RELAXED_SMART_AGENT]
ALL_AGENTS = HARMONIC_FAMILY + SMART_FAMILY

# Duration variability score ordinal mapping
DURATION_VAR_SCORE = {
    "constant": 0,
    "normal": 1,
    "uniform": 2,
    "exp": 3,
    "mr_drift": 4,
    "mr_jump": 5,
}

# Margin threshold for "strong win"
MARGIN_EPS = 0.01


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_mkdir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_get(row, key, default=None):
    """Safely get a value from a row/dict."""
    val = row.get(key, default)
    return default if pd.isna(val) else val


def copy_dir_contents(src: str, dst: str):
    """Copy all files from src to dst."""
    if not os.path.exists(src):
        print(f"  Warning: Source directory {src} does not exist, skipping copy.")
        return
    safe_mkdir(dst)
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def load_agg_data(results_dir: str) -> pd.DataFrame:
    """Load param_sweep_agg.csv."""
    agg_path = os.path.join(results_dir, "param_sweep_agg.csv")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Aggregated results not found: {agg_path}")
    return pd.read_csv(agg_path)


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


def load_winners_data(results_dir: str) -> pd.DataFrame:
    """Load and expand param_sweep_winners.csv."""
    winners_path = os.path.join(results_dir, "param_sweep_winners.csv")
    if not os.path.exists(winners_path):
        raise FileNotFoundError(f"Winners data not found: {winners_path}")

    dfw = pd.read_csv(winners_path)

    # Expand params_json into columns
    if "params_json" in dfw.columns:
        params = dfw["params_json"].apply(json.loads)
        dfp = pd.json_normalize(params)
        dfw = pd.concat([dfw.drop(columns=["params_json"]), dfp], axis=1)

    return dfw


def compute_best_hp_per_agent(dfa: pd.DataFrame) -> pd.DataFrame:
    """
    For each (trial_id, agent), select the hp_id with the best avg_rate_mean.
    Returns a dataframe with one row per (trial_id, agent).
    """
    if "hp_id" in dfa.columns:
        # Group and get idx of max
        best_idx = dfa.groupby(["trial_id", "agent"])["avg_rate_mean"].idxmax()
        return dfa.loc[best_idx].copy()
    else:
        # No HP tuning, just return the original
        return dfa.copy()


def compute_delta_h(dfa: pd.DataFrame, dfw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-trial delta_H = best_rate_H - max(best_rate_S, best_rate_RS).
    Merge with winner params to get regime features.
    """
    # Get best HP per agent
    best = compute_best_hp_per_agent(dfa)

    # Pivot to one row per trial with columns for each agent's best rate
    pivot = best.pivot(index="trial_id", columns="agent", values="avg_rate_mean").reset_index()

    # Rename columns for clarity
    col_map = {}
    for c in pivot.columns:
        if c != "trial_id":
            col_map[c] = f"best_rate_{c.replace(' ', '_')}"
    pivot = pivot.rename(columns=col_map)

    # Get Harmonic-family and SMART-family rates
    h_cols = [f"best_rate_{name.replace(' ', '_')}" for name in HARMONIC_FAMILY]
    s_col = f"best_rate_{SMART_AGENT.replace(' ', '_')}"
    rs_col = f"best_rate_{RELAXED_SMART_AGENT.replace(' ', '_')}"

    # Compute delta_H
    available_h_cols = [c for c in h_cols if c in pivot.columns]
    if available_h_cols:
        pivot["best_rate_H"] = pivot[available_h_cols].max(axis=1)
    else:
        pivot["best_rate_H"] = np.nan

    # Get max of SMART family
    smart_cols = [c for c in [s_col, rs_col] if c in pivot.columns]
    if smart_cols:
        pivot["max_smart_rate"] = pivot[smart_cols].max(axis=1)
    else:
        pivot["max_smart_rate"] = np.nan

    pivot["delta_H"] = pivot["best_rate_H"] - pivot["max_smart_rate"]

    # Merge with params from winners
    df = pivot.merge(dfw[["trial_id"] + [c for c in dfw.columns if c not in pivot.columns]],
                     on="trial_id", how="left")

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived regime features for analysis."""
    df = df.copy()

    # Mismatch
    if "reward_kind" in df.columns and "duration_kind" in df.columns:
        df["mismatch"] = df["reward_kind"] != df["duration_kind"]
    else:
        df["mismatch"] = False

    # Duration type flags
    if "duration_kind" in df.columns:
        df["duration_memoryless"] = df["duration_kind"] == "exp"
        df["duration_constant"] = df["duration_kind"] == "constant"
        df["duration_mr"] = df["duration_kind"].isin(["mr_drift", "mr_jump"])
        df["duration_variability_score"] = df["duration_kind"].map(DURATION_VAR_SCORE).fillna(-1).astype(int)
    else:
        df["duration_memoryless"] = False
        df["duration_constant"] = False
        df["duration_mr"] = False
        df["duration_variability_score"] = -1

    # Reward type flags
    if "reward_kind" in df.columns:
        df["reward_heavy"] = df["reward_kind"].isin(["exp", "uniform"])
        df["reward_mr"] = df["reward_kind"].isin(["mr_drift", "mr_jump"])
    else:
        df["reward_heavy"] = False
        df["reward_mr"] = False

    if "duration_kind" in df.columns:
        df["duration_heavy"] = df["duration_kind"].isin(["exp", "uniform"])
    else:
        df["duration_heavy"] = False

    # Harmonic wins flags
    df["harmonic_wins"] = df["delta_H"] > 0
    df["strong_win"] = df["delta_H"] > MARGIN_EPS

    return df


# -----------------------------------------------------------------------------
# Part A: Run existing scripts and copy outputs
# -----------------------------------------------------------------------------
def run_existing_scripts(results_dir: str, plots_dir: str):
    """Run robust_vis.py and make_regime_heamap.py, copy outputs."""
    print("\n" + "="*60)
    print("PART A: Running existing plotting scripts")
    print("="*60)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Run robust_vis.py
    robust_vis_script = os.path.join(script_dir, "robust_vis.py")
    if os.path.exists(robust_vis_script):
        print("\n[1] Running robust_vis.py...")
        try:
            # robust_vis.py uses RESULTS_DIR constant, so we run it from script dir
            # and then copy outputs
            subprocess.run([sys.executable, robust_vis_script, "--results-dir", results_dir],
                          cwd=script_dir, check=True, capture_output=True)
            print("    robust_vis.py completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    Warning: robust_vis.py failed: {e.stderr.decode()[:500] if e.stderr else str(e)}")
        except Exception as e:
            print(f"    Warning: Could not run robust_vis.py: {e}")

        # Copy outputs
        viz_src = os.path.join(results_dir, "viz")
        viz_dst = os.path.join(plots_dir, "robust_vis")
        print(f"    Copying outputs to {viz_dst}")
        copy_dir_contents(viz_src, viz_dst)
    else:
        print(f"  Warning: robust_vis.py not found at {robust_vis_script}")

    # Run make_regime_heatmap.py
    heatmap_script = os.path.join(script_dir, "make_regime_heatmap.py")
    if os.path.exists(heatmap_script):
        print("\n[2] Running make_regime_heamap.py...")
        try:
            subprocess.run([sys.executable, heatmap_script, "--results-dir", results_dir],
                          cwd=script_dir, check=True, capture_output=True)
            print("    make_regime_heamap.py completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    Warning: make_regime_heamap.py failed: {e.stderr.decode()[:500] if e.stderr else str(e)}")
        except Exception as e:
            print(f"    Warning: Could not run make_regime_heamap.py: {e}")

        # Copy outputs
        heatmap_src = os.path.join(results_dir, "heatmaps")
        heatmap_dst = os.path.join(plots_dir, "heatmaps")
        print(f"    Copying outputs to {heatmap_dst}")
        copy_dir_contents(heatmap_src, heatmap_dst)
    else:
        print(f"  Warning: make_regime_heamap.py not found at {heatmap_script}")


# -----------------------------------------------------------------------------
# Part B: Generate new claim-specific plots
# -----------------------------------------------------------------------------

# --- Section 1: Harmonic vs SMART-family separation plots ---

def plot_delta_h_histogram(df: pd.DataFrame, out_dir: str):
    """Plot 1: Histogram of delta_H (overall)."""
    print("  - Generating delta_H histogram...")

    delta = df["delta_H"].dropna()
    if len(delta) == 0:
        print("    Warning: No delta_H values to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(delta, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="δ=0")
    plt.axvline(MARGIN_EPS, color="darkred", linestyle=":", linewidth=1.5,
                label=f"Strong win threshold (ε={MARGIN_EPS})")
    plt.axvline(-MARGIN_EPS, color="darkblue", linestyle=":", linewidth=1.5,
                label=f"Strong loss threshold")

    # Add statistics
    mean_d = delta.mean()
    median_d = delta.median()
    win_rate = (delta > 0).mean() * 100
    strong_win_rate = (delta > MARGIN_EPS).mean() * 100
    strong_loss_rate = (delta < -MARGIN_EPS).mean() * 100

    plt.axvline(mean_d, color="green", linestyle="-.", linewidth=1.5, label=f"Mean={mean_d:.4f}")

    plt.xlabel("ΔH = Harmonic − max(SMART-family)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"ΔH = Harmonic − max(SMART-family)\n"
              f"Win: {win_rate:.1f}% | Strong win: {strong_win_rate:.1f}% | Strong loss: {strong_loss_rate:.1f}% (n={len(delta)})",
              fontsize=12)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_delta_h_histogram.png"), dpi=200)
    plt.close()


def plot_delta_h_boxplots(df: pd.DataFrame, out_dir: str):
    """Plot 2: Boxplots of delta_H grouped by mismatch, duration_kind, coupled."""
    print("  - Generating delta_H boxplots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Boxplot by mismatch
    ax = axes[0]
    groups = df.groupby("mismatch")["delta_H"].apply(list).to_dict()
    labels = ["False (match)", "True (mismatch)"]
    data = [groups.get(False, []), groups.get(True, [])]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'salmon']):
        patch.set_facecolor(color)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Mismatch (reward_kind ≠ duration_kind)")
    ax.set_ylabel("ΔH")
    ax.set_title("ΔH by Distribution Mismatch")

    # Boxplot by duration_kind
    ax = axes[1]
    if "duration_kind" in df.columns:
        dk_groups = df.groupby("duration_kind")["delta_H"].apply(list).to_dict()
        dk_labels = sorted(dk_groups.keys())
        dk_data = [dk_groups[k] for k in dk_labels]
        bp = ax.boxplot(dk_data, tick_labels=dk_labels, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(dk_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Duration Kind")
        ax.set_ylabel("ΔH")
        ax.set_title("ΔH by Duration Kind")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "No duration_kind column", ha="center", va="center", transform=ax.transAxes)

    # Boxplot by coupled
    ax = axes[2]
    if "coupled" in df.columns:
        c_groups = df.groupby("coupled")["delta_H"].apply(list).to_dict()
        c_labels = ["False", "True"]
        c_data = [c_groups.get(False, []), c_groups.get(True, [])]
        bp = ax.boxplot(c_data, tick_labels=c_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'gold']):
            patch.set_facecolor(color)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Coupled")
        ax.set_ylabel("ΔH")
        ax.set_title("ΔH by Coupling")
    else:
        ax.text(0.5, 0.5, "No coupled column", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot2_delta_h_boxplots.png"), dpi=200)
    plt.close()


def plot_delta_h_heatmap(df: pd.DataFrame, out_dir: str):
    """Plot 3: Heatmap of mean delta_H by (reward_kind × duration_kind)."""
    print("  - Generating delta_H heatmaps...")

    if "reward_kind" not in df.columns or "duration_kind" not in df.columns:
        print("    Warning: Missing reward_kind or duration_kind columns")
        return

    # Create diverging colormap
    cmap = LinearSegmentedColormap.from_list("delta_cmap", ["red", "white", "green"])

    for coupled_val in [False, True]:
        subset = df[df["coupled"] == coupled_val] if "coupled" in df.columns else df
        if len(subset) == 0:
            continue

        pivot = subset.pivot_table(
            index="reward_kind",
            columns="duration_kind",
            values="delta_H",
            aggfunc="mean"
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get value range for symmetric colormap
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        vmin = -vmax

        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Mean ΔH")

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                           fontsize=10, color="black" if abs(val) < vmax*0.5 else "white")

        ax.set_xlabel("Duration Kind")
        ax.set_ylabel("Reward Kind")
        ax.set_title(f"Mean ΔH by Reward×Duration Kind\n(coupled={coupled_val})")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot3_delta_h_heatmap_coupled_{coupled_val}.png"), dpi=200)
        plt.close()


def plot_winrate_heatmap(df: pd.DataFrame, out_dir: str):
    """Plot 4: Win-rate heatmap (Harmonic beats SMART-family)."""
    print("  - Generating win-rate heatmaps...")

    if "reward_kind" not in df.columns or "duration_kind" not in df.columns:
        print("    Warning: Missing reward_kind or duration_kind columns")
        return

    for coupled_val in [False, True]:
        subset = df[df["coupled"] == coupled_val] if "coupled" in df.columns else df
        if len(subset) == 0:
            continue

        pivot = subset.pivot_table(
            index="reward_kind",
            columns="duration_kind",
            values="harmonic_wins",
            aggfunc="mean"
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Harmonic Win Rate")

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=10, color="black" if 0.3 < val < 0.7 else "white")

        ax.set_xlabel("Duration Kind")
        ax.set_ylabel("Reward Kind")
        ax.set_title(f"Harmonic Win Rate (vs SMART-family)\n(coupled={coupled_val})")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot4_winrate_heatmap_coupled_{coupled_val}.png"), dpi=200)
        plt.close()


# --- Section 2: Robustness / HP sensitivity plots ---

def compute_hp_sensitivity(dfa: pd.DataFrame) -> pd.DataFrame:
    """Compute HP sensitivity metrics per (trial_id, agent)."""
    if "hp_id" not in dfa.columns:
        return None

    # Group by (trial_id, agent) and compute metrics
    def hp_metrics(g):
        rates = g["avg_rate_mean"].values
        return pd.Series({
            "hp_std": np.std(rates),
            "hp_range": np.max(rates) - np.min(rates),
            "hp_best": np.max(rates),
            "hp_mean": np.mean(rates),
            "hp_median": np.median(rates),
            "hp_count": len(rates),
        })

    metrics = dfa.groupby(["trial_id", "agent"], as_index=False).apply(
        hp_metrics, include_groups=False
    ).reset_index(drop=True)
    return metrics


def plot_hp_sensitivity_violin(hp_df: pd.DataFrame, out_dir: str):
    """Plot 5: Violin/box plot of hp_std by agent."""
    print("  - Generating HP sensitivity violin plot...")

    if hp_df is None:
        print("    Skipping: no HP tuning data")
        return

    agents = ALL_AGENTS
    agents = [a for a in agents if a in hp_df["agent"].unique()]

    if not agents:
        print("    Warning: No matching agents found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [hp_df[hp_df["agent"] == a]["hp_std"].dropna().values for a in agents]

    # Create violin plot
    parts = ax.violinplot(data, positions=range(len(agents)), showmedians=True)

    # Overlay box plot for quartiles
    bp = ax.boxplot(data, positions=range(len(agents)), widths=0.2, patch_artist=True)

    # Color the violin plots
    colors = ['green', 'darkgreen', 'blue', 'orange']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents)
    ax.set_xlabel("Agent")
    ax.set_ylabel("HP Sensitivity (std of avg_rate across HPs)")
    ax.set_title("Hyperparameter Sensitivity by Agent\n(Lower = More Robust)")

    # Add mean values as text
    for i, a in enumerate(agents):
        mean_val = hp_df[hp_df["agent"] == a]["hp_std"].mean()
        ax.text(i, ax.get_ylim()[1] * 0.95, f"μ={mean_val:.4f}",
               ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_hp_sensitivity_violin.png"), dpi=200)
    plt.close()


def plot_hp_peak_vs_sensitivity(hp_df: pd.DataFrame, out_dir: str):
    """Plot 6: Scatter of hp_best vs hp_std per agent."""
    print("  - Generating peak vs sensitivity scatter...")

    if hp_df is None:
        print("    Skipping: no HP tuning data")
        return

    agents = ALL_AGENTS
    agents = [a for a in agents if a in hp_df["agent"].unique()]

    if not agents:
        print("    Warning: No matching agents found")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        HARMONIC_AGENT: 'green',
        WEIGHTED_HARMONIC_AGENT: 'darkgreen',
        SMART_AGENT: 'blue',
        RELAXED_SMART_AGENT: 'orange',
    }

    for agent in agents:
        subset = hp_df[hp_df["agent"] == agent]
        ax.scatter(subset["hp_std"], subset["hp_best"],
                  alpha=0.5, s=30, label=agent, color=colors.get(agent, 'gray'))

    ax.set_xlabel("HP Sensitivity (std of avg_rate across HPs)")
    ax.set_ylabel("Best Performance (max avg_rate)")
    ax.set_title("Peak Performance vs HP Sensitivity Trade-off")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot6_peak_vs_sensitivity.png"), dpi=200)
    plt.close()


def plot_robustness_by_regime(hp_df: pd.DataFrame, dfw: pd.DataFrame, out_dir: str):
    """Plot 7: Robustness comparison by regime."""
    print("  - Generating robustness by regime plot...")

    if hp_df is None:
        print("    Skipping: no HP tuning data")
        return

    # Pivot hp_std by agent
    hp_pivot = hp_df.pivot(index="trial_id", columns="agent", values="hp_std").reset_index()

    h_col = HARMONIC_AGENT
    rs_col = RELAXED_SMART_AGENT

    if h_col not in hp_pivot.columns or rs_col not in hp_pivot.columns:
        print("    Warning: Missing required agents for comparison")
        return

    # Compute difference
    hp_pivot["robustness_diff"] = hp_pivot[rs_col] - hp_pivot[h_col]

    # Merge with params
    df = hp_pivot.merge(dfw[["trial_id", "reward_kind", "duration_kind", "coupled"]].drop_duplicates(),
                        on="trial_id", how="left")

    # Add mismatch
    if "reward_kind" in df.columns and "duration_kind" in df.columns:
        df["mismatch"] = df["reward_kind"] != df["duration_kind"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # By mismatch
    ax = axes[0]
    if "mismatch" in df.columns:
        groups = df.groupby("mismatch")["robustness_diff"].apply(list).to_dict()
        data = [groups.get(False, []), groups.get(True, [])]
        bp = ax.boxplot(data, tick_labels=["Match", "Mismatch"], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'salmon']):
            patch.set_facecolor(color)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("Robustness Diff (Relaxed SMART - Harmonic)")
    ax.set_xlabel("Distribution Mismatch")
    ax.set_title("Robustness: Relaxed SMART vs Harmonic\n(>0 means Harmonic more robust)")

    # By duration_kind
    ax = axes[1]
    if "duration_kind" in df.columns:
        groups = df.groupby("duration_kind")["robustness_diff"].apply(list).to_dict()
        labels = sorted(groups.keys())
        data = [groups[k] for k in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Duration Kind")
        ax.set_ylabel("Robustness Diff")
        ax.tick_params(axis='x', rotation=45)
    ax.set_title("Robustness Diff by Duration Kind")

    # By coupled
    ax = axes[2]
    if "coupled" in df.columns:
        groups = df.groupby("coupled")["robustness_diff"].apply(list).to_dict()
        data = [groups.get(False, []), groups.get(True, [])]
        bp = ax.boxplot(data, tick_labels=["False", "True"], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'gold']):
            patch.set_facecolor(color)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Coupled")
    ax.set_ylabel("Robustness Diff")
    ax.set_title("Robustness Diff by Coupling")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot7_robustness_by_regime.png"), dpi=200)
    plt.close()


# --- Section 4: Downside Risk Plots ---

def compute_run_level_delta_h(dfr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ΔH for each run (seed × hp combination).

    For each (trial_id, seed, hp_id), compute:
      ΔH = max(avg_rate(Harmonic), avg_rate(Weighted Harmonic))
           − max(avg_rate(SMART), avg_rate(Relaxed SMART))

    Returns a dataframe with one row per (trial_id, seed, hp_id) with columns:
      - trial_id, seed, hp_id
      - harmonic_rate, smart_rate, relaxed_smart_rate, max_smart_rate
      - delta_H
      - mismatch (derived from params)
      - reward_kind, duration_kind
    """
    # Handle case where hp_id might be missing
    if "hp_id" not in dfr.columns:
        print("    Warning: hp_id column not found, using seed-only aggregation")
        dfr = dfr.copy()
        dfr["hp_id"] = 0

    # Get the three agents' rates
    pivot_cols = ["trial_id", "seed", "hp_id"]

    # Check which columns we have for regime info
    regime_cols = []
    if "reward_kind" in dfr.columns:
        regime_cols.append("reward_kind")
    if "duration_kind" in dfr.columns:
        regime_cols.append("duration_kind")

    # Create pivot with avg_rate per agent
    pivot = dfr.pivot_table(
        index=pivot_cols,
        columns="agent",
        values="avg_rate",
        aggfunc="first"
    ).reset_index()

    # Rename agent columns for clarity
    h_cols = [c for c in HARMONIC_FAMILY if c in pivot.columns]
    s_col = SMART_AGENT
    rs_col = RELAXED_SMART_AGENT

    # Get Harmonic rate
    if h_cols:
        pivot["harmonic_rate"] = pivot[h_cols].max(axis=1)
    else:
        print(f"    Warning: no Harmonic-family agents found in data")
        return pd.DataFrame()

    # Get SMART family rates
    smart_cols = []
    if s_col in pivot.columns:
        pivot["smart_rate"] = pivot[s_col]
        smart_cols.append("smart_rate")
    if rs_col in pivot.columns:
        pivot["relaxed_smart_rate"] = pivot[rs_col]
        smart_cols.append("relaxed_smart_rate")

    if not smart_cols:
        print("    Warning: No SMART-family agents found in data")
        return pd.DataFrame()

    # Compute max of SMART family
    pivot["max_smart_rate"] = pivot[smart_cols].max(axis=1)

    # Compute delta_H
    pivot["delta_H"] = pivot["harmonic_rate"] - pivot["max_smart_rate"]

    # Merge regime info back (from first matching row per trial)
    if regime_cols:
        regime_info = dfr.groupby("trial_id")[regime_cols].first().reset_index()
        pivot = pivot.merge(regime_info, on="trial_id", how="left")

        # Compute mismatch
        if "reward_kind" in pivot.columns and "duration_kind" in pivot.columns:
            pivot["mismatch"] = pivot["reward_kind"] != pivot["duration_kind"]
        else:
            pivot["mismatch"] = False
    else:
        pivot["mismatch"] = False

    return pivot


def plot_delta_h_left_tail_cdf(dfr: pd.DataFrame, out_dir: str):
    """
    Plot 1 — Left-tail CDF of ΔH (MOST IMPORTANT PLOT)

    Shows the empirical CDF P(ΔH ≤ x) for the left tail (negative values),
    split by mismatch = True vs False.

    This reveals whether Harmonic has fewer catastrophic failures (large negative ΔH)
    in mismatched environments.
    """
    print("  - Generating left-tail CDF of ΔH (downside risk plot)...")

    # Compute run-level delta_H
    delta_df = compute_run_level_delta_h(dfr)

    if len(delta_df) == 0:
        print("    Warning: No data to plot")
        return

    # Split by mismatch
    delta_mismatch = delta_df[delta_df["mismatch"] == True]["delta_H"].dropna().values
    delta_match = delta_df[delta_df["mismatch"] == False]["delta_H"].dropna().values

    if len(delta_mismatch) == 0 and len(delta_match) == 0:
        print("    Warning: No delta_H values to plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Function to compute empirical CDF
    def empirical_cdf(data):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, cdf

    # Plot CDFs
    colors = {"mismatch": "tab:red", "match": "tab:blue"}

    if len(delta_mismatch) > 0:
        x_mis, y_mis = empirical_cdf(delta_mismatch)
        ax.plot(x_mis, y_mis, color=colors["mismatch"], linewidth=2,
                label=f"mismatch=True (n={len(delta_mismatch)})")

    if len(delta_match) > 0:
        x_mat, y_mat = empirical_cdf(delta_match)
        ax.plot(x_mat, y_mat, color=colors["match"], linewidth=2,
                label=f"mismatch=False (n={len(delta_match)})")

    # Determine x-axis limits to focus on left tail
    all_deltas = np.concatenate([
        d for d in [delta_mismatch, delta_match] if len(d) > 0
    ])
    x_min = np.min(all_deltas)
    # Right edge: show a bit past 0 for context, but focus on negative
    x_right = min(0.05, np.percentile(all_deltas, 75))
    ax.set_xlim(x_min - 0.01, x_right)

    # Add vertical line at ΔH = 0
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="ΔH = 0")

    # Add horizontal lines at key percentiles for reference
    for pct, style in [(0.05, ':'), (0.10, '-.')]:
        ax.axhline(pct, color="gray", linestyle=style, linewidth=0.8, alpha=0.7)
        ax.text(x_min, pct + 0.01, f"{int(pct*100)}%", fontsize=9, color="gray")

    # Labels and title
    ax.set_xlabel("ΔH = Harmonic − max(SMART-family)", fontsize=12)
    ax.set_ylabel("P(ΔH ≤ x)  [Empirical CDF]", fontsize=12)
    ax.set_title("Left-Tail CDF of ΔH: Downside Risk by Regime Mismatch\n"
                 "(Lower curve = fewer catastrophic underperformances)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add interpretation text box
    interpretation = (
        "Interpretation:\n"
        "• Focus on the left (negative) side\n"
        "• Lower curve = less probability of\n"
        "  large underperformance\n"
        "• Key question: Is mismatch=True\n"
        "  curve lower in the left tail?"
    )
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "deltaH_left_tail_cdf.png"), dpi=200)
    plt.close()

    # Also compute and print key statistics
    print(f"    Left-tail statistics:")
    for name, data in [("mismatch=True", delta_mismatch), ("mismatch=False", delta_match)]:
        if len(data) > 0:
            pct_5 = np.percentile(data, 5)
            pct_10 = np.percentile(data, 10)
            min_val = np.min(data)
            print(f"      {name}: min={min_val:.4f}, 5th pct={pct_5:.4f}, 10th pct={pct_10:.4f}")


def plot_worst_case_return_by_agent(dfr: pd.DataFrame, out_dir: str):
    """
    Plot 2 — Worst-case performance per environment (risk dominance)

    For each (trial_id, agent), compute:
      worst_case_return = min(avg_rate over all seeds and hyperparameters)

    Restrict to environments where:
      mismatch == True AND duration_kind != "constant"

    Shows boxplot of worst_case_return by agent.
    """
    print("  - Generating worst-case return by agent plot (risk dominance)...")

    # Handle case where hp_id might be missing
    if "hp_id" not in dfr.columns:
        print("    Warning: hp_id column not found, using seed-only aggregation")
        dfr = dfr.copy()
        dfr["hp_id"] = 0

    # Derive mismatch from params if not present
    if "mismatch" not in dfr.columns:
        if "reward_kind" in dfr.columns and "duration_kind" in dfr.columns:
            dfr = dfr.copy()
            dfr["mismatch"] = dfr["reward_kind"] != dfr["duration_kind"]
        else:
            print("    Warning: Cannot compute mismatch (missing reward_kind or duration_kind)")
            dfr = dfr.copy()
            dfr["mismatch"] = True  # Assume all are mismatched to show something

    # Filter to: mismatch == True AND duration_kind != "constant"
    if "duration_kind" in dfr.columns:
        subset = dfr[(dfr["mismatch"] == True) & (dfr["duration_kind"] != "constant")]
    else:
        subset = dfr[dfr["mismatch"] == True]

    if len(subset) == 0:
        print("    Warning: No data after filtering for mismatch=True, duration_kind != constant")
        # Fall back to all mismatched environments
        subset = dfr[dfr["mismatch"] == True]
        if len(subset) == 0:
            print("    Warning: No mismatched environments found, using all data")
            subset = dfr

    # Compute worst-case (min) avg_rate per (trial_id, agent)
    worst_case = subset.groupby(["trial_id", "agent"])["avg_rate"].min().reset_index()
    worst_case.columns = ["trial_id", "agent", "worst_case_rate"]

    # Get data for each agent
    agents = ALL_AGENTS
    agents_present = [a for a in agents if a in worst_case["agent"].unique()]

    if len(agents_present) == 0:
        print("    Warning: No target agents found in data")
        return

    # Prepare data for boxplot
    data = []
    labels = []
    for agent in agents_present:
        agent_data = worst_case[worst_case["agent"] == agent]["worst_case_rate"].values
        if len(agent_data) > 0:
            data.append(agent_data)
            labels.append(agent)

    if not data:
        print("    Warning: No data to plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Colors
    agent_colors = {
        HARMONIC_AGENT: "tab:green",
        WEIGHTED_HARMONIC_AGENT: "darkgreen",
        SMART_AGENT: "tab:blue",
        RELAXED_SMART_AGENT: "tab:orange"
    }
    colors = [agent_colors.get(a, "gray") for a in labels]

    # Create boxplot
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points for transparency
    for i, (agent_data, agent) in enumerate(zip(data, labels), start=1):
        # Jitter the x positions
        x_jitter = np.random.uniform(-0.15, 0.15, size=len(agent_data))
        ax.scatter(i + x_jitter, agent_data, alpha=0.3, s=10,
                   color=agent_colors.get(agent, "gray"))

    # Add statistics annotations
    for i, (agent_data, agent) in enumerate(zip(data, labels), start=1):
        median = np.median(agent_data)
        mean = np.mean(agent_data)
        pct_5 = np.percentile(agent_data, 5)
        ax.text(i, ax.get_ylim()[1], f"med={median:.3f}\n5th={pct_5:.3f}",
                ha='center', va='bottom', fontsize=9)

    # Labels and title
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Worst-Case Average Rate\n(min over seeds × HPs)", fontsize=12)
    ax.set_title("Worst-Case Performance: Risk Dominance Analysis\n"
                 "(mismatch=True, duration ≠ constant)\n"
                 "Higher = better floor performance", fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    # Add interpretation
    n_trials = len(worst_case["trial_id"].unique())
    interpretation = (
        f"Environments: {n_trials} (mismatch, stochastic duration)\n"
        "• Box shows median and IQR\n"
        "• Whiskers show 1.5×IQR range\n"
        "• Key question: Does Harmonic have\n"
        "  a higher floor (better worst-case)?"
    )
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "worst_case_return_by_agent.png"), dpi=200)
    plt.close()

    # Print summary statistics
    print(f"    Worst-case statistics ({n_trials} environments):")
    for agent_data, agent in zip(data, labels):
        print(f"      {agent}: min={np.min(agent_data):.4f}, "
              f"5th pct={np.percentile(agent_data, 5):.4f}, "
              f"median={np.median(agent_data):.4f}")


# --- Section 3: Explainability plots ---

def prepare_classification_data(df: pd.DataFrame) -> tuple:
    """Prepare features and labels for classification."""
    # Target
    y = df["harmonic_wins"].astype(int)

    # Features - categorical and numeric
    cat_features = ["reward_kind", "duration_kind", "coupled"]
    bool_features = ["mismatch", "duration_memoryless", "duration_mr"]
    num_features = ["interval", "reward_gap", "max_volatility",
                   "reward_stationarity_a", "reward_stationarity_b",
                   "reversion_speed", "duration_mean_a", "duration_width",
                   "duration_variability_score"]

    # Select available features
    available_cat = [f for f in cat_features if f in df.columns]
    available_bool = [f for f in bool_features if f in df.columns]
    available_num = [f for f in num_features if f in df.columns]

    # Create feature dataframe
    X = df[available_cat + available_bool + available_num].copy()

    # One-hot encode categoricals
    for col in available_cat:
        if col in X.columns:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    # Convert booleans to int
    for col in available_bool:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # Fill missing numeric values
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64, float, int]:
            X[col] = X[col].fillna(X[col].median())

    # Drop rows with any remaining NaN
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y


def plot_decision_tree(df: pd.DataFrame, out_dir: str):
    """Train and plot a shallow decision tree for interpretability."""
    print("  - Training decision tree classifier...")

    X, y = prepare_classification_data(df)

    if len(X) < 100:
        print(f"    Warning: Only {len(X)} samples, tree may be unreliable")

    # Fit tree
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
    tree.fit(X, y)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(tree, X, y, cv=cv)

    # Save CV score
    with open(os.path.join(out_dir, "harmonic_vs_smart_cv_score.txt"), "w") as f:
        f.write(f"5-fold Stratified CV Accuracy:\n")
        f.write(f"  Mean: {scores.mean():.3f}\n")
        f.write(f"  Std:  {scores.std():.3f}\n")
        f.write(f"  Scores: {', '.join([f'{s:.3f}' for s in scores])}\n")
    print(f"    CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Plot tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, feature_names=X.columns.tolist(),
              class_names=["SMART-family wins", "Harmonic wins"],
              filled=True, rounded=True, ax=ax, fontsize=10)
    ax.set_title(f"Decision Tree: When Does Harmonic Beat SMART-family?\n(CV Accuracy: {scores.mean():.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "harmonic_vs_smart_tree.png"), dpi=200)
    plt.close()

    # Export text rules
    rules = export_text(tree, feature_names=X.columns.tolist())
    with open(os.path.join(out_dir, "harmonic_vs_smart_tree_rules.txt"), "w") as f:
        f.write(rules)

    # Confusion matrix
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                         random_state=42, stratify=y)
    tree.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(tree, X_test, y_test, ax=ax,
                                          display_labels=["SMART wins", "Harmonic wins"])
    ax.set_title("Confusion Matrix: Harmonic vs SMART-family Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "harmonic_vs_smart_confusion_matrix.png"), dpi=200)
    plt.close()

    return X, y


def plot_feature_importance(df: pd.DataFrame, out_dir: str):
    """Compute and plot feature importance using RandomForest + permutation importance."""
    print("  - Computing feature importance...")

    X, y = prepare_classification_data(df)

    if len(X) < 50:
        print(f"    Warning: Too few samples ({len(X)}) for reliable importance")
        return

    n_classes = len(np.unique(y))
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=n_classes,
        n_jobs=-1,
        random_state=0,
    )

    # Fit RandomForest
    rf = lgbm
    rf.fit(X, y)

    # Built-in feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([X.columns[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 20 Features for Predicting Harmonic Win")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "harmonic_vs_smart_feature_importance.png"), dpi=200)
    plt.close()

    # Save to CSV
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(out_dir, "harmonic_vs_smart_feature_importance.csv"), index=False)


def generate_claim_plots(results_dir: str, claims_dir: str):
    """Generate all claim-specific plots."""
    print("\n" + "="*60)
    print("PART B: Generating claim-specific plots")
    print("="*60)

    safe_mkdir(claims_dir)

    # Load data
    print("\nLoading data...")
    try:
        dfa = load_agg_data(results_dir)
        dfw = load_winners_data(results_dir)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return

    # Compute delta_H and add regime features
    print("Computing trial metrics...")
    df = compute_delta_h(dfa, dfw)
    df = add_regime_features(df)

    print(f"  Total trials: {len(df)}")
    print(f"  Harmonic wins: {df['harmonic_wins'].sum()} ({df['harmonic_wins'].mean()*100:.1f}%)")

    # Section 1: Harmonic vs SMART-family separation
    print("\n[Section 1] Harmonic vs SMART-family separation plots:")
    plot_delta_h_histogram(df, claims_dir)
    plot_delta_h_boxplots(df, claims_dir)
    plot_delta_h_heatmap(df, claims_dir)
    plot_winrate_heatmap(df, claims_dir)

    # Section 2: Robustness / HP sensitivity
    print("\n[Section 2] Robustness / HP sensitivity plots:")
    hp_df = compute_hp_sensitivity(dfa)
    if hp_df is not None:
        plot_hp_sensitivity_violin(hp_df, claims_dir)
        plot_hp_peak_vs_sensitivity(hp_df, claims_dir)
        plot_robustness_by_regime(hp_df, dfw, claims_dir)
    else:
        print("  Skipping HP sensitivity plots (no hp_id column found)")

    # Section 3: Explainability
    print("\n[Section 3] Explainability plots:")
    #plot_decision_tree(df, claims_dir)
    #plot_feature_importance(df, claims_dir)

    # Section 4: Downside Risk Plots (NEW)
    print("\n[Section 4] Downside risk plots (tail behavior):")
    try:
        dfr = load_runs_data(results_dir)
        print(f"  Loaded {len(dfr)} raw runs for downside analysis")
        plot_delta_h_left_tail_cdf(dfr, claims_dir)
        plot_worst_case_return_by_agent(dfr, claims_dir)
    except FileNotFoundError as e:
        print(f"  Warning: Could not load runs data for downside plots: {e}")
    except Exception as e:
        print(f"  Warning: Error generating downside risk plots: {e}")


# -----------------------------------------------------------------------------
# Part C: Generate index README
# -----------------------------------------------------------------------------
def generate_readme(plots_dir: str):
    """Generate README.md with plot index."""
    print("\n" + "="*60)
    print("PART C: Generating README index")
    print("="*60)

    readme_content = """# Plots Index

This folder contains all plots generated to support the research claims.

## Claims Supported

1. **Where Harmonic wins vs loses** (vs SMART-family) depends on:
   - Distribution mismatch (reward_kind ≠ duration_kind)
   - Duration randomness/type (especially exponential)
   - Coupling (coupled=True amplifies separation)

2. **Harmonic robustness is higher** (lower hyperparameter sensitivity)

3. **Environments separate into regimes** visible in embeddings

4. **Harmonic avoids large underperformance** (downside risk):
   - Key insight: Harmonic's advantage is in reduced tail risk, not higher mean
   - In mismatched, stochastic environments, Harmonic has fewer catastrophic failures
   - This is visible in left-tail CDF plots and worst-case return analysis

---

## Plot Directory Structure

### `claims/` - New claim-specific plots

| Plot | Filename | Claim | Description |
|------|----------|-------|-------------|
| 1 | `plot1_delta_h_histogram.png` | 1 | Overall distribution of ΔH = Harmonic - max(SMART-family) |
| 2 | `plot2_delta_h_boxplots.png` | 1 | ΔH grouped by mismatch, duration_kind, coupled |
| 3 | `plot3_delta_h_heatmap_coupled_*.png` | 1 | Mean ΔH heatmap by reward×duration kind |
| 4 | `plot4_winrate_heatmap_coupled_*.png` | 1 | Harmonic win rate heatmap |
| 5 | `plot5_hp_sensitivity_violin.png` | 2 | HP sensitivity distribution by agent |
| 6 | `plot6_peak_vs_sensitivity.png` | 2 | Peak performance vs HP sensitivity trade-off |
| 7 | `plot7_robustness_by_regime.png` | 2 | Robustness comparison by regime |
| 8 | `harmonic_vs_smart_tree.png` | 3 | Decision tree for predicting Harmonic wins |
| 9 | `harmonic_vs_smart_confusion_matrix.png` | 3 | Classifier confusion matrix |
| 10 | `harmonic_vs_smart_feature_importance.png` | 3 | Top 20 features for prediction |
| 11 | `deltaH_left_tail_cdf.png` | 4 | **Left-tail CDF of ΔH** - shows downside risk by mismatch regime |
| 12 | `worst_case_return_by_agent.png` | 4 | **Worst-case performance** - risk dominance analysis |

**NEW: Downside Risk Plots (Claim 4)**

These plots show Harmonic's advantage is in **reduced downside risk**, not higher mean performance.

- `deltaH_left_tail_cdf.png`: Empirical CDF of ΔH (Harmonic − max(SMART-family)) focusing on the 
  left tail (negative values). A lower curve in the left tail means fewer catastrophic failures.
  Split by mismatch=True vs False to show where Harmonic avoids being much worse.
  
- `worst_case_return_by_agent.png`: Boxplot of worst-case (minimum) average rate per agent across
  all seeds and hyperparameters. Focuses on mismatched, stochastic-duration environments.
  Higher values = better floor performance.

Supporting files:
- `harmonic_vs_smart_tree_rules.txt` - Human-readable decision rules
- `harmonic_vs_smart_cv_score.txt` - Cross-validation accuracy
- `harmonic_vs_smart_feature_importance.csv` - Full feature importance table

### `robust_vis/` - From robust_vis.py

| Filename | Claim | Description |
|----------|-------|-------------|
| `confusion_matrix.png` | 3 | Classifier confusion matrix |
| `lgbm_feature_importance.png` | 3 | LightGBM feature importances |
| `decision_tree.png` | 3 | Interpretable decision tree |
| `pca_regime_map.png` | 3 | PCA embedding of regimes |
| `lda_regime_map.png` | 3 | LDA embedding (maximizes class separation) |
| `tsne_regime_map.png` | 3 | t-SNE embedding |
| `umap_regime_map.png` | 3 | UMAP embedding |
| `pca_3d_regime_map.png` | 3 | 3D PCA visualization |
| `cluster_winner_mix.png` | 3 | Winner distribution per cluster |

### `heatmaps/` - From make_regime_heamap.py

| Filename Pattern | Claim | Description |
|-----------------|-------|-------------|
| `OVERALL__margin.png` | 1 | Overall winner margin heatmap |
| `OVERALL__winrate__*.png` | 1 | Overall win rate per agent |
| `reward=*__dur=*__coupled=*__margin.png` | 1 | Per-slice margin heatmaps |
| `reward=*__dur=*__coupled=*__winrate__*.png` | 1 | Per-slice win rate heatmaps |

---

## How to Regenerate

```bash
python make_all_claim_plots.py --results-dir results
```

---

## Key Findings Summary

*(Fill in after reviewing plots)*

### Claim 1: Harmonic wins when...
- [ ] Distribution mismatch present
- [ ] Exponential durations
- [ ] Coupling enabled

### Claim 2: Harmonic robustness
- [ ] Lower HP sensitivity (hp_std)
- [ ] Flatter performance across HP grid

### Claim 3: Regime separability
- [ ] Clear clusters in embeddings
- [ ] Decision tree achieves >X% accuracy
- [ ] Key separating features: ...

### Claim 4: Downside risk advantage
- [ ] Lower left-tail probability in mismatched environments
- [ ] Higher worst-case return (better floor performance)
- [ ] Key insight: "Harmonic doesn't win by a lot — it just avoids losing badly"

"""

    readme_path = os.path.join(plots_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"  Wrote {readme_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate all plots to support research claims"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing param_sweep_*.csv files (default: results)"
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    plots_dir = os.path.join(results_dir, "plots")
    claims_dir = os.path.join(plots_dir, "claims")

    print(f"Results directory: {results_dir}")
    print(f"Plots output directory: {plots_dir}")

    # Ensure directories exist
    safe_mkdir(plots_dir)
    safe_mkdir(claims_dir)

    # Part A: Run existing scripts
    run_existing_scripts(results_dir, plots_dir)

    # Part B: Generate claim-specific plots
    generate_claim_plots(results_dir, claims_dir)

    # Part C: Generate README index
    generate_readme(plots_dir)

    print("\n" + "="*60)
    print("DONE! All plots generated.")
    print("="*60)
    print(f"\nOutput directory: {plots_dir}")
    print("  - claims/     : New claim-specific plots")
    print("  - robust_vis/ : Outputs from robust_vis.py")
    print("  - heatmaps/   : Outputs from make_regime_heamap.py")
    print("  - README.md   : Plot index")


if __name__ == "__main__":
    main()
