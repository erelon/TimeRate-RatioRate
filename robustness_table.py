#!/usr/bin/env python3
"""
robustness_table.py

Sweeps over (exploration_rate, learning_rate, rho_learning_rate) for HarmonicROLAgent
and measures what fraction of parameter combinations converge to the correct policy
(action 0 at state s1).

Produces:
  - robustness_table.csv        : raw results per parameter combination
  - robustness_heatmap_*.png    : 2-D heatmaps (% correct) for each fixed beta slice
  - robustness_summary_table.png: paper-ready summary table
"""

from __future__ import annotations

import os
import itertools
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from agents.harmonic_r import HarmonicROLAgent
from main import train_agent_with_tracking
from more_smdp_envs import SMDPConfigFactory
from run_smdp_experiment import get_greedy_policy
from smdp_env import SMDPEnvironment

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = "robustness_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── sweep grid ──────────────────────────────────────────────────────────────
ER_VALUES   = np.round(np.linspace(0.00, 0.50, 10), 4)   # exploration rate
LR_VALUES   = np.round(np.linspace(0.001, 0.30, 10), 4)  # learning rate
BETA_VALUES = np.round(np.linspace(0.001, 0.20, 10), 4)  # rho learning rate

# Training budget (kept small so the sweep is fast; increase for final runs)
NUM_EPISODES          = 10
MAX_STEPS_PER_EPISODE = 1000

CORRECT_ACTION = 0   # the ground-truth optimal action at s1
TARGET_STATE   = "s1"

# ── colour map ──────────────────────────────────────────────────────────────
CMAP = LinearSegmentedColormap.from_list(
    "robust", ["#d73027", "#fee090", "#1a9850"], N=256
)


# ────────────────────────────────────────────────────────────────────────────
# Core sweep
# ────────────────────────────────────────────────────────────────────────────

def run_sweep(env: SMDPEnvironment, cfg_name: str) -> pd.DataFrame:
    """Run the full (er × lr × beta) sweep and return a tidy DataFrame."""
    action_space = env.action_space
    rows: List[Dict] = []

    total = len(ER_VALUES) * len(LR_VALUES) * len(BETA_VALUES)
    done  = 0

    for er in ER_VALUES:
        for lr in LR_VALUES:
            for beta in BETA_VALUES:
                agent = HarmonicROLAgent(
                    name=f"er={er} lr={lr} beta={beta}",
                    action_space=action_space,
                    env=env,
                    learning_rate=float(lr),
                    exploration_rate=float(er),
                    rho_learning_rate=float(beta),
                    with_rho_trick=True,
                )
                res = train_agent_with_tracking(
                    env, agent,
                    num_episodes=NUM_EPISODES,
                    max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                )
                policy = get_greedy_policy(agent, env.states, action_space)
                chosen = policy.get(TARGET_STATE)
                correct = int(chosen == CORRECT_ACTION)
                avg_rate = (res["total_return"] / res["total_time"]
                            if res["total_time"] != 0 else 0.0)

                rows.append(dict(
                    cfg=cfg_name,
                    er=float(er),
                    lr=float(lr),
                    beta=float(beta),
                    chosen_action=chosen,
                    correct=correct,
                    avg_rate=avg_rate,
                    rho=res["rho"],
                ))

                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total} combinations done …")

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(df: pd.DataFrame, cfg_name: str) -> None:
    """
    For each value of beta, plot a heatmap of % correct over (er × lr).
    All heatmaps share the same colour scale [0, 100].
    """
    betas = sorted(df["beta"].unique())
    n_cols = min(5, len(betas))
    n_rows = int(np.ceil(len(betas) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    for ax_idx, beta in enumerate(betas):
        ax = axes[ax_idx]
        sub = df[df["beta"] == beta]
        pivot = (sub.groupby(["er", "lr"])["correct"]
                    .mean()
                    .mul(100)
                    .unstack("lr"))   # rows=er, cols=lr

        im = ax.imshow(
            pivot.values,
            vmin=0, vmax=100,
            cmap=CMAP,
            aspect="auto",
            origin="lower",
        )
        ax.set_title(f"β = {beta:.3f}", fontsize=9)
        ax.set_xlabel("Learning rate (α)", fontsize=8)
        ax.set_ylabel("Exploration rate (ε)", fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=6)

        # Annotate cells
        for (i, j), val in np.ndenumerate(pivot.values):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=5.5, color="white" if val < 50 else "black")

    # Hide unused axes
    for ax in axes[len(betas):]:
        ax.set_visible(False)

    # Shared colour-bar
    cbar = fig.colorbar(im, ax=axes[:len(betas)], fraction=0.02, pad=0.02)
    cbar.set_label("% runs with correct policy", fontsize=9)

    fig.suptitle(
        f"HarmonicROL – Robustness Heatmaps\n"
        f"Env: {cfg_name} | correct action at {TARGET_STATE} = {CORRECT_ACTION}",
        fontsize=11,
    )

    slug = cfg_name.lower().replace(" ", "_")
    path = os.path.join(OUT_DIR, f"robustness_heatmap_{slug}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def plot_summary_table(df: pd.DataFrame, cfg_name: str) -> None:
    """
    Aggregate over all parameter combinations and produce a paper-ready
    summary table image (overall % correct + breakdown by er / lr / beta bins).
    """
    overall_pct = df["correct"].mean() * 100

    # Bin each axis into Low / Mid / High thirds
    def thirds(col: pd.Series, label: str) -> pd.DataFrame:
        vals = sorted(col.unique())
        n = len(vals)
        low  = vals[:n//3]
        mid  = vals[n//3: 2*(n//3)]
        high = vals[2*(n//3):]
        bin_map = {v: "Low"  for v in low}
        bin_map.update({v: "Mid"  for v in mid})
        bin_map.update({v: "High" for v in high})
        binned = col.map(bin_map)
        return df.groupby(binned)["correct"].mean().mul(100).rename(label)

    er_pct   = thirds(df["er"],   "Exploration rate (ε)")
    lr_pct   = thirds(df["lr"],   "Learning rate (α)")
    beta_pct = thirds(df["beta"], "ρ learning rate (β)")

    # ── build a combined summary table ──
    rows = []
    for series in [er_pct, lr_pct, beta_pct]:
        param_name = series.name
        for bin_label in ["Low", "Mid", "High"]:
            val = series.get(bin_label, float("nan"))
            rows.append({
                "Parameter": param_name,
                "Range": bin_label,
                "% Correct Policy": f"{val:.1f}%",
            })

    table_df = pd.DataFrame(rows)

    # ── figure ──
    fig_h = 0.35 * (len(rows) + 3) + 1.0
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    ax.axis("off")

    # Title box
    fig.text(
        0.5, 0.97,
        "HarmonicROL — Parameter Robustness",
        ha="center", va="top",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.92,
        f"Environment: {cfg_name}\n"
        f"Correct policy: action {CORRECT_ACTION} at state '{TARGET_STATE}'\n"
        f"Overall: {overall_pct:.1f}% of {len(df)} parameter combinations "
        f"converge to the correct policy",
        ha="center", va="top",
        fontsize=9, color="#333333",
    )

    # Table
    col_labels = ["Parameter", "Range", "% Correct Policy"]
    table_data = table_df[col_labels].values.tolist()

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    # Colour header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Colour data rows: green if high, red if low
    for i, row in enumerate(table_data):
        try:
            pct = float(row[2].rstrip("%"))
        except ValueError:
            pct = 50.0
        colour = (
            "#c8e6c9" if pct >= 80
            else "#fff9c4" if pct >= 50
            else "#ffcdd2"
        )
        for j in range(len(col_labels)):
            tbl[(i + 1, j)].set_facecolor(colour)

    # Separate rows by parameter
    current_param = None
    for i, row in enumerate(table_data):
        if row[0] != current_param:
            current_param = row[0]
            for j in range(len(col_labels)):
                cell = tbl[(i + 1, j)]
                cell.set_edgecolor("#888888")
                # thicker top edge to separate groups
                # matplotlib Table doesn't expose per-edge widths directly;
                # we use visible separator rows instead.

    slug = cfg_name.lower().replace(" ", "_")
    path = os.path.join(OUT_DIR, f"robustness_summary_{slug}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def plot_marginal_bars(df: pd.DataFrame, cfg_name: str) -> None:
    """
    Three bar-charts side by side: % correct as a function of ε, α, β individually
    (marginalised over the other two).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    params = [
        ("er",   "Exploration rate (ε)", axes[0]),
        ("lr",   "Learning rate (α)",    axes[1]),
        ("beta", "ρ learning rate (β)",  axes[2]),
    ]
    for col, label, ax in params:
        grp = df.groupby(col)["correct"].mean().mul(100)
        colours = [
            "#1a9850" if v >= 80 else "#fee090" if v >= 50 else "#d73027"
            for v in grp.values
        ]
        bars = ax.bar(range(len(grp)), grp.values, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels([f"{v:.3f}" for v in grp.index], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% Correct Policy", fontsize=9)
        ax.set_xlabel(label, fontsize=9)
        ax.axhline(100, color="#888", lw=0.8, ls="--")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        # Annotate
        for i, v in enumerate(grp.values):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"HarmonicROL – Marginal Robustness per Hyperparameter\n"
        f"Env: {cfg_name}",
        fontsize=11,
    )
    slug = cfg_name.lower().replace(" ", "_")
    path = os.path.join(OUT_DIR, f"robustness_bars_{slug}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    smdp_factory = SMDPConfigFactory()
    configs = smdp_factory.get_all_configs()
    all_dfs: List[pd.DataFrame] = []

    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Environment: {cfg_name}")
        print(f"{'='*60}")
        env = SMDPEnvironment(cfg)

        df = run_sweep(env, cfg_name)
        csv_path = os.path.join(OUT_DIR, f"raw_{cfg_name.lower().replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  → raw data saved to {csv_path}")

        overall = df["correct"].mean() * 100
        print(f"\n  Overall % correct policy: {overall:.1f}%  "
              f"({df['correct'].sum()} / {len(df)} combinations)")

        # Per-parameter marginal summary (printed to console)
        for col, label in [("er", "ε"), ("lr", "α"), ("beta", "β")]:
            grp = df.groupby(col)["correct"].mean().mul(100)
            vals_str = "  ".join(f"{v:.3f}→{p:.0f}%" for v, p in grp.items())
            print(f"  {label}: {vals_str}")

        plot_heatmaps(df, cfg_name)
        plot_summary_table(df, cfg_name)
        plot_marginal_bars(df, cfg_name)
        all_dfs.append(df)

    # Combined CSV across all environments
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, "robustness_all.csv"), index=False)

        # ── Print a LaTeX-style table for the paper ──
        print("\n" + "="*60)
        print("LaTeX summary table (copy into your paper)")
        print("="*60)
        _print_latex_table(combined)


def _print_latex_table(df: pd.DataFrame) -> None:
    """Print a compact LaTeX tabular for the paper."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Robustness of HarmonicROL: percentage of hyperparameter",
        r"combinations that converge to the correct policy (action 0 at state $s_1$).}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Parameter & Range & \multicolumn{3}{c}{\% Correct} \\",
        r"\cmidrule(lr){3-5}",
        r" & & Low & Mid & High \\",
        r"\midrule",
    ]

    param_info = [
        ("er",   r"Exploration rate $\varepsilon$"),
        ("lr",   r"Learning rate $\alpha$"),
        ("beta", r"$\rho$ learning rate $\beta$"),
    ]

    for col, label in param_info:
        vals = sorted(df[col].unique())
        n = len(vals)
        low  = set(vals[:n//3])
        mid  = set(vals[n//3: 2*(n//3)])
        high = set(vals[2*(n//3):])

        def pct(subset):
            sub = df[df[col].isin(subset)]
            return sub["correct"].mean() * 100 if len(sub) > 0 else float("nan")

        lines.append(
            f"{label} & & {pct(low):.1f}\\% & {pct(mid):.1f}\\% & {pct(high):.1f}\\% \\\\"
        )

    lines += [
        r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Overall}}}} & "
        f"\\multicolumn{{3}}{{c}}{{{df['correct'].mean()*100:.1f}\\% "
        f"({df['correct'].sum()} / {len(df)} combinations)}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()

