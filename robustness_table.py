#!/usr/bin/env python3
"""
robustness_table.py

Sweeps over (exploration_rate, learning_rate, rho_learning_rate) for
Harmonic, WeightedHarmonic, RelaxedSMART, and SMART, and measures what
fraction of parameter combinations converge to the correct policy.

Produces per-agent, per-env:
  - robustness_bars_<agent>_<env>.png  : marginal bar-charts (ε / α / β)

Produces across all envs (when > 1 env):
  - robustness_3d_<agent>.png  : one 3-D chart per agent (env × hyperparam)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, cast

import numpy as np
import pandas as pd
import matplotlib
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from agents.harmonic_r import Harmonic, WeightedHarmonic
from agents.relaxed_smart import RelaxedSMART
from agents.smart_r import SMART
from agents.oracle import Oracle
from main import train_agent_with_tracking
from more_smdp_envs import SMDPConfigFactory
from run_smdp_experiment import get_greedy_policy
from smdp_env import SMDPEnvironment

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = "robustness_results"
INTERMEDIATE_DIR = os.path.join(OUT_DIR, "intermediate")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# ── sweep grid ──────────────────────────────────────────────────────────────
N = 20
ER_VALUES   = [0.2]   # exploration rate
LR_VALUES   = np.round(np.logspace(-4, -1,  N), 6)   # learning rate
BETA_VALUES = np.round(np.logspace(-4, -1,  N), 6)   # rho learning rate

NUM_EPISODES          = 4
MAX_STEPS_PER_EPISODE = 1000
NUM_WORKERS           = 28

CORRECT_ACTION = 0  # GalK I think should be 0.
TARGET_STATE   = "s1"

# ── colour map ──────────────────────────────────────────────────────────────
CMAP = LinearSegmentedColormap.from_list(
    "robust", ["#d73027", "#fee090", "#1a9850"], N=256
)

# ── agents to sweep ─────────────────────────────────────────────────────────
AGENT_CLASSES = {
    "Harmonic":          Harmonic,
    "Weighted Harmonic": WeightedHarmonic,
    "Relaxed SMART":     RelaxedSMART,
    "SMART":             SMART,
}

# ── hyperparameter display names ─────────────────────────────────────────────
PARAM_LABELS = {
    "er":   "Exploration rate (ε)",
    "lr":   "Learning rate (α)",
    "beta": "ρ learning rate (β)",
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _fmt_val(v: float) -> str:
    """Format a hyperparameter value in compact scientific/engineering notation."""
    if v == 0:
        return "0"
    if v < 0.01:
        exp = int(np.floor(np.log10(abs(v))))
        mantissa = v / (10 ** exp)
        if abs(mantissa - round(mantissa)) < 1e-9:
            return f"1e{exp}"
        return f"{mantissa:.2g}e{exp}"
    if v < 1:
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s
    return f"{v:.4g}"


# ────────────────────────────────────────────────────────────────────────────
# Core sweep – parallelised with ThreadPoolExecutor
# ────────────────────────────────────────────────────────────────────────────

def _run_single(task) -> dict:
    """Evaluate one (er, lr, beta) combination. Designed for thread-pool use."""
    er, lr, beta, AgentClass, agent_name, cfg_name, cfg = task
    env = SMDPEnvironment(cfg)          # each thread gets its own env instance
    action_space = env.action_space

    agent = AgentClass(
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

    return dict(
        cfg=cfg_name,
        agent=agent_name,
        er=float(er),
        lr=float(lr),
        beta=float(beta),
        chosen_action=chosen,
        correct=correct,
        avg_rate=avg_rate,
        rho=res["rho"],
    )


def run_sweep(cfg, cfg_name: str,
              agent_name: str, AgentClass) -> pd.DataFrame:
    """Run the full (er × lr × beta) sweep for one agent using a thread pool."""
    tasks = [
        (er, lr, beta, AgentClass, agent_name, cfg_name, cfg)
        for er in ER_VALUES
        for lr in LR_VALUES
        for beta in BETA_VALUES
    ]
    rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(_run_single, t): t for t in tasks}
        for fut in as_completed(futures):
            rows.append(fut.result())

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 2-D marginal bar plot (one env, one agent)
# ────────────────────────────────────────────────────────────────────────────

def plot_marginal_bars(df: pd.DataFrame, cfg_name: str, agent_name: str) -> None:
    """
    Three bar-charts side by side: % correct as a function of ε, α, β individually
    (marginalised over the other two).  X-axis labels use compact sci notation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    params = [
        ("er",   PARAM_LABELS["er"],   axes[0]),
        ("lr",   PARAM_LABELS["lr"],   axes[1]),
        ("beta", PARAM_LABELS["beta"], axes[2]),
    ]
    for col, label, ax in params:
        grp = df.groupby(col)["correct"].mean().mul(100)
        colours = [
            "#1a9850" if v >= 80 else "#fee090" if v >= 50 else "#d73027"
            for v in grp.values
        ]
        ax.bar(range(len(grp)), grp.values, color=colours,
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(
            [_fmt_val(v) for v in grp.index],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 108)
        ax.set_ylabel("% Correct Policy", fontsize=9)
        ax.set_xlabel(label, fontsize=9)
        ax.axhline(100, color="#888", lw=0.8, ls="--")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        for i, v in enumerate(grp.values):
            ax.text(i, v + 1.5, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"{agent_name} – Marginal Robustness per Hyperparameter\n"
        f"Env: {cfg_name}",
        fontsize=11,
    )
    slug_cfg   = cfg_name.lower().replace(" ", "_")
    slug_agent = agent_name.lower().replace(" ", "_")
    path = os.path.join(INTERMEDIATE_DIR, f"robustness_bars_{slug_agent}_{slug_cfg}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Multi-env plot – grouped heatmap rows: one heatmap per hyperparameter,
# columns = param values, rows = environments.  Replaces the confusing 3-D.
# ────────────────────────────────────────────────────────────────────────────

def plot_multi_env_grouped_3d(all_df: pd.DataFrame, agent_name: str) -> None:
    """
    One 3-D grouped bar chart per agent.

      X axis = hyperparameter values, grouped by param (ε | α | β) with a gap
               between groups — so you read it just like the 2-D bar charts
      Y axis = environment (log-scale variant)
      Z axis = % correct policy  (colour-coded red→yellow→green)

    Every bar still shows the per-value breakdown, but now the environment
    dimension is the depth axis so cross-env comparisons are immediate.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType

    cfg_names  = sorted(all_df["cfg"].unique())
    param_keys = ["er", "lr", "beta"]
    param_labels_short = ["ε", "α", "β"]
    GAP = 1.5          # extra spacing between param groups on X axis

    n_envs = len(cfg_names)
    norm   = plt.Normalize(vmin=0, vmax=100)

    # ── collect (x_pos, label, param_key, param_val) for every bar column ───
    bar_specs: list[tuple[float, str, str, float]] = []
    group_centre_x: list[float] = []
    x = 0.0
    for ki, col in enumerate(param_keys):
        vals = sorted(all_df[col].unique())
        start_x = x
        for v in vals:
            bar_specs.append((x, _fmt_val(v), col, v))
            x += 1.0
        group_centre_x.append((start_x + x - 1) / 2.0)
        x += GAP   # gap between groups

    # ── short env labels ─────────────────────────────────────────────────────
    short_names = [
        cfg.split("_logscale_")[-1] if "_logscale_" in cfg else cfg
        for cfg in cfg_names
    ]

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(12.0, len(bar_specs) * 0.7 + 3.0),
                               max(6.0,  n_envs * 1.2 + 2.0)))
    ax = cast(Axes3DType, fig.add_subplot(111, projection="3d"))

    bar_w = 0.55   # narrower → less occlusion
    bar_d = 0.5

    for yi, (cfg, short) in enumerate(zip(cfg_names, short_names)):
        sub_cfg = all_df[all_df["cfg"] == cfg]
        for (xpos, lbl, col, val) in bar_specs:
            sub = sub_cfg[sub_cfg[col] == val]
            z = sub["correct"].mean() * 100 if len(sub) > 0 else 0.0
            colour = CMAP(norm(z))
            ax.bar3d(
                xpos - bar_w / 2,
                yi   - bar_d / 2,
                0,
                bar_w, bar_d, max(z, 0.5),
                color=colour, alpha=0.82, shade=True,
            )
            # annotate top of each bar
            if z >= 1:
                ax.text(xpos, yi, z + 2, f"{z:.0f}",
                        ha="center", va="bottom", fontsize=5.5,
                        color="#111", zorder=5)

    # ── X ticks: param values ─────────────────────────────────────────────
    ax.set_xticks([s[0] for s in bar_specs])
    ax.set_xticklabels([s[1] for s in bar_specs], rotation=40, ha="right", fontsize=6)

    # ── Y ticks: environments ─────────────────────────────────────────────
    ax.set_yticks(range(n_envs))
    ax.set_yticklabels(short_names, fontsize=7)

    ax.set_zlim(0, 115)
    ax.set_zlabel("% Correct", fontsize=8, labelpad=4)
    ax.set_ylabel("Environment (logscale)", fontsize=8, labelpad=12)
    # angle: enough elevation to see top annotations, azim so Y fans right→left
    ax.view_init(elev=32, azim=-60)

    # ── param-group labels floating above the X axis ──────────────────────
    z_top = 117
    for label, xc in zip(param_labels_short, group_centre_x):
        ax.text(xc, -0.8, z_top, label,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#222")

    # ── vertical divider planes between groups ────────────────────────────
    x_cursor = 0.0
    for ki, col in enumerate(param_keys[:-1]):
        x_cursor += len(sorted(all_df[col].unique()))
        div_x = x_cursor + GAP / 2 - 1.0
        ys = np.array([0, n_envs - 1])
        zs = np.array([0, 115])
        YY, ZZ = np.meshgrid(ys, zs)
        ax.plot_surface(
            np.full_like(YY, div_x, dtype=float), YY, ZZ,
            alpha=0.08, color="grey",
        )
        x_cursor += GAP

    # ── colorbar ─────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.45, aspect=14, pad=0.06,
                 label="% Correct Policy")

    fig.suptitle(
        f"{agent_name} – Robustness: all hyperparameters × environments",
        fontsize=11,
    )

    slug_agent = agent_name.lower().replace(" ", "_")
    path = os.path.join(INTERMEDIATE_DIR, f"robustness_multienv_{slug_agent}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Clear multi-environment visualizations (recommended)
# =============================================================================

def plot_multi_env_3d_small_multiples(all_df: pd.DataFrame, agent_name: str) -> None:
    """
    Clearer alternative to the single grouped 3D chart:

      - 3 subplots (ε / α / β), each showing env (Y) × param-value (X) × % (Z)
      - reduced occlusion (narrower bars + higher elevation)
      - optional orthographic projection (less perspective distortion)
      - shading disabled (color encodes value without lighting artifacts)

    Output:
      robustness_multienv_<agent>_3d.png
    """
    from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType  # noqa: F401

    cfg_names  = sorted(all_df["cfg"].unique())
    n_envs     = len(cfg_names)

    # ── short env labels ─────────────────────────────────────────────────────
    short_names = [
        cfg.split("_logscale_")[-1] if "_logscale_" in cfg else cfg
        for cfg in cfg_names
    ]

    param_keys   = ["er", "lr", "beta"]
    x_labels     = [
        "Exploration rate (ε)",
        "Learning rate (α)",
        "ρ learning rate (β)",
    ]
    titles_short = ["ε", "α", "β"]

    norm = plt.Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=(16.5, 5.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.06], wspace=0.25)
    axes: list[Axes3DType] = []

    for i, (pkey, xlabel, ttl) in enumerate(zip(param_keys, x_labels, titles_short)):
        ax = cast(Axes3DType, fig.add_subplot(gs[0, i], projection="3d"))
        axes.append(ax)

        # Orthographic projection (when supported) reduces perspective distortion.
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

        vals = sorted(all_df[pkey].unique())
        xs = np.arange(len(vals), dtype=float)
        ys = np.arange(n_envs, dtype=float)

        dx = 0.70
        dy = 0.55

        for yi, cfg in enumerate(cfg_names):
            sub_cfg = all_df[all_df["cfg"] == cfg]
            for xi, v in enumerate(vals):
                sub = sub_cfg[sub_cfg[pkey] == v]
                z = sub["correct"].mean() * 100 if len(sub) > 0 else 0.0
                color = CMAP(norm(z))
                ax.bar3d(
                    xi - dx / 2, yi - dy / 2, 0,
                    dx, dy, max(z, 0.2),
                    color=color, alpha=0.95, shade=False,
                )
                # annotate only "meaningful" bars to reduce clutter
                if z >= 60:
                    ax.text(xi, yi, z + 2, f"{z:.0f}%",
                            ha="center", va="bottom", fontsize=7, color="#111")

        ax.set_title(ttl, fontsize=16, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=10)
        ax.set_ylabel("Env (logscale)", fontsize=10, labelpad=12)
        ax.set_zlabel("% Correct", fontsize=10, labelpad=6)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in vals], rotation=30, ha="right", fontsize=9)

        ax.set_yticks(ys)
        ax.set_yticklabels(short_names, fontsize=9)

        ax.set_zlim(0, 110)
        ax.zaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.zaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

        # cleaner panes
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            try:
                pane.set_alpha(0.0)
            except Exception:
                pass

        # A higher elevation reduces overlap; azim keeps env axis readable
        ax.view_init(elev=52, azim=-58)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(sm, cax=cax, label="% Correct Policy")

    fig.suptitle(f"{agent_name} – Robustness across environments (3-D small multiples)", fontsize=18)
    out_path = os.path.join(INTERMEDIATE_DIR, f"robustness_multienv_{agent_name.lower()}_3d.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_multi_env_heatmap_small_multiples(all_df: pd.DataFrame, agent_name: str) -> None:
    """
    Most readable alternative (2D): env × value heatmaps, one per hyperparameter.

    Output:
      robustness_multienv_<agent>_heatmap.png
    """
    cfg_names  = sorted(all_df["cfg"].unique())
    n_envs     = len(cfg_names)

    short_names = [
        cfg.split("_logscale_")[-1] if "_logscale_" in cfg else cfg
        for cfg in cfg_names
    ]

    param_keys   = ["er", "lr", "beta"]
    titles_short = ["ε", "α", "β"]
    x_tick_names = [
        [f"{v:g}" for v in sorted(all_df["er"].unique())],
        [f"{v:g}" for v in sorted(all_df["lr"].unique())],
        [f"{v:g}" for v in sorted(all_df["beta"].unique())],
    ]

    norm = plt.Normalize(vmin=0, vmax=100)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw={"wspace": 0.18})
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, (ax, pkey, ttl, xticks) in enumerate(zip(axes, param_keys, titles_short, x_tick_names)):
        vals = sorted(all_df[pkey].unique())
        mat = np.zeros((n_envs, len(vals)), dtype=float)

        for yi, cfg in enumerate(cfg_names):
            sub_cfg = all_df[all_df["cfg"] == cfg]
            for xi, v in enumerate(vals):
                sub = sub_cfg[sub_cfg[pkey] == v]
                z = sub["correct"].mean() * 100 if len(sub) > 0 else 0.0
                mat[yi, xi] = z

        im = ax.imshow(mat, cmap=CMAP, norm=norm, aspect="auto")

        ax.set_title(ttl, fontsize=16, fontweight="bold", pad=10)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(xticks, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_envs))
        ax.set_yticklabels(short_names, fontsize=9)

        # light gridlines between cells
        ax.set_xticks(np.arange(-.5, len(vals), 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_envs, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotate all cells (small numbers, but still readable in 2D)
        for yi in range(n_envs):
            for xi in range(len(vals)):
                z = mat[yi, xi]
                if z >= 1:
                    ax.text(xi, yi, f"{z:.0f}%", ha="center", va="center", fontsize=8, color="#111")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("% Correct Policy")

    fig.suptitle(f"{agent_name} – Robustness across environments (heatmaps)", fontsize=18)
    out_path = os.path.join(INTERMEDIATE_DIR, f"robustness_multienv_{agent_name.lower()}_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Envelope plot: min / median / max robustness across hyperparameters
#
# NOTE: Each sweep point is a single run (correct ∈ {0,1}), so taking min/median/max
# over the raw grid is not informative (min≈0, max≈100 almost always). Instead we
# summarize the *marginal robustness curves* (the same quantities shown in the 2-D
# marginal bar charts):
#   - For each env: compute %correct vs ε (marginalized over α,β)
#                 compute %correct vs α (marginalized over ε,β)
#                 compute %correct vs β (marginalized over ε,α)
#   - Concatenate those 3×N values and take min/median/max.
# This gives a compact “best / typical / worst” summary of sensitivity to HP choices.
# ────────────────────────────────────────────────────────────────────────────



def plot_alpha_beta_grouped(
    cfg_name: str,
    per_agent_df: Dict[str, pd.DataFrame],
) -> None:
    """
    One figure per environment.

    X-axis: (α, β) pairs  — all combinations in the sweep grid
    Groups: one clustered bar per agent.
    Y-axis: % Correct Policy

    The figure is saved to OUT_DIR/robustness_ab_<env>.png
    """
    agents = list(per_agent_df.keys())
    n_agents = len(agents)

    # Collect all (lr, beta) pairs that appear in the data
    all_dfs = pd.concat(per_agent_df.values(), ignore_index=True)
    lr_vals   = sorted(all_dfs["lr"].unique())
    beta_vals = sorted(all_dfs["beta"].unique())
    ab_pairs  = [(lr, b) for lr in lr_vals for b in beta_vals]
    n_pairs   = len(ab_pairs)

    # ── colours per agent ────────────────────────────────────────────────────
    default_colors = {
        "Harmonic": "#1a1a1a",
        "Weighted Harmonic": "#1b7837",
        "Relaxed SMART": "#fdb462",
        "SMART": "#ef3b2c",
    }
    agent_colors = {agent: default_colors.get(agent, "#377eb8") for agent in agents}

    group_w = 0.8                          # total width occupied by one group
    bar_w   = group_w / max(n_agents, 1)   # width of a single bar
    offsets = np.linspace(-group_w / 2 + bar_w / 2,
                           group_w / 2 - bar_w / 2, n_agents)

    x = np.arange(n_pairs, dtype=float)

    fig_w = max(12, n_pairs * 0.9 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6), constrained_layout=True)

    for i, agent in enumerate(agents):
        df = per_agent_df[agent]
        heights = []
        for (lr, b) in ab_pairs:
            sub = df[(df["lr"] == lr) & (df["beta"] == b)]
            pct = sub["correct"].mean() * 100 if len(sub) > 0 else 0.0
            heights.append(pct)

        bars = ax.bar(
            x + offsets[i],
            heights,
            width=bar_w * 0.92,
            label=agent,
            color=agent_colors[agent],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        # value label on top of each bar
        for rect, h in zip(bars, heights):
            if h >= 1:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + 1.5,
                    f"{h:.0f}%",
                    ha="center", va="bottom",
                    fontsize=7.5,
                    color="#222",
                )

    # ── X tick labels  (α=..., β=...) ───────────────────────────────────────
    tick_labels = [
        f"α={_fmt_val(lr)}\nβ={_fmt_val(b)}"
        for (lr, b) in ab_pairs
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=9)

    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("% Correct Policy", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    # ── vertical separators between α groups ─────────────────────────────────
    for k in range(1, len(lr_vals)):
        sep_x = k * len(beta_vals) - 0.5
        ax.axvline(sep_x, color="#aaa", lw=0.8, ls="--", zorder=1)

    # ── α group labels above the plot ────────────────────────────────────────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    group_centres = [
        (k * len(beta_vals) + (len(beta_vals) - 1) / 2)
        for k in range(len(lr_vals))
    ]
    ax2.set_xticks(group_centres)
    ax2.set_xticklabels(
        [f"α = {_fmt_val(lr)}" for lr in lr_vals],
        fontsize=11, fontweight="bold",
    )
    ax2.tick_params(length=0)

    # ── short env name for title ──────────────────────────────────────────────
    if "logscale_" in cfg_name:
        short_env = "logscale " + cfg_name.split("logscale_")[-1]
    else:
        short_env = cfg_name

    ax.set_title(
        f"Robustness by (α, β) — {short_env}",
        fontsize=14, pad=32,
    )
    ax.legend(ncols=n_agents, frameon=False, fontsize=12, loc="upper right")

    slug = cfg_name.lower().replace(" ", "_")
    out_path = os.path.join(INTERMEDIATE_DIR, f"robustness_ab_{slug}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_success_rate_vs_env(
    env_order: List[str],
    per_agent_all_df: Dict[str, pd.DataFrame],
) -> None:
    """
    Line plot: X = environment (ordered by increasing difficulty, log-scale),
               Y = % of (α, β) combinations that produced the correct policy.
    One line per algorithm.  ε is fixed so every row is one (α,β) combo.
    """
    agents = list(per_agent_all_df.keys())

    # ── colours / markers per agent ──────────────────────────────────────────
    styles = [
        dict(color="#1a1a1a", marker="o", ls="-",  lw=2.5),   # Harmonic
        dict(color="#1b7837", marker="D", ls=":",  lw=2.5),   # Weighted Harmonic
        dict(color="#e08c00", marker="s", ls="--", lw=2.5),   # Relaxed SMART
        dict(color="#ef3b2c", marker="^", ls="-.", lw=2.5),   # SMART
        dict(color="#31a354", marker="v", ls="-",  lw=2.5),
    ]

    # ── extract numeric x-values from env names ───────────────────────────
    has_logscale = any("logscale_" in env for env in env_order)
    x_values = []
    for env in env_order:
        if "logscale_" in env:
            try:
                x_values.append(float(env.split("logscale_")[-1]))
            except ValueError:
                x_values.append(float(len(x_values) + 1))
        else:
            x_values.append(float(len(x_values) + 1))
    x = np.array(x_values)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for i, agent in enumerate(agents):
        df = per_agent_all_df[agent]
        pct_values = []
        for env in env_order:
            df_env = df[df["cfg"] == env]
            if df_env.empty:
                pct_values.append(np.nan)
            else:
                pct_values.append(df_env["correct"].mean() * 100)

        sty = styles[i % len(styles)]
        ax.plot(
            x, pct_values,
            label=agent,
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["ls"],
            linewidth=sty["lw"],
            markersize=11,
            zorder=3,
        )

    if has_logscale:
        ax.set_xscale("log")
        ax.invert_xaxis()

    ax.tick_params(axis="x", labelrotation=90, labelsize=12)
    xlabel = "Environment Difficulty" + (" (log-scale)" if has_logscale else "")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("% of (α, β) combinations → correct policy", fontsize=14)
    ax.set_ylim(-5, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.grid(True, axis="x", alpha=0.15, zorder=0)
    ax.set_title(
        "Robustness vs. environment difficulty\n"
        "(% of hyperparameter combinations that converge to correct policy)",
        fontsize=15,
    )
    ax.legend(ncols=len(agents), frameon=False, fontsize=13, loc="upper right")

    out_path = os.path.join(OUT_DIR, "robustness_success_rate_vs_env.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

# Backwards-compatible name (old implementation):
plot_multi_env = plot_multi_env_grouped_3d


def main() -> None:
    smdp_factory = SMDPConfigFactory()
    configs = smdp_factory.get_all_configs()

    env_order = list(configs.keys())

    agent_all_dfs: Dict[str, List[pd.DataFrame]] = {k: [] for k in AGENT_CLASSES}

    for cfg_name, cfg in tqdm.tqdm(configs.items()):
        # per-agent DFs for this environment (used by plot_alpha_beta_grouped)
        env_agent_dfs: Dict[str, pd.DataFrame] = {}

        for agent_name, AgentClass in AGENT_CLASSES.items():
            df = run_sweep(cfg, cfg_name, agent_name, AgentClass)

            csv_path = os.path.join(
                INTERMEDIATE_DIR,
                f"raw_{agent_name.lower()}_{cfg_name.lower().replace(' ', '_')}.csv",
            )
            df.to_csv(csv_path, index=False)

            plot_marginal_bars(df, cfg_name, agent_name)
            agent_all_dfs[agent_name].append(df)
            env_agent_dfs[agent_name] = df

        # plot_alpha_beta_grouped(cfg_name, env_agent_dfs)

    # ── Multi-env heatmap (one per agent, only when multiple environments) ──
    per_agent_combined: Dict[str, pd.DataFrame] = {}
    for agent_name, dfs in agent_all_dfs.items():
        if len(dfs) > 1:
            combined = pd.concat(dfs, ignore_index=True)
            per_agent_combined[agent_name] = combined
            combined.to_csv(
                os.path.join(INTERMEDIATE_DIR, f"robustness_all_{agent_name.lower()}.csv"),
                index=False,
            )
            # plot_multi_env_3d_small_multiples(combined, agent_name)
            # plot_multi_env_heatmap_small_multiples(combined, agent_name)

    # ── Success-rate line plot across agents (only when multiple environments) ──
    if len(env_order) > 1 and per_agent_combined:
        plot_success_rate_vs_env(env_order, per_agent_combined)


if __name__ == "__main__":
    main()
