#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
WINNERS_CSV = os.path.join(RESULTS_DIR, "param_sweep_winners.csv")
AGG_CSV = os.path.join(RESULTS_DIR, "param_sweep_agg.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Regime binning helpers
# -----------------------------
def bin_stationarity(x: float) -> str:
    # interpret: low stationarity => highly nonstationary
    if x is None or not np.isfinite(x):
        return "unk"
    if x < 0.33:
        return "low"
    if x < 0.67:
        return "mid"
    return "high"


def bin_volatility(x: float) -> str:
    # works for both your old wide sweep and the narrower “structured” sweep
    if x is None or not np.isfinite(x):
        return "unk"
    if x < 0.2:
        return "low"
    if x < 0.8:
        return "mid"
    return "high"


def safe_get(d: dict, k: str, default=None):
    v = d.get(k, default)
    return default if v is None else v


# -----------------------------
# Load + join
# -----------------------------
def load_join():
    dfw = pd.read_csv(WINNERS_CSV)
    dfa = pd.read_csv(AGG_CSV)

    # Decode params_json from winners (1 row per trial)
    params = dfw["params_json"].apply(json.loads)
    dfp = pd.json_normalize(params)

    # Merge params into winners
    dfw2 = pd.concat([dfw.drop(columns=["params_json"]), dfp], axis=1)

    # Pivot agg -> one row per trial with each agent's avg_rate_mean
    # columns: avg_rate_mean__<agent_name>
    # First, take the best (max avg_rate_mean) hp_id per (trial_id, agent)
    best_hp = dfa.loc[dfa.groupby(["trial_id", "agent"])["avg_rate_mean"].idxmax()]
    pivot = best_hp.pivot(index="trial_id", columns="agent", values="avg_rate_mean")
    pivot.columns = [f"avg_rate_mean__{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Merge winners with per-agent avg_rate_mean
    df = dfw2.merge(pivot, on="trial_id", how="left")

    return df


# -----------------------------
# Compute margin (winner vs runner-up)
# -----------------------------
def compute_margin(df: pd.DataFrame) -> pd.DataFrame:
    # winner and runner_up exist in df
    # Use the pivoted avg_rate_mean columns to compute the margin
    def get_score(row, agent_name):
        if agent_name is None or (not isinstance(agent_name, str)):
            return np.nan
        col = f"avg_rate_mean__{agent_name}"
        return row[col] if col in row and pd.notna(row[col]) else np.nan

    winner_score = []
    runner_score = []
    margin = []

    for _, r in df.iterrows():
        ws = get_score(r, r.get("winner"))
        rs = get_score(r, r.get("runner_up"))
        winner_score.append(ws)
        runner_score.append(rs)
        margin.append(ws - rs if (pd.notna(ws) and pd.notna(rs)) else np.nan)

    df = df.copy()
    df["winner_avg_rate_mean_recomputed"] = winner_score
    df["runner_up_avg_rate_mean_recomputed"] = runner_score
    df["margin_vs_runner"] = margin
    return df


# -----------------------------
# Plotting utilities
# -----------------------------
ORDER_ST = ["low", "mid", "high"]
ORDER_VOL = ["low", "mid", "high"]


def make_heatmap_grid(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    outpath: str,
    vmin=None,
    vmax=None,
):
    # pivot to 3x3
    piv = (
        data.pivot_table(
            index="vol_bin",
            columns="st_bin",
            values=value_col,
            aggfunc="mean",
        )
        .reindex(index=ORDER_VOL, columns=ORDER_ST)
    )

    mat = piv.values.astype(float)

    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im)

    plt.xticks(range(len(ORDER_ST)), ORDER_ST)
    plt.yticks(range(len(ORDER_VOL)), ORDER_VOL)
    plt.xlabel("Nonstationarity (stationarity bin)")
    plt.ylabel("Volatility bin")

    # annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def slice_key(row) -> str:
    return f"reward={row['reward_kind']}__dur={row['duration_kind']}__coupled={bool(row['coupled'])}"


# -----------------------------
# Main heatmap generation
# -----------------------------
def main():
    global RESULTS_DIR, WINNERS_CSV, AGG_CSV, OUT_DIR

    parser = argparse.ArgumentParser(description="Generate regime heatmaps from parameter-sweep CSVs")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    RESULTS_DIR = os.path.abspath(args.results_dir)
    WINNERS_CSV = os.path.join(RESULTS_DIR, "param_sweep_winners.csv")
    AGG_CSV = os.path.join(RESULTS_DIR, "param_sweep_agg.csv")
    OUT_DIR = os.path.join(RESULTS_DIR, "heatmaps")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_join()
    df = compute_margin(df)

    # Pick stationarity/volatility signals (reward-side; adjust if you prefer duration-side)
    df["st_a"] = pd.to_numeric(df.get("reward_stationarity_a"), errors="coerce")
    df["st_b"] = pd.to_numeric(df.get("reward_stationarity_b"), errors="coerce")
    df["st_mean"] = df[["st_a", "st_b"]].mean(axis=1)

    df["vol"] = pd.to_numeric(df.get("max_volatility"), errors="coerce")

    df["st_bin"] = df["st_mean"].apply(lambda x: bin_stationarity(x))
    df["vol_bin"] = df["vol"].apply(lambda x: bin_volatility(x))

    # remove unknown bins
    df = df[(df["st_bin"] != "unk") & (df["vol_bin"] != "unk")].copy()

    # for each slice (reward_kind, duration_kind, coupled) produce:
    # - 3 winrate heatmaps (one per agent)
    # - 1 margin heatmap (mean winner margin vs runner-up)
    slices = df.groupby(["reward_kind", "duration_kind", "coupled"], dropna=False)

    rate_prefix = "avg_rate_mean__"
    agents = sorted(
        column[len(rate_prefix):]
        for column in df.columns
        if column.startswith(rate_prefix)
    )

    for (rk, dk, coup), g in slices:
        if len(g) < 30:
            # avoid tiny slices (too noisy)
            continue

        base = f"reward={rk}__dur={dk}__coupled={bool(coup)}"
        g = g.copy()

        # Margin heatmap
        make_heatmap_grid(
            g,
            value_col="margin_vs_runner",
            title=f"Mean winner margin vs runner-up\n{base} (avg_rate_mean)",
            outpath=os.path.join(OUT_DIR, f"{base}__margin.png"),
        )

        # Winrate heatmaps per agent
        for a in agents:
            g[f"win_{a}"] = (g["winner"] == a).astype(float)
            make_heatmap_grid(
                g,
                value_col=f"win_{a}",
                title=f"Win-rate of {a}\n{base}",
                outpath=os.path.join(OUT_DIR, f"{base}__winrate__{a.replace(' ', '_')}.png"),
                vmin=0.0,
                vmax=1.0,
            )

    # Also produce an overall (no slice) set of heatmaps
    overall = df.copy()
    make_heatmap_grid(
        overall,
        value_col="margin_vs_runner",
        title="Overall mean winner margin vs runner-up (avg_rate_mean)",
        outpath=os.path.join(OUT_DIR, "OVERALL__margin.png"),
    )
    for a in agents:
        overall[f"win_{a}"] = (overall["winner"] == a).astype(float)
        make_heatmap_grid(
            overall,
            value_col=f"win_{a}",
            title=f"Overall win-rate of {a}",
            outpath=os.path.join(OUT_DIR, f"OVERALL__winrate__{a.replace(' ', '_')}.png"),
            vmin=0.0,
            vmax=1.0,
        )

    print(f"Wrote heatmaps to: {OUT_DIR}")


if __name__ == "__main__":
    main()
