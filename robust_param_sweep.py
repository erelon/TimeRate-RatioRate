#!/usr/bin/env python3
"""
robust_param_sweep.py

Randomized environment-parameter sweep using scikit-learn's ParameterSampler.

Finds cases where each of the 4 algorithms is better, aggregated by:
  1) reward/time ratio: avg_rate = total_return / total_time
  2) convergence stage: early / mid / late / no_convergence

Outputs (default under ./results/):
- param_sweep_runs.csv      : raw per (trial, seed, agent) metrics
- param_sweep_agg.csv       : per (trial, agent) aggregated metrics (mean/std)
- param_sweep_winners.csv   : per-trial winner + runner-ups
- report.md                 : human-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import uniform, loguniform
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import uniform, loguniform, randint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from dataclasses import replace
import numpy as np


@contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# -------------------------
# Convergence utilities
# -------------------------

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(float, copy=True)
    window = min(window, len(x))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="valid")


def compute_convergence_episode(
        episode_returns: List[float],
        *,
        threshold: float = 0.9,
        tail_frac: float = 0.1,
        smooth_window: int = 10,
) -> Optional[int]:
    """
    First episode where smoothed return reaches `threshold * final_level`.

    - final_level = mean return over the last `tail_frac` of episodes
    - returns a 1-based episode index, or None
    """
    if not episode_returns:
        return None

    y = np.asarray(episode_returns, dtype=float)
    n = len(y)
    tail_n = max(5, int(round(n * tail_frac)))
    final_level = float(np.mean(y[-tail_n:]))

    # If final level is near zero, convergence is not informative
    if abs(final_level) < 1e-12:
        return None

    y_sm = moving_average(y, smooth_window)
    target = threshold * final_level

    if final_level >= 0:
        meets = np.where(y_sm >= target)[0]
    else:
        # If final convergence is negative, being "close" means <= target
        meets = np.where(y_sm <= target)[0]

    if len(meets) == 0:
        return None

    return int(meets[0]) + 1  # 1-based


def convergence_stage(converged_at: Optional[int], num_episodes: int) -> str:
    if converged_at is None:
        return "no_convergence"
    frac = converged_at / float(max(1, num_episodes))
    if frac <= 1.0 / 3.0:
        return "early"
    if frac <= 2.0 / 3.0:
        return "mid"
    return "late"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# -------------------------
# Parameterization
# -------------------------

@dataclass(frozen=True)
class EnvParams:
    # -------------------------
    # Common knobs
    # -------------------------
    reward_kind: str  # e.g. 'constant'|'linear'|'normal'|'exp'|'gamma'|'uniform'|'mr_drift'|'mr_jump'
    duration_kind: str  # same set as reward_kind
    coupled: bool

    # optional “global” knobs some builders may use
    drift_op: str = "add"  # 'add' | 'mul'
    interval: int = 1  # generic interval (if you choose to use one)

    # -------------------------
    # Reward params (superset; only a subset is used depending on reward_kind)
    # -------------------------

    # constant
    reward_const_a: Optional[float] = None
    reward_const_b: Optional[float] = None

    # linear
    reward_linear_start_a: Optional[float] = None
    reward_linear_start_b: Optional[float] = None
    reward_linear_step_a: Optional[float] = None
    reward_linear_step_b: Optional[float] = None
    reward_linear_interval_a: Optional[int] = None
    reward_linear_interval_b: Optional[int] = None

    # normal (drifting mean) + legacy fields you already used
    reward_base_a: Optional[float] = None
    reward_gap: Optional[float] = None  # base_b = base_a + gap
    normal_stddev: Optional[float] = None
    mean_drift_a: Optional[float] = None
    mean_drift_b: Optional[float] = None

    # exponential (drifting mean)
    reward_exp_mean_a: Optional[float] = None
    reward_exp_mean_b: Optional[float] = None
    reward_exp_mean_drift_a: Optional[float] = None
    reward_exp_mean_drift_b: Optional[float] = None

    # gamma (drifting scale)
    reward_gamma_shape_a: Optional[float] = None
    reward_gamma_shape_b: Optional[float] = None
    reward_gamma_scale_a: Optional[float] = None
    reward_gamma_scale_b: Optional[float] = None
    reward_gamma_scale_drift_a: Optional[float] = None
    reward_gamma_scale_drift_b: Optional[float] = None

    # uniform (drifting bounds)
    reward_uniform_low_a: Optional[float] = None
    reward_uniform_width_a: Optional[float] = None  # high_a = low_a + width_a
    reward_uniform_low_b: Optional[float] = None
    reward_uniform_width_b: Optional[float] = None  # high_b = low_b + width_b
    reward_uniform_low_drift_a: Optional[float] = None
    reward_uniform_low_drift_b: Optional[float] = None
    reward_uniform_high_drift_a: Optional[float] = None
    reward_uniform_high_drift_b: Optional[float] = None

    # mean-reverting (drift/jump)
    reward_stationarity_a: Optional[float] = None
    reward_stationarity_b: Optional[float] = None
    reversion_speed: Optional[float] = None
    max_volatility: Optional[float] = None
    reward_ensure_positive: bool = False

    # jump params (only used if reward_kind == 'mr_jump')
    jump_low: Optional[float] = None
    jump_high: Optional[float] = None

    # -------------------------
    # Duration params (superset; only a subset is used depending on duration_kind)
    # -------------------------

    # constant (legacy)
    duration_const_a: Optional[float] = None
    duration_const_b: Optional[float] = None

    # linear
    duration_linear_start_a: Optional[float] = None
    duration_linear_start_b: Optional[float] = None
    duration_linear_step_a: Optional[float] = None
    duration_linear_step_b: Optional[float] = None
    duration_linear_interval_a: Optional[int] = None
    duration_linear_interval_b: Optional[int] = None

    # normal
    duration_normal_mean_a: Optional[float] = None
    duration_normal_mean_b: Optional[float] = None
    duration_normal_stddev: Optional[float] = None
    duration_normal_mean_drift_a: Optional[float] = None
    duration_normal_mean_drift_b: Optional[float] = None

    # exp (legacy)
    duration_mean_a: Optional[float] = None
    duration_mean_b: Optional[float] = None
    duration_exp_mean_drift_a: Optional[float] = None
    duration_exp_mean_drift_b: Optional[float] = None

    # gamma
    duration_gamma_shape_a: Optional[float] = None
    duration_gamma_shape_b: Optional[float] = None
    duration_gamma_scale_a: Optional[float] = None
    duration_gamma_scale_b: Optional[float] = None
    duration_gamma_scale_drift_a: Optional[float] = None
    duration_gamma_scale_drift_b: Optional[float] = None

    # uniform (legacy + per-state)
    duration_low: Optional[float] = None
    duration_width: Optional[float] = None  # high = low + width

    duration_uniform_low_a: Optional[float] = None
    duration_uniform_width_a: Optional[float] = None  # high_a = low_a + width_a
    duration_uniform_low_b: Optional[float] = None
    duration_uniform_width_b: Optional[float] = None  # high_b = low_b + width_b
    duration_uniform_low_drift_a: Optional[float] = None
    duration_uniform_low_drift_b: Optional[float] = None
    duration_uniform_high_drift_a: Optional[float] = None
    duration_uniform_high_drift_b: Optional[float] = None

    # mean-reverting (drift/jump) for durations
    duration_stationarity_a: Optional[float] = None
    duration_stationarity_b: Optional[float] = None
    duration_reversion_speed: Optional[float] = None
    duration_max_volatility: Optional[float] = None
    duration_ensure_positive: bool = True
    duration_jump_low: Optional[float] = None
    duration_jump_high: Optional[float] = None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def build_env_config(params: EnvParams, seed: int):
    """
    Builds an SMDPConfig using:
      - dist_factory.make_reward / make_duration
      - more_smdp_envs.non_stationary_simple_unichain2

    IMPORTANT: creates *fresh* reward/duration objects per call for fairness.
    """
    try:
        from dist_factory import make_reward, make_duration
        from more_smdp_envs import non_stationary_simple_unichain2

        def _key(kind: str) -> str:
            return kind.strip().lower().replace("-", "_")

        def _get(name: str, default=None):
            v = getattr(params, name, None)
            return default if v is None else v

        def _pos(x: float, eps: float = 1e-6) -> float:
            return float(x) if float(x) > eps else eps

        def _build_reward(which: str):
            k = _key(params.reward_kind)
            drift_op = _get("drift_op", "add")
            interval = int(_get("interval", 1))

            # helpful legacy values for kinds that use a base and (optionally) a gap
            base_a = float(_get("reward_base_a", 0.0))
            gap = float(_get("reward_gap", 0.0))
            base_b = base_a + gap

            if which == "a":
                base = base_a
                other_base = base_b
            else:
                base = base_b
                other_base = base_a

            if k == "constant":
                value = float(_get(f"reward_const_{which}", base))
                return make_reward("constant", value=value)

            if k == "linear":
                start = float(_get(f"reward_linear_start_{which}", base))
                step = float(_get(f"reward_linear_step_{which}", 1.0))
                lin_interval = int(_get(f"reward_linear_interval_{which}", interval))
                return make_reward("linear", start=start, step=step, interval=lin_interval, drift_op=drift_op)

            if k in ("normal", "gauss"):
                mean = float(_get("reward_base_a", 0.0)) if which == "a" else float(_get("reward_base_a", 0.0)) + float(
                    _get("reward_gap", 0.0))
                stddev = float(max(0.0, _get("normal_stddev", 1.0)))
                mean_drift = float(_get(f"mean_drift_{which}", 0.0))
                return make_reward(
                    "normal",
                    mean=mean,
                    stddev=stddev,
                    mean_drift=mean_drift,
                    drift_op=drift_op,
                    seed=seed + (10 if which == "a" else 20),
                    interval=interval,
                )

            if k in ("exp", "exponential"):
                mean = float(_get(f"reward_exp_mean_{which}", _pos(base)))
                mean = _pos(mean)
                mean_drift = float(_get(f"reward_exp_mean_drift_{which}", 0.0))
                return make_reward(
                    "exp",
                    mean=mean,
                    mean_drift=mean_drift,
                    drift_op=drift_op,
                    seed=seed + (10 if which == "a" else 20),
                    interval=interval,
                )

            if k == "gamma":
                shape = float(_get(f"reward_gamma_shape_{which}", 1.0))
                scale = float(_get(f"reward_gamma_scale_{which}", _pos(base)))
                shape = _pos(shape)
                scale = _pos(scale)
                scale_drift = float(_get(f"reward_gamma_scale_drift_{which}", 0.0))
                return make_reward(
                    "gamma",
                    shape=shape,
                    scale=scale,
                    scale_drift=scale_drift,
                    drift_op=drift_op,
                    seed=seed + (10 if which == "a" else 20),
                    interval=interval,
                )

            if k == "uniform":
                low = float(_get(f"reward_uniform_low_{which}", 0.0))
                width = float(_get(f"reward_uniform_width_{which}", 1.0))
                high = float(low + max(1e-6, width))
                low_drift = float(_get(f"reward_uniform_low_drift_{which}", 0.0))
                high_drift = float(_get(f"reward_uniform_high_drift_{which}", 0.0))
                return make_reward(
                    "uniform",
                    low=low,
                    high=high,
                    low_drift=low_drift,
                    high_drift=high_drift,
                    drift_op=drift_op,
                    seed=seed + (10 if which == "a" else 20),
                    interval=interval,
                )

            if k in ("mr_drift", "mean_reverting_drift"):
                stationarity = _clip01(float(_get(f"reward_stationarity_{which}", 0.5)))
                reversion_speed = float(_get("reversion_speed", 0.1))
                max_volatility = float(max(0.0, _get("max_volatility", 1.0)))
                ensure_positive = bool(_get("reward_ensure_positive", False))
                return make_reward(
                    "mr_drift",
                    base_value=float(base),
                    stationarity=stationarity,
                    reversion_speed=reversion_speed,
                    ensure_positive=ensure_positive,
                    max_volatility=max_volatility,
                    seed=seed + (10 if which == "a" else 20),
                    interval=interval,
                )

            if k in ("mr_jump", "mean_reverting_jump"):
                stationarity = _clip01(float(_get(f"reward_stationarity_{which}", 0.5)))
                reversion_speed = float(_get("reversion_speed", 0.1))
                ensure_positive = bool(_get("reward_ensure_positive", False))
                jlow = float(_get("jump_low", 2.0))
                jhigh = float(_get("jump_high", 5.0))
                if jhigh < jlow:
                    jhigh = jlow
                return make_reward(
                    "mr_jump",
                    base_value=float(base),
                    stationarity=stationarity,
                    reversion_speed=reversion_speed,
                    ensure_positive=ensure_positive,
                    jump_low=jlow,
                    jump_high=jhigh,
                    seed=seed + (10 if which == "a" else 20),
                    interval=interval,
                )

            raise ValueError(f"Unknown reward_kind={params.reward_kind!r}")

        def _build_duration(which: str):
            k = _key(params.duration_kind)
            drift_op = _get("drift_op", "add")
            interval = int(_get("interval", 1))

            # We need a "base_value" for mean-reverting duration dists:
            # use explicit duration_mean_* first, then duration_const_*, else 1.0
            base = float(_get(f"duration_mean_{which}", _get(f"duration_const_{which}", 1.0)))

            if k == "constant":
                value = float(_get(f"duration_const_{which}", 1.0))
                value = _pos(value)
                return make_duration("constant", value=value)

            if k == "linear":
                start = float(_get(f"duration_linear_start_{which}", base))
                step = float(_get(f"duration_linear_step_{which}", 0.0))
                lin_interval = int(_get(f"duration_linear_interval_{which}", interval))
                return make_duration("linear", start=start, step=step, interval=lin_interval, drift_op=drift_op)

            if k in ("normal", "gauss"):
                mean = float(_get(f"duration_normal_mean_{which}", base))
                stddev = float(max(0.0, _get("duration_normal_stddev", 1.0)))
                mean_drift = float(_get(f"duration_normal_mean_drift_{which}", 0.0))
                return make_duration(
                    "normal",
                    mean=mean,
                    stddev=stddev,
                    mean_drift=mean_drift,
                    drift_op=drift_op,
                    seed=seed + (30 if which == "a" else 40),
                    interval=interval,
                )

            if k in ("exp", "exponential"):
                mean = float(_get(f"duration_mean_{which}", base))
                mean = _pos(mean)
                mean_drift = float(_get(f"duration_exp_mean_drift_{which}", 0.0))
                return make_duration(
                    "exp",
                    mean=mean,
                    mean_drift=mean_drift,
                    drift_op=drift_op,
                    seed=seed + (30 if which == "a" else 40),
                    interval=interval,
                )

            if k == "gamma":
                shape = float(_get(f"duration_gamma_shape_{which}", 1.0))
                scale = float(_get(f"duration_gamma_scale_{which}", base))
                shape = _pos(shape)
                scale = _pos(scale)
                scale_drift = float(_get(f"duration_gamma_scale_drift_{which}", 0.0))
                return make_duration(
                    "gamma",
                    shape=shape,
                    scale=scale,
                    scale_drift=scale_drift,
                    drift_op=drift_op,
                    seed=seed + (30 if which == "a" else 40),
                    interval=interval,
                )

            if k == "uniform":
                # Prefer per-action params; fallback to legacy duration_low/duration_width (shared).
                low = _get(f"duration_uniform_low_{which}", _get("duration_low", 0.001))
                width = _get(f"duration_uniform_width_{which}", _get("duration_width", 1.0))
                low = float(_pos(low))
                width = float(max(1e-6, width))
                high = float(low + width)
                low_drift = float(_get(f"duration_uniform_low_drift_{which}", 0.0))
                high_drift = float(_get(f"duration_uniform_high_drift_{which}", 0.0))
                return make_duration(
                    "uniform",
                    low=low,
                    high=high,
                    low_drift=low_drift,
                    high_drift=high_drift,
                    drift_op=drift_op,
                    seed=seed + (30 if which == "a" else 40),
                    interval=interval,
                )

            if k in ("mr_drift", "mean_reverting_drift"):
                stationarity = _clip01(float(_get(f"duration_stationarity_{which}", 0.5)))
                reversion_speed = float(_get("duration_reversion_speed", _get("reversion_speed", 0.1)))
                max_volatility = float(max(0.0, _get("duration_max_volatility", 1.0)))
                ensure_positive = bool(_get("duration_ensure_positive", True))
                return make_duration(
                    "mr_drift",
                    base_value=_pos(base),
                    stationarity=stationarity,
                    reversion_speed=reversion_speed,
                    ensure_positive=ensure_positive,
                    max_volatility=max_volatility,
                    seed=seed + (30 if which == "a" else 40),
                    interval=interval,
                )

            if k in ("mr_jump", "mean_reverting_jump"):
                stationarity = _clip01(float(_get(f"duration_stationarity_{which}", 0.5)))
                reversion_speed = float(_get("duration_reversion_speed", _get("reversion_speed", 0.1)))
                ensure_positive = bool(_get("duration_ensure_positive", True))
                jlow = float(_get("duration_jump_low", 2.0))
                jhigh = float(_get("duration_jump_high", 5.0))
                if jhigh < jlow:
                    jhigh = jlow
                return make_duration(
                    "mr_jump",
                    base_value=_pos(base),
                    stationarity=stationarity,
                    reversion_speed=reversion_speed,
                    ensure_positive=ensure_positive,
                    jump_low=jlow,
                    jump_high=jhigh,
                    seed=seed + (30 if which == "a" else 40),
                    interval=interval,
                )

            raise ValueError(f"Unknown duration_kind={params.duration_kind!r}")

        # Build rewards/durations for action A and B
        rA = _build_reward("a")
        rB = _build_reward("b")
        dA = _build_duration("a")
        dB = _build_duration("b")

        # Global coupling via hooks (your hook design expects passing the wrapper object;
        # the Distribution calls hook.dist(False)) :contentReference[oaicite:2]{index=2}
        if params.coupled:
            rA.register_hook(rB)
            rB.register_hook(rA)
            dA.register_hook(dB)
            dB.register_hook(dA)

        return non_stationary_simple_unichain2(rA, dA, rB, dB)

    except Exception as e:
        raise RuntimeError(f"Failed to build environment config: {e}") from e


def make_agents(env, er: float, lr: float, beta: float, no_update_on_explore: bool):
    """
    Matches your main.py agent choices & hyperparams (you can tune via CLI).
    """
    from agents.smart_r import SMART
    from agents.relaxed_smart import RelaxedSMART
    from agents.harmonic_r import Harmonic, WeightedHarmonic

    action_space = env.action_space
    return [
        Harmonic(
            name="Harmonic",
            action_space=action_space,
            env=env,
            learning_rate=lr,
            exploration_rate=er,
            rho_learning_rate=beta,
            with_rho_trick=no_update_on_explore,
        ),
        WeightedHarmonic(
            name="Weighted Harmonic",
            action_space=action_space,
            env=env,
            learning_rate=lr,
            exploration_rate=er,
            rho_learning_rate=beta,
            with_rho_trick=no_update_on_explore,
        ),
        RelaxedSMART(
            name="Relaxed SMART",
            action_space=action_space,
            env=env,
            learning_rate=lr,
            exploration_rate=er,
            rho_learning_rate=beta,
            with_rho_trick=no_update_on_explore,
        ),
        SMART(
            name="SMART",
            action_space=action_space,
            env=env,
            learning_rate=lr,
            exploration_rate=er,
            rho_learning_rate=beta,
            with_rho_trick=no_update_on_explore,
        ),
    ]


@dataclass(frozen=True)
class HyperParams:
    lr: float
    er: float
    beta: float
    no_update_on_explore: bool = True


def build_hyper_candidates(seed0: int, n: int) -> list[HyperParams]:
    """
    Hyper candidates are shared across ALL seeds for the same env/trial.
    Keep this modest (e.g., 8-24) unless you have a lot of compute.
    """
    rng = np.random.default_rng(seed0)
    candidates: list[HyperParams] = []

    # A small “robust” log-space-ish random design
    for _ in range(n):
        lr = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))  # was up to 0.6
        er = float(rng.uniform(0.1, 0.1)) # was 0.05 up to 0.6
        # beta = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-2))))
        beta = float(rng.uniform(0.001,0.2))
        candidates.append(HyperParams(lr=lr, er=er, beta=beta, no_update_on_explore=True))

    # optional: include your current baseline explicitly
    candidates.append(HyperParams(lr=0.2, er=0.3, beta=0.01, no_update_on_explore=True))

    # de-dup (optional)
    uniq = []
    seen = set()
    for c in candidates:
        key = (round(c.lr, 6), round(c.er, 6), round(c.beta, 6), c.no_update_on_explore)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def run_one_seed(
        trial_id: int,
        params: EnvParams,
        seed: int,
        *,
        hp: HyperParams,
        hp_id: int,
        num_episodes: int,
        max_steps_per_episode: int,
        verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Runs all agents on the SAME env parameters + SAME seed, but rebuilds the env fresh
    per agent (important when drift is present).

    Returns empty list if trial crashes - this allows the sweep to continue.
    """
    try:
        # Make randomness reproducible (agent exploration often depends on these)
        random.seed(seed)
        np.random.seed(seed)

        from smdp_env import SMDPEnvironment
        from run_smdp_experiment import train_agent

        rows: List[Dict[str, Any]] = []

        # Build & train per agent for fairness under non-stationarity
        for agent_idx in range(4):
            try:
                cfg = build_env_config(params, seed)
                env = SMDPEnvironment(cfg)

                agents = make_agents(env, er=hp.er, lr=hp.lr, beta=hp.beta,
                                              no_update_on_explore=hp.no_update_on_explore)

                agent = agents[agent_idx]

                res = train_agent(env, agent, num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode)

                total_return = safe_float(res.get("total_return")) or 0.0
                total_time = safe_float(res.get("total_time")) or 0.0
                avg_rate = (total_return / total_time) if total_time != 0.0 else 0.0

                episode_returns = res.get("episode_returns") or []
                converged_at = res.get("converged_at")
                if converged_at is None:
                    converged_at = compute_convergence_episode(episode_returns)

                row = {
                    "trial_id": trial_id,
                    "seed": seed,
                    "agent": agent.name,
                    "hp_id": hp_id,
                    "lr": hp.lr,
                    "er": hp.er,
                    "beta": hp.beta,
                    "no_update_on_explore": hp.no_update_on_explore,
                    "total_return": total_return,
                    "total_time": total_time,
                    "avg_rate": avg_rate,
                    "rho": safe_float(res.get("rho")),
                    "converged_at": converged_at,
                    "convergence_stage": convergence_stage(converged_at, num_episodes),
                    "params_json": json.dumps(asdict(params), sort_keys=True),
                }
                rows.append(row)

            except Exception as e:
                # If a specific agent fails, skip it but continue with other agents
                if verbose:
                    print(f"Warning: Agent {agent_idx} failed for trial {trial_id}, seed {seed}: {e}")
                continue

        return rows

    except Exception as e:
        # If the entire trial fails (e.g., env creation), return empty list
        if verbose:
            print(f"Warning: Trial {trial_id}, seed {seed} completely failed: {e}")
        return []


# -------------------------
# Regime Discovery Utilities
# -------------------------

def derive_regime_features(params: dict) -> dict:
    """
    Derive interpretable regime features from raw params_json dict.

    Returns a dict of derived features for regime analysis.
    """
    reward_kind = params.get("reward_kind", "")
    duration_kind = params.get("duration_kind", "")
    coupled = params.get("coupled", False)

    # Variability score mapping (categorical -> ordinal)
    variability_map = {
        "constant": 0,
        "normal": 1,
        "uniform": 2,
        "exp": 3,
        "exponential": 3,
        "mr_drift": 4,
        "mean_reverting_drift": 4,
        "mr_jump": 5,
        "mean_reverting_jump": 5,
    }

    # Heavy-tail families
    heavy_tail_kinds = {"exp", "exponential", "uniform"}
    mr_kinds = {"mr_drift", "mr_jump", "mean_reverting_drift", "mean_reverting_jump"}

    features = {
        # Binary derived features
        "mismatch": reward_kind != duration_kind,
        "reward_heavy": reward_kind in heavy_tail_kinds,
        "duration_heavy": duration_kind in heavy_tail_kinds,
        "duration_memoryless": duration_kind in ("exp", "exponential"),
        "duration_constant": duration_kind == "constant",
        "duration_mr": duration_kind in mr_kinds,
        "reward_mr": reward_kind in mr_kinds,
        "coupled": bool(coupled),

        # Ordinal variability scores
        "duration_variability_score": variability_map.get(duration_kind, -1),
        "reward_variability_score": variability_map.get(reward_kind, -1),

        # Raw categorical features
        "reward_kind": reward_kind,
        "duration_kind": duration_kind,

        # Raw numeric features (with safe defaults)
        "reward_stationarity_a": params.get("reward_stationarity_a"),
        "reward_stationarity_b": params.get("reward_stationarity_b"),
        "duration_stationarity_a": params.get("duration_stationarity_a"),
        "duration_stationarity_b": params.get("duration_stationarity_b"),
        "max_volatility": params.get("max_volatility"),
        "reversion_speed": params.get("reversion_speed"),
        "interval": params.get("interval"),
        "reward_gap": params.get("reward_gap"),
        "duration_mean_a": params.get("duration_mean_a"),
        "duration_width": params.get("duration_width"),
        "normal_stddev": params.get("normal_stddev"),
    }

    return features


def compute_harmonic_vs_smart_metrics(agg: pd.DataFrame, margin_eps: float = 0.0) -> pd.DataFrame:
    """
    Compute per (trial_id, hp_id) the delta_h and binary labels.

    delta_h = max(avg_rate_mean(Harmonic), avg_rate_mean(Weighted Harmonic))
              - max(avg_rate_mean(SMART), avg_rate_mean(Relaxed SMART))
    """
    # Pivot to get agents as columns
    pivot = agg.pivot_table(
        index=["trial_id", "hp_id"],
        columns="agent",
        values="avg_rate_mean",
        aggfunc="first"
    ).reset_index()

    # Compute max of SMART family
    smart_cols = [c for c in ["SMART", "Relaxed SMART"] if c in pivot.columns]
    if smart_cols:
        pivot["smart_family_max"] = pivot[smart_cols].max(axis=1)
    else:
        pivot["smart_family_max"] = 0.0

    # Compute delta_h
    harmonic_cols = [c for c in ["Harmonic", "Weighted Harmonic"] if c in pivot.columns]
    if harmonic_cols:
        pivot["harmonic_family_max"] = pivot[harmonic_cols].max(axis=1)
        pivot["delta_h"] = pivot["harmonic_family_max"] - pivot["smart_family_max"]
    else:
        pivot["delta_h"] = 0.0

    # Binary labels
    pivot["harmonic_beats_smart_family"] = pivot["delta_h"] > 0
    pivot["harmonic_strong_win"] = pivot["delta_h"] > margin_eps

    return pivot


def compute_hp_robustness(agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute hyperparameter robustness per (trial_id, agent).

    Returns:
        trial_robustness: per (trial_id, agent) with hp_mean, hp_std, hp_range
        robustness_winners: per trial, the agent with lowest hp_std
    """
    # Group by (trial_id, agent) and compute stats across hp_id
    robustness = (
        agg.groupby(["trial_id", "agent"], as_index=False)
        .agg(
            hp_mean=("avg_rate_mean", "mean"),
            hp_std=("avg_rate_mean", "std"),
            hp_min=("avg_rate_mean", "min"),
            hp_max=("avg_rate_mean", "max"),
            hp_count=("avg_rate_mean", "count"),
        )
    )
    robustness["hp_range"] = robustness["hp_max"] - robustness["hp_min"]
    robustness["hp_std"] = robustness["hp_std"].fillna(0.0)

    # Find most robust agent per trial (lowest hp_std)
    winners = []
    for tid, g in robustness.groupby("trial_id"):
        g_sorted = g.sort_values("hp_std", ascending=True)
        best = g_sorted.iloc[0]
        winners.append({
            "trial_id": tid,
            "most_robust_agent": best["agent"],
            "best_hp_std": best["hp_std"],
            "best_hp_range": best["hp_range"],
        })

    robustness_winners = pd.DataFrame(winners)

    return robustness, robustness_winners


def build_trial_hp_table(
    agg: pd.DataFrame,
    df_runs: pd.DataFrame,
    margin_eps: float = 0.0
) -> pd.DataFrame:
    """
    Build trial_hp_table.csv: one row per (trial_id, hp_id) with avg_rate_mean
    for all agents + delta_h + labels + params_json + derived features.
    """
    # Get harmonic vs smart metrics
    harmonic_metrics = compute_harmonic_vs_smart_metrics(agg, margin_eps)

    # Pivot agent rates
    pivot_rates = agg.pivot_table(
        index=["trial_id", "hp_id"],
        columns="agent",
        values=["avg_rate_mean", "converged_at_mean"],
        aggfunc="first"
    )
    # Flatten column names
    pivot_rates.columns = [f"{col[0]}_{col[1].replace(' ', '_')}" for col in pivot_rates.columns]
    pivot_rates = pivot_rates.reset_index()

    # Get params_json per trial
    params_by_trial = df_runs.groupby("trial_id", as_index=False)["params_json"].first()

    # Merge
    result = pivot_rates.merge(
        harmonic_metrics[["trial_id", "hp_id", "delta_h", "harmonic_beats_smart_family", "harmonic_strong_win"]],
        on=["trial_id", "hp_id"],
        how="left"
    )
    result = result.merge(params_by_trial, on="trial_id", how="left")

    # Derive regime features
    def extract_features(params_json_str):
        params = json.loads(params_json_str) if pd.notna(params_json_str) else {}
        return derive_regime_features(params)

    feature_dicts = result["params_json"].apply(extract_features)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    feature_df.columns = [f"feat_{c}" for c in feature_df.columns]

    result = pd.concat([result.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    return result


def build_regime_summary(trial_hp_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime_summary.csv: aggregated win-rate and mean delta_h grouped by regime slices.
    """
    groupings = [
        # Main grouping
        ["feat_reward_kind", "feat_duration_kind", "feat_coupled"],
        # Mismatch-based grouping
        ["feat_mismatch", "feat_duration_memoryless", "feat_coupled"],
        # Variability-based grouping
        ["feat_duration_variability_score", "feat_mismatch", "feat_coupled"],
    ]

    summaries = []
    for group_cols in groupings:
        # Filter out missing columns
        valid_cols = [c for c in group_cols if c in trial_hp_table.columns]
        if not valid_cols:
            continue

        summary = (
            trial_hp_table.groupby(valid_cols, as_index=False)
            .agg(
                n_samples=("delta_h", "count"),
                harmonic_win_rate=("harmonic_beats_smart_family", "mean"),
                harmonic_strong_win_rate=("harmonic_strong_win", "mean"),
                mean_delta_h=("delta_h", "mean"),
                std_delta_h=("delta_h", "std"),
                median_delta_h=("delta_h", "median"),
            )
        )
        summary["grouping"] = " × ".join(valid_cols)
        summaries.append(summary)

    if summaries:
        return pd.concat(summaries, ignore_index=True)
    return pd.DataFrame()


def fit_regime_separator(
    trial_hp_table: pd.DataFrame,
    outdir: str,
    max_depth: int = 3,
    min_samples_leaf: int = 50
) -> Optional[Tuple[DecisionTreeClassifier, pd.DataFrame, float]]:
    """
    Fit a shallow DecisionTreeClassifier to predict harmonic_beats_smart_family.

    Returns:
        (tree, feature_importance_df, cv_accuracy)
    """
    # Select feature columns
    feature_cols = [c for c in trial_hp_table.columns if c.startswith("feat_") and
                    trial_hp_table[c].dtype in [np.float64, np.int64, bool, np.bool_]]

    # Also include numeric feature columns that aren't categorical
    for c in trial_hp_table.columns:
        if c.startswith("feat_") and c not in feature_cols:
            try:
                trial_hp_table[c] = pd.to_numeric(trial_hp_table[c], errors='coerce')
                if trial_hp_table[c].notna().sum() > 0:
                    feature_cols.append(c)
            except Exception:
                pass

    if not feature_cols:
        return None

    # Prepare features and target
    X = trial_hp_table[feature_cols].copy()

    # Convert booleans to int
    for c in X.columns:
        if X[c].dtype == bool or X[c].dtype == np.bool_:
            X[c] = X[c].astype(int)

    # Fill NaN with -999 (tree can handle it)
    X = X.fillna(-999)

    y = trial_hp_table["harmonic_beats_smart_family"].astype(int)

    # Filter rows with valid target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    if len(y) < 100 or y.nunique() < 2:
        return None

    # Fit decision tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    tree.fit(X, y)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(tree, X, y, cv=cv, scoring='accuracy')
    cv_accuracy = float(cv_scores.mean())

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": tree.feature_importances_
    }).sort_values("importance", ascending=False)

    # Save tree rules
    tree_rules = export_text(tree, feature_names=feature_cols)
    with open(os.path.join(outdir, "harmonic_vs_smart_tree.txt"), "w") as f:
        f.write(tree_rules)

    # Save feature importance
    feature_importance.to_csv(
        os.path.join(outdir, "harmonic_vs_smart_feature_importance.csv"),
        index=False
    )

    # Optional: confusion matrix plot
    try:
        y_pred = tree.predict(X)
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=['SMART wins', 'Harmonic wins'],
            yticklabels=['SMART wins', 'Harmonic wins'],
            ylabel='True label',
            xlabel='Predicted label',
            title='Harmonic vs SMART-family Confusion Matrix'
        )

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.savefig(os.path.join(outdir, "harmonic_vs_smart_confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass  # Don't fail if plotting fails

    return tree, feature_importance, cv_accuracy


def harmonic_regime_profiling(
    agg: pd.DataFrame,
    df_runs: pd.DataFrame,
    outdir: str,
    margin_eps: float = 0.0,
    num_episodes: int = 100
) -> str:
    """
    Perform Harmonic regime profiling vs SMART-family.

    Returns additional report section as string.
    """
    lines = []
    lines.append("\n\n## Harmonic regime profiling (vs SMART-family)\n")

    # 1. Build trial_hp_table
    trial_hp_table = build_trial_hp_table(agg, df_runs, margin_eps)
    trial_hp_table.to_csv(os.path.join(outdir, "trial_hp_table.csv"), index=False)

    # 2. Compute overall win stats
    total_samples = len(trial_hp_table)
    harmonic_wins = trial_hp_table["harmonic_beats_smart_family"].sum()
    harmonic_strong_wins = trial_hp_table["harmonic_strong_win"].sum()

    lines.append(f"\n### Overall Results\n")
    lines.append(f"| Metric | Value |\n")
    lines.append(f"|--------|-------|\n")
    lines.append(f"| Total (trial, hp) samples | {total_samples} |\n")
    lines.append(f"| Harmonic beats SMART-family | {harmonic_wins} ({100*harmonic_wins/max(1, total_samples):.1f}%) |\n")
    if margin_eps > 0:
        lines.append(f"| Harmonic strong wins (Δ > {margin_eps}) | {harmonic_strong_wins} ({100*harmonic_strong_wins/max(1, total_samples):.1f}%) |\n")
    lines.append(f"| Mean delta_h | {trial_hp_table['delta_h'].mean():.4f} |\n")
    lines.append(f"| Std delta_h | {trial_hp_table['delta_h'].std():.4f} |\n")

    # 3. Win rate by specific conditions
    lines.append(f"\n### Win rate by regime conditions\n")
    lines.append(f"| Condition | N | Harmonic Win Rate | Mean Δh |\n")
    lines.append(f"|-----------|---|-------------------|----------|\n")

    conditions = [
        ("All", trial_hp_table),
        ("mismatch=True", trial_hp_table[trial_hp_table["feat_mismatch"] == True]),
        ("duration_kind=exp", trial_hp_table[trial_hp_table["feat_duration_kind"] == "exp"]),
        ("coupled=True", trial_hp_table[trial_hp_table["feat_coupled"] == True]),
        ("mismatch & coupled", trial_hp_table[(trial_hp_table["feat_mismatch"] == True) &
                                               (trial_hp_table["feat_coupled"] == True)]),
        ("mismatch & duration=exp", trial_hp_table[(trial_hp_table["feat_mismatch"] == True) &
                                                    (trial_hp_table["feat_duration_kind"] == "exp")]),
        ("coupled & duration=exp", trial_hp_table[(trial_hp_table["feat_coupled"] == True) &
                                                   (trial_hp_table["feat_duration_kind"] == "exp")]),
        ("all three", trial_hp_table[(trial_hp_table["feat_mismatch"] == True) &
                                      (trial_hp_table["feat_coupled"] == True) &
                                      (trial_hp_table["feat_duration_kind"] == "exp")]),
    ]

    for cond_name, subset in conditions:
        if len(subset) > 0:
            wr = subset["harmonic_beats_smart_family"].mean()
            mdh = subset["delta_h"].mean()
            lines.append(f"| {cond_name} | {len(subset)} | {100*wr:.1f}% | {mdh:.4f} |\n")

    # 4. Regime summary
    regime_summary = build_regime_summary(trial_hp_table)
    regime_summary.to_csv(os.path.join(outdir, "regime_summary.csv"), index=False)

    # Top 10 regime slices by Harmonic beat-rate
    lines.append(f"\n### Top 10 regime slices by Harmonic beat-rate\n")
    if len(regime_summary) > 0:
        top_beat = regime_summary.nlargest(10, "harmonic_win_rate")
        lines.append(f"| Regime Slice | N | Win Rate | Mean Δh |\n")
        lines.append(f"|--------------|---|----------|----------|\n")
        for _, row in top_beat.iterrows():
            slice_desc = " | ".join([str(row.get(c, "")) for c in row.index
                                     if c.startswith("feat_") and pd.notna(row.get(c))])
            lines.append(f"| {slice_desc[:60]} | {row['n_samples']} | {100*row['harmonic_win_rate']:.1f}% | {row['mean_delta_h']:.4f} |\n")

    # Top 10 by mean delta_h
    lines.append(f"\n### Top 10 regime slices by mean delta_h\n")
    if len(regime_summary) > 0:
        top_delta = regime_summary.nlargest(10, "mean_delta_h")
        lines.append(f"| Regime Slice | N | Win Rate | Mean Δh |\n")
        lines.append(f"|--------------|---|----------|----------|\n")
        for _, row in top_delta.iterrows():
            slice_desc = " | ".join([str(row.get(c, "")) for c in row.index
                                     if c.startswith("feat_") and pd.notna(row.get(c))])
            lines.append(f"| {slice_desc[:60]} | {row['n_samples']} | {100*row['harmonic_win_rate']:.1f}% | {row['mean_delta_h']:.4f} |\n")

    # 5. Hyperparameter robustness
    lines.append(f"\n### Hyperparameter Robustness Analysis\n")

    robustness, robustness_winners = compute_hp_robustness(agg)
    robustness.to_csv(os.path.join(outdir, "trial_robustness.csv"), index=False)

    # Mean hp_std per agent
    agent_robustness = robustness.groupby("agent", as_index=False).agg(
        mean_hp_std=("hp_std", "mean"),
        mean_hp_range=("hp_range", "mean"),
        mean_hp_mean=("hp_mean", "mean"),
    )

    lines.append(f"\n**Mean hyperparameter sensitivity per agent:**\n")
    lines.append(f"| Agent | Mean hp_std | Mean hp_range | Mean hp_mean |\n")
    lines.append(f"|-------|-------------|---------------|---------------|\n")
    for _, row in agent_robustness.sort_values("mean_hp_std").iterrows():
        lines.append(f"| {row['agent']} | {row['mean_hp_std']:.4f} | {row['mean_hp_range']:.4f} | {row['mean_hp_mean']:.4f} |\n")

    # Fraction of trials where each agent is most robust
    if len(robustness_winners) > 0:
        robust_counts = robustness_winners["most_robust_agent"].value_counts()
        total_trials = len(robustness_winners)

        lines.append(f"\n**Fraction of trials where agent is most robust (lowest hp_std):**\n")
        lines.append(f"| Agent | Count | Fraction |\n")
        lines.append(f"|-------|-------|----------|\n")
        for agent, count in robust_counts.items():
            lines.append(f"| {agent} | {count} | {100*count/total_trials:.1f}% |\n")

    # 6. Decision tree separator
    lines.append(f"\n### Decision Tree Regime Separator\n")

    tree_result = fit_regime_separator(trial_hp_table, outdir)
    if tree_result is not None:
        tree, feature_importance, cv_accuracy = tree_result

        lines.append(f"\n**Cross-validation accuracy (5-fold):** {100*cv_accuracy:.1f}%\n")

        lines.append(f"\n**Top 5 most important features:**\n")
        lines.append(f"| Feature | Importance |\n")
        lines.append(f"|---------|------------|\n")
        for _, row in feature_importance.head(5).iterrows():
            lines.append(f"| {row['feature']} | {row['importance']:.4f} |\n")

        # Include tree rules (shortened)
        tree_rules_path = os.path.join(outdir, "harmonic_vs_smart_tree.txt")
        lines.append(f"\n**Decision tree rules** (see `harmonic_vs_smart_tree.txt` for full details):\n")
        try:
            with open(tree_rules_path, "r") as f:
                tree_text = f.read()
            # Show first 20 lines
            rule_lines = tree_text.split("\n")[:20]
            lines.append("```\n")
            lines.append("\n".join(rule_lines))
            if len(tree_text.split("\n")) > 20:
                lines.append("\n... (truncated)\n")
            lines.append("```\n")
        except Exception:
            lines.append("(Could not read tree rules file)\n")
    else:
        lines.append("Could not fit decision tree (insufficient data or variance).\n")

    # 7. Save additional CSVs
    # trial_best_by_combo (already saved in aggregate_and_report as param_sweep_best_combo.csv)
    # trial_best_hp_per_algo (already saved in aggregate_and_report as param_sweep_best_per_algo.csv)

    return "".join(lines)


def aggregate_and_report(df_runs: pd.DataFrame, outdir: str, num_episodes: int, margin_eps: float = 0.0) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Aggregate per (trial, agent)
    agg = (
        df_runs
        .groupby(["trial_id", "agent", "hp_id"], as_index=False)
        .agg(
            avg_rate_mean=("avg_rate", "mean"),
            avg_rate_std=("avg_rate", "std"),
            converged_at_mean=("converged_at", lambda x: float(np.nanmean([v for v in x if v is not None])) if any(
                v is not None for v in x) else np.inf),
            total_return_mean=("total_return", "mean"),
            total_time_mean=("total_time", "mean"),
            lr=("lr", "first"),
            er=("er", "first"),
            beta=("beta", "first"),
            no_update_on_explore=("no_update_on_explore", "first"),
        )
    )

    # Winner per trial: maximize avg_rate_mean, tie-break by convergence (only if >20 steps earlier)
    # If there's a tie (same avg_rate_mean, and convergence within 20 steps), log as tie instead of win
    winners = []
    delta_abs = 0.01  # example absolute margin; tune to your scale
    delta_rel = 0.001  # 0.1% relative margin (optional)
    convergence_threshold = 20

    for tid, g in agg.groupby("trial_id"):
        g2 = g.copy()
        g2["converged_at_mean"] = g2["converged_at_mean"].replace([np.inf], num_episodes * 10.0)

        # sort by rate desc, conv asc (for consistent "best" row selection)
        g2_sorted = g2.sort_values(["avg_rate_mean", "converged_at_mean"], ascending=[False, True])

        best = g2_sorted.iloc[0]
        runner_up = g2_sorted.iloc[1] if len(g2_sorted) > 1 else None

        best_rate = float(best["avg_rate_mean"])
        second_rate = float(runner_up["avg_rate_mean"]) if runner_up is not None else -np.inf

        # meaningful tie threshold
        delta_rate = max(delta_abs, delta_rel * max(1.0, abs(best_rate)))

        # who is "close enough" to best to be considered tied
        close_to_best = g2_sorted[(best_rate - g2_sorted["avg_rate_mean"]) <= delta_rate].copy()
        tied_agents = sorted(close_to_best["agent"].tolist())
        num_tied = len(tied_agents)

        # decide outcome by rate margin
        if (best_rate - second_rate) > delta_rate:
            outcome = "win"
            winner_agent = best["agent"]
            tied_agents_str = None
        else:
            winner_agent = None
            tied_agents_str = ",".join(tied_agents)
            outcome = f"tie_{num_tied}way"

            # OPTIONAL: label a speed-win, but keep winner_agent=None unless you want to force it
            # (this keeps "wins" as rate-separated cases only)
            # fastest_conv = close_to_best["converged_at_mean"].min()
            # second_fastest = close_to_best["converged_at_mean"].nsmallest(2).iloc[1] if num_tied > 1 else fastest_conv
            # if (second_fastest - fastest_conv) > convergence_threshold:
            #     outcome = "tie_rate__speed_separation"

        winners.append({
            "trial_id": int(tid),
            "outcome": outcome,
            "winner": winner_agent,
            "tied_agents": tied_agents_str,
            "best_avg_rate": best_rate,
            "best_converged_at": float(best["converged_at_mean"]),
            "runner_up": (runner_up["agent"] if runner_up is not None else None),
            "runner_up_avg_rate": (second_rate if runner_up is not None else None),
            "delta_rate_used": float(delta_rate),
            "rate_gap": float(best_rate - second_rate) if runner_up is not None else None,
        })

    df_winners = pd.DataFrame(winners)

    # (A) Best hp per algo per trial
    best_per_algo = (
        agg.sort_values(["trial_id", "agent", "avg_rate_mean", "converged_at_mean"],
                        ascending=[True, True, False, True])
        .groupby(["trial_id", "agent"], as_index=False)
        .first()
    )

    # (B) Best combo overall per trial
    best_combo = (
        agg.sort_values(["trial_id", "avg_rate_mean", "converged_at_mean"],
                        ascending=[True, False, True])
        .groupby(["trial_id"], as_index=False)
        .first()
    )

    # Save best per-algo and best-combo aggregates
    best_per_algo.to_csv(os.path.join(outdir, "param_sweep_best_per_algo.csv"), index=False)
    best_combo.to_csv(os.path.join(outdir, "param_sweep_best_combo.csv"), index=False)


    # Join one params_json per trial (they are identical across rows in same trial)
    params_by_trial = df_runs.groupby("trial_id", as_index=False)["params_json"].first()
    df_winners = df_winners.merge(params_by_trial, on="trial_id", how="left")

    # Save artifacts
    df_runs.to_csv(os.path.join(outdir, "param_sweep_runs.csv"), index=False)
    agg.to_csv(os.path.join(outdir, "param_sweep_agg.csv"), index=False)
    df_winners.to_csv(os.path.join(outdir, "param_sweep_winners.csv"), index=False)

    # Human-readable report
    # Separate ties from wins
    df_wins_only = df_winners[df_winners["outcome"] == "win"]

    tie_2way_count = len(df_winners[df_winners["outcome"] == "tie_2way"])
    tie_3way_count = len(df_winners[df_winners["outcome"] == "tie_3way"])
    tie_4way_count = len(df_winners[df_winners["outcome"] == "tie_4way"])
    total_ties = tie_2way_count + tie_3way_count + tie_4way_count
    total_wins = len(df_wins_only)

    win_counts = df_wins_only["winner"].value_counts().to_dict() if len(df_wins_only) > 0 else {}

    lines = []
    lines.append("# Parameter Sweep Report\n")
    lines.append("\n## Results Summary\n")
    lines.append(f"| Outcome | Count |\n")
    lines.append(f"|---------|-------|\n")
    lines.append(f"| Total trials | {len(df_winners)} |\n")
    lines.append(f"| **Wins** | {total_wins} |\n")
    lines.append(f"| **2-way ties** | {tie_2way_count} |\n")
    lines.append(f"| **3-way ties** | {tie_3way_count} |\n")
    lines.append(f"| **4-way ties** | {tie_4way_count} |\n")

    lines.append("\n## Wins by Agent\n")
    if win_counts:
        lines.append(f"| Agent | Wins |\n")
        lines.append(f"|-------|------|\n")
        for k, v in sorted(win_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v} |\n")
    else:
        lines.append("No clear wins.\n")

    # Show tie breakdown if there are ties
    if total_ties > 0:
        lines.append("\n## Ties Breakdown\n")
        if tie_2way_count > 0:
            tie_2way_pairs = df_winners[df_winners["outcome"] == "tie_2way"]["tied_agents"].value_counts().to_dict()
            lines.append(f"| Tied Agents | Count |\n")
            lines.append(f"|-------------|-------|\n")
            for pair, count in sorted(tie_2way_pairs.items(), key=lambda x: -x[1]):
                lines.append(f"| {pair} | {count} |\n")
        if tie_3way_count > 0:
            lines.append(f"\n**3-way ties:** {tie_3way_count}\n")
        if tie_4way_count > 0:
            lines.append(f"\n**4-way ties (all agents equal):** {tie_4way_count}\n")

    lines.append("\n## Typical parameter patterns per winner (wins only)\n")
    # explode params for quick summaries - only for clear wins
    decoded = df_wins_only.copy()
    if len(decoded) > 0:
        decoded["params"] = decoded["params_json"].apply(json.loads)
        for agent_name in sorted(decoded["winner"].dropna().unique()):
            sub = decoded[decoded["winner"] == agent_name]
            if len(sub) == 0:
                continue

            # summarize a few key env knobs
            def col(p):
                return sub["params"].apply(lambda d: d.get(p))

            lines.append(f"\n### {agent_name}\n")
            for key in ["reward_kind", "duration_kind", "coupled"]:
                lines.append(f"- {key}: {col(key).value_counts().to_dict()}\n")
            for key in ["reward_stationarity_a", "reward_stationarity_b", "reversion_speed", "max_volatility",
                        "reward_gap"]:
                vals = pd.to_numeric(col(key), errors="coerce").dropna()
                if len(vals):
                    lines.append(
                        f"- {key}: mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}\n")
    else:
        lines.append("\nNo clear wins to analyze.\n")

    # Harmonic regime profiling section
    regime_section = harmonic_regime_profiling(
        agg=agg,
        df_runs=df_runs,
        outdir=outdir,
        margin_eps=margin_eps,
        num_episodes=num_episodes
    )
    lines.append(regime_section)

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))


def _U(rng, a, b):  # uniform
    return float(rng.uniform(a, b))


def _L(rng, a, b):  # log-uniform
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def _LogU(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))





def build_filtered_trials(n_trials: int, seed0: int) -> list[EnvParams]:
    """
    Build a list of EnvParams for the sweep. This version generates more
    "interesting" environments where algorithms meaningfully separate,
    especially where Weighted Harmonic is robust/risk-dominant.

    New regime axes:
      - duration_pathology: "benign" | "bursty" | "heavy_tail"
          * benign: moderate base_dur (0.3-3.0), low variance
          * bursty: includes very small durations (0.05-2.0), high variance (2-8x)
          * heavy_tail: very small durations (0.01-0.5), extreme variance (4-15x)

      - reward_spiky: True | False
          * True: large reward jumps, high volatility, wide gaps
          * False: moderate/normal reward behavior

      - coupled: True (70%) | False (30%) - now an axis again
    """
    rng = np.random.default_rng(seed0)

    reward_kinds = ["normal", "mr_drift", "mr_jump", "constant", "exp", "uniform"]
    duration_kinds = ["normal", "mr_drift", "mr_jump", "constant", "exp", "uniform"]

    # Build mismatch pairs (reward_kind != duration_kind)
    mismatch_pairs = [(rk, dk) for rk, dk in itertools.product(reward_kinds, duration_kinds) if rk != dk]

    # Heavy-tail friendly duration kinds for biased sampling
    heavy_tail_duration_kinds = ["exp", "mr_jump", "uniform"]
    # Spiky-friendly reward kinds for biased sampling
    spiky_reward_kinds = ["mr_jump", "uniform", "exp"]

    trials: list[EnvParams] = []
    for i in range(n_trials):
        # --- Coupling axis: 70% True, 30% False ---
        coup = bool(rng.choice([True, False], p=[0.7, 0.3]))

        # --- Duration pathology axis ---
        # benign (35%), bursty (35%), heavy_tail (30%)
        duration_pathology = rng.choice(["benign", "bursty", "heavy_tail"], p=[0.35, 0.35, 0.30])

        # --- Reward spiky axis ---
        # spiky (45%), moderate (55%)
        reward_spiky = bool(rng.choice([True, False], p=[0.45, 0.55]))

        # --- Duration pathology determines base_dur and dur_var_mag ---
        if duration_pathology == "benign":
            base_dur = _U(rng, 0.3, 3.0)
            dur_var_mag = _U(rng, 0.1, 0.9)
        elif duration_pathology == "bursty":
            base_dur = _LogU(rng, 0.05, 2.0)  # includes very small
            dur_var_mag = _U(rng, 2.0, 8.0)
        else:  # heavy_tail
            base_dur = _LogU(rng, 0.01, 0.5)  # very small
            dur_var_mag = _U(rng, 4.0, 15.0)

        # --- Biased distribution selection ---
        # 60% of trials in heavy_tail/bursty: prefer heavy-tail-friendly duration kinds
        # 60% of trials in reward_spiky: prefer spiky-friendly reward kinds
        if duration_pathology in ("bursty", "heavy_tail") and rng.random() < 0.6:
            # Prefer heavy-tail-friendly duration kinds
            dk = rng.choice(heavy_tail_duration_kinds)
            # Pick a mismatched reward kind
            valid_rks = [rk for rk in reward_kinds if rk != dk]
            if reward_spiky and rng.random() < 0.6:
                # Further bias toward spiky reward kinds
                spiky_valid = [rk for rk in spiky_reward_kinds if rk != dk]
                rk = rng.choice(spiky_valid) if spiky_valid else rng.choice(valid_rks)
            else:
                rk = rng.choice(valid_rks)
        elif reward_spiky and rng.random() < 0.6:
            # Prefer spiky reward kinds
            rk = rng.choice(spiky_reward_kinds)
            # Pick a mismatched duration kind
            valid_dks = [dk for dk in duration_kinds if dk != rk]
            dk = rng.choice(valid_dks)
        else:
            # Uniform over mismatch pairs (cycle through for coverage)
            rk, dk = mismatch_pairs[i % len(mismatch_pairs)]

        # --- Interval: wide range 1..20 ---
        interval = int(rng.integers(1, 21))

        drift_op = "add"

        # --- Reward parameters based on reward_spiky ---
        reward_base_a = _U(rng, 40.0, 140.0)

        if reward_spiky:
            # Spiky mode: large gaps, high volatility, aggressive jumps
            reward_gap = _U(rng, -80.0, 80.0)
            mv = _LogU(rng, 0.2, 20.0)  # max_volatility

            # mr_jump params for spiky
            jump_low = _U(rng, 0.0, 30.0)
            jump_high = _U(rng, 80.0, 500.0)

            # uniform reward params for spiky
            reward_uniform_width_a = _LogU(rng, 10.0, 300.0)
            reward_uniform_width_b = _LogU(rng, 10.0, 300.0)

            # exp reward drift for spiky
            reward_exp_mean_drift_a = _U(rng, -1.0, 1.0)
            reward_exp_mean_drift_b = _U(rng, -1.0, 1.0)
        else:
            # Moderate mode: keep reward behavior normal
            reward_gap = _U(rng, -25.0, 25.0)
            mv = _LogU(rng, 0.02, 3.0)  # max_volatility

            # mr_jump params moderate
            jump_low = _U(rng, 0.0, 10.0)
            jump_high = _U(rng, 12.0, 40.0)

            # uniform reward params moderate
            reward_uniform_width_a = _LogU(rng, 0.5, 120.0)
            reward_uniform_width_b = _LogU(rng, 0.5, 120.0)

            # exp reward drift moderate
            reward_exp_mean_drift_a = _U(rng, -0.2, 0.2)
            reward_exp_mean_drift_b = _U(rng, -0.2, 0.2)

        # Ensure jump_high >= jump_low + 1e-6
        if jump_high < jump_low + 1e-6:
            jump_high = jump_low + 1e-6

        # --- MR knobs (shared stationarity/reversion) ---
        st = _U(rng, 0.05, 0.95)
        rev = _U(rng, 0.01, 0.6)

        # --- Duration drift scaling based on pathology ---
        # In pathological modes, duration drifts are larger (scaled by dur_var_mag)
        dur_drift_scale = dur_var_mag if duration_pathology != "benign" else 1.0
        duration_exp_mean_drift_a = _U(rng, -0.5, 0.5) * dur_drift_scale
        duration_exp_mean_drift_b = _U(rng, -0.5, 0.5) * dur_drift_scale
        duration_normal_mean_drift_a = _U(rng, -0.5, 0.5) * dur_drift_scale
        duration_normal_mean_drift_b = _U(rng, -0.5, 0.5) * dur_drift_scale

        p = EnvParams(
            reward_kind=rk,
            duration_kind=dk,
            coupled=coup,
            drift_op=drift_op,
            interval=interval,

            # ---- reward params (superset) ----
            reward_base_a=reward_base_a,
            reward_gap=reward_gap,

            # normal reward
            normal_stddev=_U(rng, 3.0, 15.0),
            mean_drift_a=_U(rng, -0.25, 0.25),
            mean_drift_b=_U(rng, -0.25, 0.25),

            # exp reward
            reward_exp_mean_a=_LogU(rng, 0.5, 80.0),
            reward_exp_mean_b=_LogU(rng, 0.5, 80.0),
            reward_exp_mean_drift_a=reward_exp_mean_drift_a,
            reward_exp_mean_drift_b=reward_exp_mean_drift_b,

            # uniform reward
            reward_uniform_low_a=_U(rng, -30.0, 80.0),
            reward_uniform_width_a=reward_uniform_width_a,
            reward_uniform_low_b=_U(rng, -30.0, 80.0),
            reward_uniform_width_b=reward_uniform_width_b,
            reward_uniform_low_drift_a=_U(rng, -0.25, 0.25),
            reward_uniform_low_drift_b=_U(rng, -0.25, 0.25),
            reward_uniform_high_drift_a=_U(rng, -0.25, 0.25),
            reward_uniform_high_drift_b=_U(rng, -0.25, 0.25),

            # MR reward
            reward_stationarity_a=st,
            reward_stationarity_b=st,
            reversion_speed=rev,
            max_volatility=mv,
            reward_ensure_positive=bool(rng.choice([0, 1])),
            jump_low=jump_low,
            jump_high=jump_high,

            # ---- duration params ----
            duration_const_a=max(1e-3, base_dur),
            duration_const_b=max(1e-3, base_dur * _U(rng, 0.9, 1.1)),

            duration_mean_a=max(1e-3, base_dur),
            duration_mean_b=max(1e-3, base_dur * _U(rng, 0.9, 1.1)),
            duration_exp_mean_drift_a=duration_exp_mean_drift_a,
            duration_exp_mean_drift_b=duration_exp_mean_drift_b,

            duration_normal_mean_a=max(1e-3, base_dur),
            duration_normal_mean_b=max(1e-3, base_dur * _U(rng, 0.9, 1.1)),
            duration_normal_stddev=max(1e-3, base_dur * dur_var_mag),
            duration_normal_mean_drift_a=duration_normal_mean_drift_a,
            duration_normal_mean_drift_b=duration_normal_mean_drift_b,

            # uniform duration (low/width)
            duration_low=max(1e-3, base_dur * 0.5),
            duration_width=max(1e-3, base_dur * dur_var_mag * 2.0),

            # MR duration knobs
            duration_stationarity_a=st,
            duration_stationarity_b=st,
            duration_reversion_speed=rev,
            duration_max_volatility=mv,
            duration_ensure_positive=True,
            duration_jump_low=_U(rng, 0.05, 0.30),
            duration_jump_high=_U(rng, 0.40, 1.20),
        )


        trials.append(p)

    rng.shuffle(trials)
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=1000) # GalK was 10000
    ap.add_argument("--n-seeds", type=int, default=5) # GalK was 2
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--verbose", action="store_true", help="Print detailed error messages for failed trials")

    # agent hyperparams (defaults match your main.py values)
    ap.add_argument("--er", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--no-update-on-explore", action="store_true", default=True)

    ap.add_argument("--n-hp", type=int, default=12, help="Hyper candidates per env (trial), shared across seeds.")

    # Regime discovery parameters
    ap.add_argument("--margin-eps", type=float, default=0.0,
                    help="Margin threshold for 'strong win' label (delta_h > margin_eps)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Parameter distributions (randomized DOE)
    # Note: conditional params are sampled too; unused ones are ignored per kind.

    trials = build_filtered_trials(args.n_trials, seed0=args.seed0)
    if len(trials) == 0:
        raise RuntimeError("No valid trials generated.")

    hyper_candidates = build_hyper_candidates(seed0=args.seed0, n=args.n_hp)

    # Prepare tasks: (trial_id, params, seed)
    tasks: List[Tuple[int, EnvParams, int, int]] = []  # (tid, params, seed, hp_id)
    for tid, p in enumerate(trials):
        for hp_id, hp in enumerate(hyper_candidates):
            for i in range(args.n_seeds):
                tasks.append((tid, p, args.seed0 + 1000 * tid + i, hp_id))

    def _do(task):
        tid, p, sd, hp_id = task
        hp = hyper_candidates[hp_id]
        return run_one_seed(
            tid, p, sd,
            hp=hp, hp_id=hp_id,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            verbose=args.verbose,
        )

    with tqdm_joblib(tqdm(total=len(tasks), desc="Running trials x seeds")):
        rows_nested = Parallel(n_jobs=args.n_jobs, batch_size=1)(
            delayed(_do)(t) for t in tasks
        )

    rows: List[Dict[str, Any]] = []
    for block in rows_nested:
        rows.extend(block)

    df_runs = pd.DataFrame(rows)
    aggregate_and_report(df_runs, args.outdir, num_episodes=args.num_episodes, margin_eps=args.margin_eps)
    print(f"Done. Wrote CSVs + report to: {args.outdir}")


if __name__ == "__main__":
    main()
