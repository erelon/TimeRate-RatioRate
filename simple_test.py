#!/usr/bin/env python3
"""Run the four robustness-table agents on one fixed environment trajectory."""

from __future__ import annotations

import csv
from pathlib import Path

from agents.harmonic_r import Harmonic, WeightedHarmonic
from agents.relaxed_smart import RelaxedSMART
from agents.smart_r import SMART
from more_smdp_envs import SMDPConfigFactory
from smdp_env import SMDPEnvironment


ENVIRONMENT_NAME = "SinCosLog_freq_1_logscale_0.001"
SEED = 43
NUM_STEPS = 50
ACTION = 0    # 0 is best in long run
EXPLORATION_RATE = 0.0
LEARNING_RATE = 0.1
RHO_LEARNING_RATE = 0.05
RESULTS_DIR = Path(__file__).resolve().parent / "results"

AGENT_CLASSES = {
    "Harmonic": Harmonic,
    "Weighted Harmonic": WeightedHarmonic,
    "Relaxed SMART": RelaxedSMART,
    "SMART": SMART,
}

CSV_FIELDS = [
    "step",
    "reward",
    "duration",
    "instantaneous_rate",
    "selected_action",
    "agent_rho",
    "cumulative_average_reward",
    "mean_rate_per_step"
]


def run_agent(agent_name: str, agent_class) -> Path:
    # Build a fresh configuration because its distributions are stateful.
    config = SMDPConfigFactory().get_all_configs()[ENVIRONMENT_NAME]
    env = SMDPEnvironment(config)
    state = env.reset(seed=SEED)

    agent = agent_class(
        name=agent_name,
        action_space=env.action_space,
        env=env,
        seed=SEED,
        exploration_rate=EXPLORATION_RATE,
        learning_rate=LEARNING_RATE,
        rho_learning_rate=RHO_LEARNING_RATE,
        with_rho_trick=True,
    )

    total_reward = 0.0
    total_steps = 0.0
    total_rates = 0.0
    total_duration = 0.0
    rows = []

    for step in range(1, NUM_STEPS + 1):
        available_actions = env.get_available_actions(state)
        if ACTION not in available_actions:
            # raise RuntimeError(
            #     f"Action {ACTION} is unavailable in state {state!r} at step {step}"
            # )
            action = 0
        else:
            action = ACTION

        agent.initialize_table(state)
        next_state, reward, duration, done, _ = env.step(action)
        agent.learn(state,action, reward, next_state, duration)

        total_reward += reward
        total_steps += 1
        total_duration += duration
        total_rates += (reward/duration)
        rows.append(
            {
                "step": step,
                "reward": reward,
                "duration": duration,
                "instantaneous_rate": reward / duration if duration != 0 else float("nan"),
                "selected_action": action,
                "agent_rho": agent.rho,
                "cumulative_average_reward": total_reward / total_duration,
                "mean_rate_per_step": total_rates / total_steps,
            }
        )

        state = env.reset(seed=SEED) if done else next_state

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"simple_test-{agent_name}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    for agent_name, agent_class in AGENT_CLASSES.items():
        output_path = run_agent(agent_name, agent_class)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
