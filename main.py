import os

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from agents.smart_r import SMARTRLAgent, SMARTEMARLAgent
from agents.harmonic_r import HarmonicRLAgent, HarmonicROLAgent
from more_smdp_envs import SMDPConfigFactory
from run_smdp_experiment import  get_greedy_policy
from smdp_env import *


def _slugify(name: str) -> str:
    # Simple filesystem-friendly slug: lowercase, spaces and bad chars to '_'
    import re
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "env"


def sample_distributions_directly(cfg: SMDPConfig, num_samples: int = 300) -> Dict[str, Any]:
    """
    Sample rewards and durations directly from the distributions for each action,
    independent of agent behavior. This shows the true distribution behavior over time.

    Note: num_samples should match max_steps_per_episode since distributions reset each episode.
    """
    dist_rewards: Dict[int, List[float]] = defaultdict(list)
    dist_durations: Dict[int, List[float]] = defaultdict(list)

    # Get transitions from start state for actions A and B
    start_state = cfg.start_state

    for action in cfg.actions:
        key = (start_state, action)
        if key not in cfg.transitions:
            continue

        # Get the transition (assuming single transition per state-action)
        transition = cfg.transitions[key][0]

        # Reset the distributions
        if hasattr(transition._reward, 'reset') and callable(transition._reward.reset):
            transition._reward.reset()
        elif hasattr(transition._reward, 'dist') and hasattr(transition._reward.dist, 'reset'):
            transition._reward.dist.reset()

        if hasattr(transition._duration, 'reset') and callable(transition._duration.reset):
            transition._duration.reset()
        elif hasattr(transition._duration, 'dist') and hasattr(transition._duration.dist, 'reset'):
            transition._duration.dist.reset()

        # Sample the distributions
        for _ in range(num_samples):
            reward = transition.reward()
            duration = transition.duration()
            dist_rewards[action].append(reward)
            dist_durations[action].append(duration)

    return {
        "dist_rewards": dict(dist_rewards),
        "dist_durations": dict(dist_durations),
    }


def train_agent_with_tracking(env: SMDPEnvironment, agent, num_episodes: int = 100, max_steps_per_episode: int = 10000) -> Dict[str, Any]:
    """Train agent while tracking rewards and durations per action."""
    episode_returns: List[float] = []
    episode_times: List[float] = []
    rhos = []

    # Track rewards and durations per action
    action_rewards: Dict[int, List[float]] = defaultdict(list)
    action_durations: Dict[int, List[float]] = defaultdict(list)
    all_rewards: List[float] = []
    all_durations: List[float] = []
    action_sequence: List[int] = []

    # Track distribution samples independently (sample both actions at every step)
    dist_samples_rewards: Dict[int, List[float]] = defaultdict(list)
    dist_samples_durations: Dict[int, List[float]] = defaultdict(list)

    for episode_idx in range(num_episodes):
        state = env.reset(seed=episode_idx)
        total_reward = 0.0
        total_time = 0.0
        steps = 0
        while steps < max_steps_per_episode:
            action = agent.act(state)
            next_state, reward, duration, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, duration)

            # Track per-action data (what actually happened)
            action_rewards[action].append(reward)
            action_durations[action].append(duration)
            all_rewards.append(reward)
            all_durations.append(duration)
            action_sequence.append(action)

            # Track convergence
            if agent.policy_changed:
                agent.last_policy_changed_at = episode_idx

            total_reward += reward
            total_time += duration
            state = next_state
            steps += 1

            if done:
                break
        episode_returns.append(total_reward)
        episode_times.append(total_time)
        rhos.append(agent.rho)

    return {
        "episode_returns": episode_returns,
        "total_return": sum(episode_returns),
        "total_time": sum(episode_times),
        "converged_at": agent.last_policy_changed_at,
        "rho": agent.rho,
        # Distribution tracking data
        "action_rewards": dict(action_rewards),
        "action_durations": dict(action_durations),
        "all_rewards": all_rewards,
        "all_durations": all_durations,
        "action_sequence": action_sequence,
    }


def main():
    smdp_factory = SMDPConfigFactory()

    all_results = defaultdict(dict)
    distribution_data = {}  # Store distribution data per config

    for cfg_name, cfg in smdp_factory.get_all_configs().items():
        env = SMDPEnvironment(cfg)

        action_space = env.action_space
        er = 0.2
        no_update_on_explore = True
        lr = 0.2
        beta = 0.2
        agents = [
            HarmonicRLAgent(name="Weighted Harmonic", action_space=action_space, env=env, learning_rate=lr,
                            exploration_rate=er, rho_learning_rate=beta, with_rho_trick=no_update_on_explore),
            HarmonicROLAgent(name="Harmonic", action_space=action_space, env=env, learning_rate=lr,
                            exploration_rate=er, rho_learning_rate=beta, with_rho_trick=no_update_on_explore),
            SMARTEMARLAgent(name="Relaxed SMART", action_space=action_space, env=env, learning_rate=lr,
                            exploration_rate=er, rho_learning_rate=beta, with_rho_trick=no_update_on_explore),
            SMARTRLAgent(name="SMART", action_space=action_space, env=env, learning_rate=lr, exploration_rate=er,
                         rho_learning_rate=beta, with_rho_trick=no_update_on_explore),
        ]

        results = {}

        num_episodes = 10
        max_steps_per_episode = 1000

        for idx, agent in enumerate(agents):
            res = train_agent_with_tracking(env, agent, num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode)
            avg_rate = res["total_return"] / res["total_time"] if res["total_time"] != 0 else 0.0
            results[agent.name] = {
                **res,
                "policy": get_greedy_policy(agent, env.states, action_space),
                "avg_rate": avg_rate,
            }

        # Sample distributions directly (independent of agent behavior) for visualization
        # Use max_steps_per_episode since distributions reset each episode
        distribution_data[cfg_name] = sample_distributions_directly(cfg, num_samples=max_steps_per_episode * 10)

        for agent_name, res in results.items():
            # Store numeric version for plotting
            all_results[cfg_name][agent_name] = {
                "total_return": float(res["total_return"]),
                "total_time": float(res["total_time"]),
                "avg_rate": float(res["avg_rate"]),
                "rho": float(res["rho"]) if res["rho"] is not None else None,
                "converged_at": int(res["converged_at"]) if res["converged_at"] is not None else None,
                "episode_returns": res["episode_returns"],  # Store learning trajectory
                "policy": res["policy"]
            }

    visualize(all_results, smdp_factory.notes)
    visualize_q_table_histories(agents, cfg_name)
    visualize_distributions(distribution_data, {}, "_long")

    visualize_distributions(distribution_data, smdp_factory.notes, "_short", max_steps_per_episode)


def visualize_distributions(distribution_data: dict, notes: dict, s=None, d=None):
    """Create tmp.py-style plots showing reward/duration distributions per action."""
    os.makedirs(f"plots/distributions{s}", exist_ok=True)

    for env_name, data in distribution_data.items():
        if data is None:
            continue

        env_slug = _slugify(env_name)
        note = notes.get(env_name, "")

        dist_rewards = data["dist_rewards"]
        dist_durations = data["dist_durations"]

        rewards_a = np.array(dist_rewards.get(0, []))[:d]
        rewards_b = np.array(dist_rewards.get(1, []))[:d]
        durations_a = np.array(dist_durations.get(0, []))[:d]
        durations_b = np.array(dist_durations.get(1, []))[:d]

        steps_a = np.arange(1, len(rewards_a) + 1)
        steps_b = np.arange(1, len(rewards_b) + 1)

        def _add_note(fig):
            if note:
                plt.figtext(0.5, 0.01, note, wrap=True, horizontalalignment='center', fontsize=14, fontweight='bold')
                fig.subplots_adjust(bottom=0.15)

        outdir = os.path.join("plots", f"distributions{s}")

        # --- Figure 1: Rewards over time ---
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        if len(rewards_a) > 0:
            ax1.plot(steps_a, rewards_a, label="Action A Rewards", color="blue", alpha=0.8, linewidth=3)
        if len(rewards_b) > 0:
            ax1.plot(steps_b, rewards_b, label="Action B Rewards", color="orange", alpha=0.8, linewidth=3)
        ax1.set_yscale("symlog")
        ax1.set_xlabel("Step (within episode)", fontsize=20, fontweight='bold')
        ax1.set_ylabel("Reward", fontsize=20, fontweight='bold')
        ax1.set_title(f"Rewards Over Time (per episode)", fontsize=22, fontweight='bold')
        ax1.legend(fontsize=16, frameon=True, shadow=True)
        ax1.tick_params(axis='both', labelsize=16, width=2, length=6)
        ax1.grid(True, alpha=0.3)
        _add_note(fig1)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{env_slug}_rewards.png"), dpi=150)
        plt.close()

        # --- Figure 2: Durations over time ---
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        if len(durations_a) > 0:
            ax2.plot(steps_a, durations_a, label="Action A Durations", color="blue", alpha=0.8, linewidth=3)
        if len(durations_b) > 0:
            ax2.plot(steps_b, durations_b, label="Action B Durations", color="orange", alpha=0.8, linewidth=3)
        ax2.set_yscale("symlog")
        ax2.set_xlabel("Step (within episode)", fontsize=20, fontweight='bold')
        ax2.set_ylabel("Duration", fontsize=20, fontweight='bold')
        ax2.set_title(f"Durations Over Time (per episode)", fontsize=22, fontweight='bold')
        ax2.legend(fontsize=16, frameon=True, shadow=True)
        ax2.tick_params(axis='both', labelsize=16, width=2, length=6)
        ax2.grid(True, alpha=0.3)
        _add_note(fig2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{env_slug}_durations.png"), dpi=150)
        plt.close()

        # --- Figure 3: Reward/Duration ratio ---
        ratio_a = None
        ratio_b = None
        if len(rewards_a) > 0 and len(durations_a) > 0:
            ratio_a = np.divide(rewards_a, durations_a, out=np.zeros_like(rewards_a, dtype=float),
                                where=durations_a != 0)
        if len(rewards_b) > 0 and len(durations_b) > 0:
            ratio_b = np.divide(rewards_b, durations_b, out=np.zeros_like(rewards_b, dtype=float),
                                where=durations_b != 0)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        if ratio_a is not None:
            ax3.plot(steps_a, ratio_a, label="reward_A / duration_A", color="green", alpha=0.8, linewidth=3)
        if ratio_b is not None:
            ax3.plot(steps_b, ratio_b, label="reward_B / duration_B", color="red", alpha=0.8, linewidth=3)
        # add a black line on x=1000 without changing the limits of the plot:
        # this line represents the episode step limit, and shows where the distributions reset each episode
        ax3.axvline(x=1000, color="black", linestyle="--", alpha=0.6, linewidth=2,
                    label=f"Episode Step Limit ({1000} steps)")
        # make sure it is not changing the y limits of the plot:

        ax3.set_yscale("log")
        ax3.set_xlabel("Step (within episode)", fontsize=20, fontweight='bold')
        ax3.set_ylabel("Reward / Duration", fontsize=20, fontweight='bold')
        ax3.set_title(f"Reward/Duration Ratio (per episode)", fontsize=22, fontweight='bold')
        ax3.legend(fontsize=16, frameon=True, shadow=True)
        ax3.tick_params(axis='both', labelsize=16, width=2, length=6)
        ax3.grid(True, alpha=0.3)
        _add_note(fig3)
        plt.tight_layout()

        plt.savefig(os.path.join(outdir, f"{env_slug}_ratio.png"), dpi=150)
        plt.close()


def visualize(all_results: defaultdict[Any, dict], notes: dict):
    # Ensure output directories exist
    os.makedirs("plots/comparison", exist_ok=True)
    os.makedirs("plots/learning_curves", exist_ok=True)

    # Consistent colors per agent name
    agent_colors = {
        "Weighted Harmonic": "tab:blue",
        "Relaxed SMART": "tab:orange",
        "SMART": "tab:green",
    }

    # Visualize results per environment with dual-axis plots
    for env_name, agents_metrics in all_results.items():
        if not agents_metrics:
            continue

        env_slug = _slugify(env_name)
        note = notes.get(env_name, "")

        # add the policy chosen to the agent name as agent_name + f"(policy)"
        for agent_name in list(agents_metrics.keys()):
            policy = agents_metrics[agent_name]["policy"]
            policy_str = ",".join(f"{s}:{'a' if a == 0 else 'b' if a == 1 else '-'}" for s, a in policy.items())
            new_agent_name = f"{agent_name} ({policy_str})"
            agents_metrics[new_agent_name] = agents_metrics.pop(agent_name)
        agent_names = list(agents_metrics.keys())

        # Create metrics comparison plot with convergence data
        left_metrics = ["total_return", "total_time"]
        right_metrics = ["avg_rate", "rho"]
        convergence_metrics = ["converged_at"]

        x_left = np.arange(len(left_metrics))
        x_right = np.arange(len(right_metrics))
        x_conv = np.arange(len(convergence_metrics))
        width = 0.8 / max(len(agent_names), 1)

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        ax2 = ax1.twinx()

        # Plot left axis metrics (total_return, total_time)
        for idx, agent_name in enumerate(agent_names):
            vals_left = []
            for m in left_metrics:
                v = agents_metrics[agent_name][m]
                vals_left.append(v)
            offset = (idx - (len(agent_names) - 1) / 2) * width
            ax1.bar(x_left + offset, vals_left, width, label=agent_name,
                    color=agent_colors.get(agent_name, None), alpha=0.7)

        # Plot right axis metrics (avg_rate, rho)
        for idx, agent_name in enumerate(agent_names):
            vals_right = []
            for m in right_metrics:
                v = agents_metrics[agent_name][m]
                if v is None:
                    v = 0.0
                vals_right.append(v)
            offset = (idx - (len(agent_names) - 1) / 2) * width
            ax2.bar(x_right + 2.5 + offset, vals_right, width,
                    color=agent_colors.get(agent_name, None), alpha=0.7)

        # Set up left axis
        ax1.set_xticks(list(x_left) + [x + 2.5 for x in x_right])
        ax1.set_xticklabels(left_metrics + right_metrics, fontsize=18, fontweight='bold')
        ax1.set_ylabel("Total Return / Total Time", color='black', fontsize=20, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=16, width=2, length=6)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # Set up right axis
        ax2.set_ylabel("Avg Rate / Rho", color='black', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=16, width=2, length=6)

        ax1.set_title(f"{env_name} – Performance Metrics", fontsize=22, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=16, frameon=True, shadow=True)

        # add note as a caption
        if note:
            plt.figtext(0.5, 0.05, note, wrap=True, horizontalalignment='center', fontsize=12)
            # Add some margin at the bottom so caption is not cut off
            fig.subplots_adjust(bottom=0.15)

        plt.savefig(os.path.join("plots", "comparison", f"{env_slug}_metrics_comparison.png"), dpi=150)
        plt.close()

        # Create learning curves plot to show convergence trajectories
        fig, ax = plt.subplots(figsize=(10, 6))

        for agent_name in agent_names:
            episode_returns = agents_metrics[agent_name]["episode_returns"]
            episodes = range(1, len(episode_returns) + 1)

            # Plot learning curve
            ax.plot(episodes, episode_returns, label=f"{agent_name} Returns",
                    color=agent_colors.get(agent_name, None), alpha=0.8, linewidth=2)

            # Add convergence marker
            conv_episode = agents_metrics[agent_name]["converged_at"]
            if conv_episode is not None and conv_episode < len(episode_returns):
                ax.axvline(x=conv_episode, color=agent_colors.get(agent_name, None),
                           linestyle='--', alpha=0.6, linewidth=2)
                ax.annotate(f'{agent_name}\nConverged',
                            xy=(conv_episode, episode_returns[conv_episode - 1]),
                            xytext=(conv_episode + 5, episode_returns[conv_episode - 1]),
                            arrowprops=dict(arrowstyle='->', color=agent_colors.get(agent_name, None), alpha=0.6, lw=2),
                            fontsize=14, alpha=0.8, fontweight='bold')

        ax.set_xlabel("Episode", fontsize=20, fontweight='bold')
        ax.set_ylabel("Episode Return", fontsize=20, fontweight='bold')
        ax.set_title(f"{env_name} – Learning Curves and Convergence", fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16, width=2, length=6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16, frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join("plots", "learning_curves", f"{env_slug}_learning_curves.png"), dpi=150)
        plt.close()


def visualize_q_table_histories(agents: list, env_name: str, outdir: str = "plots/q_table_histories"):
    """Plot and save Q-table histories for each agent in a dedicated directory."""
    os.makedirs(outdir, exist_ok=True)
    env_slug = _slugify(env_name)
    for agent in agents:
        # Only plot if agent has q_table_past_a and q_table_past_b attributes
        if not hasattr(agent, 'q_table_past_a') or not hasattr(agent, 'q_table_past_b'):
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(agent.q_table_past_a, label="Q-table Action A", color="blue", alpha=0.8, linewidth=3)
        plt.plot(agent.q_table_past_b, label="Q-table Action B", color="orange", alpha=0.8, linewidth=3)
        plt.xlabel("Update Step", fontsize=20, fontweight='bold')
        plt.ylabel("Q-value", fontsize=20, fontweight='bold')
        plt.title(f"{env_name} – {getattr(agent, 'name', 'agent')} Q-table History", fontsize=22, fontweight='bold')
        plt.tick_params(axis='both', labelsize=16, width=2, length=6)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=16, frameon=True, shadow=True)
        plt.tight_layout()
        agent_slug = _slugify(getattr(agent, 'name', str(id(agent))))
        plt.savefig(os.path.join(outdir, f"{env_slug}_{agent_slug}_q_history.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
