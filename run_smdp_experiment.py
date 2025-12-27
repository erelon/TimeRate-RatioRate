from typing import Dict, Any, List

from agents.r_learning import RLAgent
from agents.smart_r import SMARTRLAgent, SMARTEMARLAgent
from agents.harmonic_r import HarmonicRLAgent, HarmonicROLAgent
from smdp_env import * # SMDPEnvironment, * # default_three_state_smdp_config
import numpy as np

def train_agent(env: SMDPEnvironment, agent, num_episodes: int = 100, max_steps_per_episode: int = 1000) -> Dict[ str, Any]:
    episode_returns: List[float] = []
    episode_times: List[float] = []
    rhos = []

    for episode_idx in range(num_episodes):
        state = env.reset(seed=episode_idx)
        total_reward = 0.0
        total_time = 0.0
        steps = 0
        # agent.reset()
        while steps < max_steps_per_episode:
            if state not in agent.q_table:
                available_actions = env.get_available_actions(state)
                agent.q_table[state] = {a: 0 for a in available_actions}
            action = agent.act(state)
            next_state, reward, duration, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, duration)
            # print(f"Episode {episode_idx}, Step {steps}, State {state}, Action {action}, rho {agent.rho:.4f} reward {reward}/{duration} q: {agent.q_table} ")
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

    print(f"Finished training agent {agent.name}: rho = {agent.rho} q:{agent.q_table}")

    return {
        "episode_returns": episode_returns,
        "total_return": sum(episode_returns),
        "total_time": sum(episode_times),
        "converged_at": agent.last_policy_changed_at,
        "rho": agent.rho # np.mean(rhos)
    }


def get_greedy_policy(agent, states, action_space):
    policy = {}
    for s in states:
        if s not in agent.q_table:
            policy[s] = None
            continue
        best_action = max(action_space, key=lambda a: agent.q_table[s].get(a, float("-inf")))
        policy[s] = best_action
    return policy


def main():
    # cfg = default_three_state_smdp_config()
    # cfg = long_three_state_smdp_config(10)
    # cfg = loopy_three_state_smdp_config(10)
    cfg = hub_three_state_smdp_config()
    # cfg = gemini_three_state_smdp_config()

    env = SMDPEnvironment(cfg)

    action_space = env.action_space
    er = 1.0
    lr = 0.1
    beta = 0.2
    agents = [
        # RLAgent(name="R-Learning", action_space=action_space, env=env, learning_rate=lr, exploration_rate=er, rho_learning_rate=beta),
        SMARTRLAgent(name="SMART", action_space=action_space, env=env, learning_rate=lr, exploration_rate=er, rho_learning_rate=beta,with_rho_trick=False),

        SMARTEMARLAgent(name="Relaxed SMART", action_space=action_space, env=env, learning_rate=lr, exploration_rate=er, rho_learning_rate=beta),

        HarmonicRLAgent(name="Weighted Harmonic", action_space=action_space, env=env, learning_rate=0.1, exploration_rate=er, rho_learning_rate=0.1),
        HarmonicROLAgent(name="Harmonic", action_space=action_space, env=env, learning_rate=0.1, exploration_rate=er, rho_learning_rate=0.1),
    ]

    results = {}

    for agent in agents:
        print(f"Training {agent.name}...")
        # agent.reset()
        res = train_agent(env, agent)
        results[agent.name] = {
            **res,
            "policy": get_greedy_policy(agent, env.states, action_space),
        }

    # Print comparison table
    states = env.states[:3]
    header_cols = ["Agent", "TotalReturn", "Total Time", "rho", "ConvergedAt"] + [f"strategy({s})" for s in states]
    rows = []
    for agent_name, res in results.items():
        row = [
            agent_name,
            f"{res['total_return']:.2f}",
            f"{res['total_time']:.2f}",
            f"{res['rho']:.4f}" if res["rho"] is not None else "-",
            str(res['converged_at']),
        ]
        for s in states:
            a = "a" if res["policy"][s] == 0 else "b" if res["policy"][s] == 1 else None
            row.append(str(a) if a is not None else "-")
        rows.append(row)

    # compute column widths
    col_widths = [max(len(str(row[i])) for row in ([header_cols] + rows)) for i in range(len(header_cols))]

    def fmt_row(row):
        return " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))

    print("\n=== SMDP Experiment Results ===")
    print(f"States: {states}")
    print(fmt_row(header_cols))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
