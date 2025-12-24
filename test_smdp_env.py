from smdp_env import SMDPEnvironment, default_three_state_smdp_config
from agents.smart_r import SMARTRLAgent


def test_basic_env_properties():
    cfg = default_three_state_smdp_config()
    env = SMDPEnvironment(cfg, seed=123)

    assert set(env.states) == {"s1", "s2", "s3"}
    assert set(env.action_space) == {0, 1}

    # probabilities per (state, action) sum to 1
    for (state, action), transitions in cfg.transitions.items():
        p_sum = sum(t.prob for t in transitions)
        assert abs(p_sum - 1.0) < 1e-8


def test_env_agent_interaction_smoke():
    cfg = default_three_state_smdp_config()
    env = SMDPEnvironment(cfg, seed=123)
    agent = SMARTRLAgent(name="SMART-test", action_space=env.action_space)

    state = env.reset()
    for _ in range(10):
        if state not in agent.q_table:
            agent.q_table[state] = {a: 0.0 for a in env.action_space}
        action = agent.act(state)
        next_state, reward, duration, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, duration)
        state = next_state
        if done:
            break

    assert len(agent.q_table) > 0

