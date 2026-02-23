from itertools import product, permutations
from typing import Dict, Tuple, List
from random import Random

from dist_factory import make_reward, make_duration
from smdp_env import SMDPConfig, Action, State, Transition


def non_stationary_simple_unichain2(reward1, duration1, reward2, duration2) -> SMDPConfig:
    A, B = 0, 1

    transitions: Dict[Tuple[State, Action], List[Transition]] = {}
    s1, s2 = "s1", "s2"
    states = [s1, s2]

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=1.0, reward=reward1, duration=duration1),
    ]
    transitions[(s1, B)] = [
        Transition(next_state=s2, prob=1.0, reward=reward2, duration=duration2),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s1, prob=1.0, reward=0.0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=states,
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


class SMDPConfigFactory:
    def __init__(self):
        # Factory for creating SMDP configurations
        self.all_configs = {}
        self.notes = {}

        # for linear_drift in [1, 2, 3]:
        #     reward = make_reward("linear", step=linear_drift)
        #     config_name = f"Linear_drift_A_only_{linear_drift}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward, 1, 10, 1)
        #     self.notes[
        #         config_name] = f"\nOnly action A has linear drift reward with drift={linear_drift}\nAction B has constant reward 10\n Durations are 1 for both actions"
        # #
        # for linear_drift in [0.1, 0.2, 0.3]:
        #     reward1 = make_reward("linear", step=1)
        #     reward2 = make_reward("linear", start=10.0, step=linear_drift)
        #     config_name = f"Linear_drift_{linear_drift}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, 1, reward2, 1)
        #     self.notes[
        #         config_name] = f"\nBoth actions have linear drift reward. Action A drift=1, Action B drift={linear_drift}\nThe flip point is at about episode 24\nDurations are 1 for both actions"
        #
        # for linear_drift in [1, 2, 3]:
        #     reward = make_reward("linear", start=0.0, step=linear_drift, interval=5)
        #     config_name = f"Linear_step_A_only_{linear_drift}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward, 1, 10, 1)
        #     self.notes[config_name] = f"\nOnly action A has linear step drift reward with step={linear_drift} every 5 calls\nAction B has constant reward 10\n Durations are 1 for both actions"
        # #
        # for linear_drift in [0.1, 0.2, 0.3]:
        #     reward1 = make_reward("linear", start=0.0, step=1)
        #     reward2 = make_reward("linear", start=10.0, step=linear_drift, interval=5)
        #     config_name = f"Linear_step_{linear_drift}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, 1, reward2, 1)
        #     self.notes[config_name] = f"\nBoth actions have linear drift reward. Action A drift=1, Action B step drift={linear_drift} every 5 calls\nDurations are 1 for both actions"

        # for mean_drift in [0.1, 0.25, 0.5, 0.75]:
        #     reward1 = make_reward("normal", mean=10.0, stddev=3.0, mean_drift=mean_drift)
        #     reward2 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=1)
        #     config_name = f"Normal_action_local_drift_{mean_drift}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, 1, reward2, 1)
        #     self.notes[config_name] = f"\nBoth actions have normal distribution drift reward. Action A mean drift=1, Action B mean drift={mean_drift}\nDurations are 1 for both actions\nThe actions update the drift for themselves"

        # for mean_drift in [0.1, 0.25, 0.5, 0.75]:
        #     reward1 = make_reward("normal", mean=10.0, stddev=1.0, mean_drift=mean_drift)
        #     reward2 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=1)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     config_name = f"Normal_action_global_drift_{mean_drift}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, 1, reward2, 1)
        #     self.notes[config_name] = f"\nBoth actions have normal distribution drift reward. Action A mean drift=1, Action B mean drift={mean_drift}\nDurations are 1 for both actions\nThe actions update the drift for both actions"

        # for mean_drift in [0.1, 0.25, 0.5, 0.75]:
        #     reward1 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=42)
        #     reward2 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=43)
        #
        #     duration1 = make_duration("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=44)
        #     duration2 = make_duration("normal", mean=0.0, stddev=1.0, mean_drift=0.1, seed=45)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     duration1.register_hook(duration2)
        #     duration2.register_hook(duration1)
        #
        #     config_name = f"Normal_action_global_drift_{mean_drift}_duration_drift_slow_{0.1}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
        #     self.notes[config_name] = f"\nThe rewards are drifting in the same way, but duration of action B drifts slower"

        ########################### Important new experiments ###########################
        # for mean_drift in [0.1, 0.15, 0.2]:
        #     reward1 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=42)
        #     reward2 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=43)
        #
        #     duration1 = make_duration("exp", mean=0.0, mean_drift=mean_drift, seed=44)
        #     duration2 = make_duration("exp", mean=0.0, mean_drift=0.1, seed=45)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     duration1.register_hook(duration2)
        #     duration2.register_hook(duration1)
        #
        #     config_name = f"ExpXNormal_action_global_drift_{mean_drift}_duration_drift_slow_{0.1}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
        #     self.notes[config_name] = ("\nRewards are taken from an normal distribution with drifting mean.\n"
        #                                "Durations are taken from an exponential distribution with drifting mean.\n"
        #                                "The rewards are drifting in the same way, but duration of action B drifts slower")

        # for scale_and_mean_drift in [0.1, 0.15, 0.2]:
        #     interval = 1
        #
        #     reward1 = make_reward("gamma", scale_drift=0.1, seed=43, interval=interval)
        #     reward2 = make_reward("gamma", scale_drift=scale_and_mean_drift, seed=42, interval=interval)
        #
        #     duration1 = make_duration("normal", mean_drift=scale_and_mean_drift, seed=44, interval=interval)
        #     duration2 = make_duration("normal", mean_drift=scale_and_mean_drift, seed=45, interval=interval)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     duration1.register_hook(duration2)
        #     duration2.register_hook(duration1)
        #
        #     config_name = f"NormalXGamma_action_global_drift_{scale_and_mean_drift}_duration_drift_slow_{0.1}"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
        #     self.notes[config_name] = ("\nRewards are taken from a gamma distribution with drifting mean.\n"
        #                                "Durations are taken from an normal distribution with drifting mean.\n"
        #                                "The duration are drifting in the same way, but reward of action A drifts slower")
        #########################################################################################################

        # for mean_drift in [0.1, 0.15, 0.2]:
        #     reward1 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=42)
        #     reward2 = make_reward("normal", mean=0.0, stddev=1.0, mean_drift=mean_drift, seed=43)
        #
        #     duration1 = make_duration("uniform", low=1, high=2, high_drift=mean_drift/2, seed=44)
        #     duration2 = make_duration("uniform", low=1, high=2, seed=45)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     duration1.register_hook(duration2)
        #     duration2.register_hook(duration1)
        #
        #     config_name = f"NormalXUniform_action_global_drift_{mean_drift}_duration_drift_slow_{0.1}_seeded"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
        #     self.notes[config_name] = ("\nRewards are taken from a normal distribution with drifting mean.\n"
        #                                "Durations are taken from an uniform distribution with drifting mean.\n"
        #                                "The rewards are drifting in the same way, but duration of action B drifts slower\n"
        #                                "All distributions are seeded for reproducibility")

        # for mean_drift in [0.999, 0.9, 0.8]:
        #     interval = 1
        #     reward1 = make_reward("normal", mean=10.0, stddev=2.0, mean_drift=mean_drift, drift_op="mul", seed=42,
        #                           interval=interval)
        #     reward2 = make_reward("normal", mean=10.0, stddev=2.0, mean_drift=mean_drift, drift_op="mul", seed=43,
        #                           interval=interval)
        #
        #     duration1 = make_duration("normal", mean=10.0, stddev=1.0, mean_drift=mean_drift, drift_op="mul", seed=44,
        #                               interval=interval)
        #     duration2 = make_duration("normal", mean=10.0, stddev=1.0, mean_drift=1, drift_op="mul", seed=45,
        #                               interval=interval)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     duration1.register_hook(duration2)
        #     duration2.register_hook(duration1)
        #
        #     config_name = f"NormalXUniform_action_global_drift_{mean_drift}_duration_drift_slow_{0.1}_seeded"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
        #     self.notes[config_name] = ("\nRewards are taken from a normal distribution with drifting mean.\n"
        #                                "Durations are taken from a normal distribution with drifting mean.\n"
        #                                "The rewards are drifting in the same way, but duration of action B drifts slower\n"
        #                                "All distributions are seeded for reproducibility")
        # sweep = [0., 0.5, 1]
        # for stationarity_r1, stationarity_r2, stationarity_d1, stationarity_d2 in product(sweep, repeat=4):
        #     reward1 = make_reward("mr_drift", base_value=10.0, stationarity=stationarity_r1, reversion_speed=0.1,
        #                           ensure_positive=False, max_volatility=1.0, seed=42)
        #     reward2 = make_reward("mr_drift", base_value=10.0, stationarity=stationarity_r2, reversion_speed=0.1,
        #                           ensure_positive=False, max_volatility=1.0, seed=43)
        #     duration1 = make_duration("mr_jump", base_value=10.0, stationarity=stationarity_d1, reversion_speed=0.1,
        #                               ensure_positive=False,# max_volatility=1.0,
        #                               seed=44)
        #     duration2 = make_duration("mr_jump", base_value=10.0, stationarity=stationarity_d2, reversion_speed=0.1,
        #                               ensure_positive=False,# max_volatility=1.0,
        #                               seed=45)
        #
        #     reward1.register_hook(reward2)
        #     reward2.register_hook(reward1)
        #
        #     duration1.register_hook(duration2)
        #     duration2.register_hook(duration1)
        #
        #     config_name = f"MR_drift_action_global_drift_stationarity_{[stationarity_r1, stationarity_r2, stationarity_d1, stationarity_d2]}_duration_constant"
        #     self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
            # self.notes[config_name] = (
            #     f"{[stationarity_r1, stationarity_r2, stationarity_d1, stationarity_d2 ]}")

        # -------------------------
        # Trigonometric with log scaling (inspired by tmp.py)
        # -------------------------
        # sin * log and cos * log environments
        for log_scale in [0.001,]:# 0.0005, 0.001]:
            for frequency in [1,]:# 0.5, 1.0]:
                # Reward: sin with exponential growth, Duration: cos with exponential growth
                reward1 = make_reward("sin_log", amplitude=1.0, frequency=frequency, offset=15.0,
                                      log_base=10.0, start_exp=0.0, log_scale=log_scale)
                # Reward2: linear slope (like linear_line/8 in tmp.py)
                reward2 = make_reward("linear", start=0.0, step=1/20)

                duration1 = make_duration("cos_log", amplitude=1.0, frequency=frequency, offset=10.0,
                                          log_base=10.0, start_exp=0.0, log_scale=log_scale * 0.5)
                # Duration2: constant 1
                duration2 = make_duration("normal", mean=1.0, stddev=0.1)

                reward1.register_hook(reward2)
                reward2.register_hook(reward1)

                config_name = f"SinCosLog_freq_{frequency}_logscale_{log_scale}"
                self.all_configs[config_name] = non_stationary_simple_unichain2(reward1, duration1, reward2, duration2)
                self.notes[config_name] = (
                    f"\nRewards: sin_log vs linear slope (step=1/8) with frequency={frequency}, log_scale={log_scale}\n"
                    f"Durations: sin_log vs constant 1\n"
                    "Mimics tmp.py pattern: oscillation * exponential growth vs linear slope"
                )

    def get_all_configs(self):
        return self.all_configs
