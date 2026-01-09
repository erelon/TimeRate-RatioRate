import sys
from .r_learning import ContinuousRLAgent
from .average_rates import hma, TIME_RATE, RATIO_RATE_EMA


class HarmonicRLAgent(ContinuousRLAgent):

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs)
        self.RHO = hma(rho_learning_rate)

    def reset(self):
        super().reset()
        self.RHO.reset()

    def calc_new_rho(self, reward,time,td_target,td_error):
        self.rho = self.RHO.update_rho(reward,time,reward)  # Weighted HMA with weight = reward


class HarmonicROLAgent(HarmonicRLAgent):

    def calc_new_rho(self, reward,time,td_target,td_error):
        self.rho = self.RHO.update_rho(reward,time,1.0)  # Weighted HMA with weight = 1.0 


# Below, to be rewritten or deleted


class HarmonicRLAgent2(ContinuousRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table or next_state not in self.rq_table:
            available_actions = self.get_available_actions(next_state)
            self.q_table[next_state] = {action: 0 for action in available_actions}
            self.rq_table[next_state] = {action: 0 for action in available_actions}
        if state not in self.q_table or state not in self.rq_table:
            available_actions = self.get_available_actions(state)
            self.q_table[state] = {action: 0 for action in available_actions}
            self.rq_table[state] = {action: 0 for action in available_actions}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        recip_td_target = (time / reward) - self.reciprocal_rho + (self.rq_table[next_state][best_next_action])
        recip_td_error = recip_td_target - self.rq_table[state][action]
        self.rq_table[state][action] += self.learning_rate * recip_td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reciprocal_rho = (1 - self.rho_learning_rate) * self.reciprocal_rho + self.rho_learning_rate * (
                recip_td_error)
            self.rho = 1 / self.reciprocal_rho
            self.total_time += time
            self.total_reward += reward


class AdaptiveHarmonicRLAgent2(ContinuousRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.error_scale = 1.0
        self.error_scale_learning_rate = 0.1
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.error_scale = 1.0
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table or next_state not in self.rq_table:
            available_actions = self.get_available_actions(next_state)
            self.q_table[next_state] = {action: 0 for action in available_actions}
            self.rq_table[next_state] = {action: 0 for action in available_actions}
        if state not in self.q_table or state not in self.rq_table:
            available_actions = self.get_available_actions(state)
            self.q_table[state] = {action: 0 for action in available_actions}
            self.rq_table[state] = {action: 0 for action in available_actions}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        recip_td_target = (time / reward) - self.reciprocal_rho + (self.rq_table[next_state][best_next_action])
        recip_td_error = recip_td_target - self.rq_table[state][action]
        rdeltarho = abs(time / reward - self.reciprocal_rho)
        z = 1.0 - ((self.reciprocal_rho) / ((rdeltarho + self.reciprocal_rho)))
        alpha_min, alpha_max = 0.001, 0.8
        alpha = alpha_min + (alpha_max - alpha_min) * (z / (z + 0.1))
        self.rq_table[state][action] += alpha * recip_td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reciprocal_rho = (1 - self.rho_learning_rate) * self.reciprocal_rho + self.rho_learning_rate * (
                recip_td_error)
            self.rho = 1 / self.reciprocal_rho
            self.total_time += time
            self.total_reward += reward


class AdaptiveHarmonicRLAgent(ContinuousRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            available_actions = self.get_available_actions(next_state)
            self.q_table[next_state] = {action: 0 for action in available_actions}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        deltarho = reward - self.rho * time
        delta = deltarho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        alpha = 1.00001 - reward / ((abs(deltarho) + reward))
        self._check_convergence(state, action, alpha * delta)
        self.q_table[state][action] += alpha * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reciprocal_rho = (1 - self.rho_learning_rate) * self.reciprocal_rho + self.rho_learning_rate * (
                    time / (reward))
            self.rho = 1 / self.reciprocal_rho
            self.total_time += time
            self.total_reward += reward
