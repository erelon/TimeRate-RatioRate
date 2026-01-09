import sys

from agents import ContinuousQLearningAgent
from .r_learning import ContinuousRLAgent
from .harmonic_average import HMA

class HarmonicQAgent(ContinuousQLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, learning_rate, discount_factor, exploration_rate, **kwargs)
        self.rq_table = {}

    def reset(self):
        super().reset()
        self.rq_table = {}

    def initialize_table(self, state):
        super().initialize_table(state)
        if state not in self.rq_table:
            available_actions = self.get_available_actions(state)
            self.rq_table[state] = {action: HMA(self.learning_rate) for action in available_actions} 

    def update_table(self, state, action, reward, time, td_target, td_error, onpolicy):
        super().update_table(state, action, reward, time, td_target, td_error, onpolicy)
        self.rq_table[state][action].update_rho(reward,time,reward)  

    def set_target(self, reward, time, state, action, next_state, next_action):
        rho = (self.reward[state][action] / self.time[state][action]) if self.time[state][action] != 0 else 0 
        return (reward-rho*time) + self.q_table[next_state][next_action]


    def learn(self, state, action, reward, next_state, time):
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = time / reward + (self.discount_factor * 1 / (self.rq_table[next_state][best_next_action]))
        td_error = td_target - self.rq_table[state][action]
        self.rq_table[state][action] += self.learning_rate * td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]



class HarmonicRLAgent(ContinuousRLAgent):

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs)
        self.RHO = HMA(rho_learning_rate)

    def reset(self):
        super().reset()
        self.hma.reset()

    def calc_new_rho(self, reward,time,td_target,td_error):
        self.rho = self.hma.update_hma(reward,time,reward)  # Weighted HMA with weight = reward


class HarmonicROLAgent(HarmonicRLAgent):

    def calc_new_rho(self, reward,time,td_target,td_error):
        self.rho = self.hma.update_hma(reward,time,1.0)  # Weighted HMA with weight = 1.0 



class HarmonicRLAgent(ContinuousRLAgent):

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs)
        self.reciprocal_rho = 0.0
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w =0 
        self.pos_w =0 
        self.zero_w =0 

    def reset(self):
        super().reset()
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w =0 
        self.pos_w =0 
        self.zero_w =0 

    def HMA_rho(self, reward, time, weight):
        pos = 1 if reward > 0 else 0
        neg = 1 if reward < 0 else 0
        zero = 1 if reward == 0 else 0

        # Erel's version
        # rho = rho + alpha * (time - reward * rho)
        # equivalent to: rho = (1 - (reward * alpha)) * rho + alpha * time

        # self.pos_reciprocal_rho += self.rho_learning_rate * (time - reward * self.pos_reciprocal_rho) * pos
        # self.pos_w = (1 - self.rho_learning_rate) * self.pos_w + self.rho_learning_rate * pos

        # Erel's version
        # self.neg_reciprocal_rho += self.rho_learning_rate * (time - reward * self.neg_reciprocal_rho) * neg
        # self.neg_w = (1 - self.rho_learning_rate) * self.neg_w + self.rho_learning_rate * neg

        # Gal's version
        reciprocal_rate = 0 if zero == 1 else time / reward
        self.pos_reciprocal_rho = (1 - self.rho_learning_rate)*self.pos_reciprocal_rho + self.rho_learning_rate * reciprocal_rate * pos * weight
        self.pos_w = (1 - self.rho_learning_rate) * self.pos_w + self.rho_learning_rate * pos * weight
        self.neg_reciprocal_rho = (1 - self.rho_learning_rate) * self.neg_reciprocal_rho+self.rho_learning_rate * reciprocal_rate * neg * weight
        self.neg_w = (1 - self.rho_learning_rate) * self.neg_w + self.rho_learning_rate * neg * weight
 
        # All versions

        H_pos = 0 if self.pos_reciprocal_rho == 0 else self.pos_w / self.pos_reciprocal_rho
        H_neg = 0 if self.neg_reciprocal_rho == 0 else self.neg_w / self.neg_reciprocal_rho
        self.zero_w = (1 - self.rho_learning_rate) * self.zero_w + self.rho_learning_rate * zero

        self.rho = (H_pos * self.pos_w + H_neg * self.neg_w) / (self.pos_w + self.neg_w + self.zero_w)

    def calc_new_rho(self, reward,time,td_target,td_error):
        self.HMA_rho(reward,time,reward)  # Weighted HMA with weight = reward


class HarmonicROLAgent(HarmonicRLAgent):

    def calc_new_rho(self, reward,time,td_target,td_error):
        self.HMA_rho(reward,time,1.0)  # Weighted HMA with weight = 1.0 

# Experimental


class HarmonicRLAgent2(ContinuousRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs)
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
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs)
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


