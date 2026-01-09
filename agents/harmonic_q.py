import sys

from agents import ContinuousQLearningAgent
from .r_learning import ContinuousRLAgent
from .harmonic_average import hma

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
            self.rq_table[state] = {action: hma(self.learning_rate) for action in available_actions} 

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


