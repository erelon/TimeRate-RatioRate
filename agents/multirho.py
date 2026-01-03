from .r_learning import ContinuousRLAgent
from .smart_r import SMARTRLAgent


class StateSMARTRLAgent(SMARTRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.03, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.time = {}
        self.reward = {}

    def reset(self):
        super().reset()
        self.time = {}
        self.reward = {}

    def initialize_table(self, state):
        super().initialize_table(state)
        if state not in self.time:
            available_actions = self.get_available_actions(state)
            self.time[state] = {action: 0 for action in available_actions} 
            self.reward[state] ={action: 0 for action in available_actions} 

    def set_target(self, reward, time, state, action, next_state, next_action):
        rho = (self.reward[state][action] / self.time[state][action]) if self.time[state][action] != 0 else 0 
        return (reward-rho*time) + self.q_table[next_state][next_action]

    def update_table(self, state, action, reward, time, td_target, td_error, onpolicy):
        super().update_table(state, action, reward, time, td_target, td_error, onpolicy)
        self.reward[state][action] += reward
        self.time[state][action] += time

