from .q_learning import ContinuousQLearningAgent
from .average_rates import HMA, TIME_RATE, RATIO_RATE_EMA

class ContinuousGLearningAgent(ContinuousQLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate=exploration_rate, **kwargs)
        self.action_count_table = {}
        self.tmp = {}


    def reset(self):
        super().reset()
        self.rho = 0.0
        self.action_count_table = {}

    def initialize_table(self, state):
        super().initialize_table(state)
        if state not in self.action_count_table:
            available_actions = self.get_available_actions(state)
            self.action_count_table[state] = {action: 0 for action in available_actions} 
            self.tmp[state] ={action: 0 for action in available_actions}  

    def set_target(self, reward, time, state, action, next_state, next_action):
        numerator =(reward/time) + self.q_table[next_state][next_action] * self.action_count_table[next_state][next_action]
        denominator = 1 + self.action_count_table[state][action]+ self.action_count_table[next_state][next_action]
        return (numerator/denominator)

    def update_table(self, state, action, reward, time, td_target, td_error, onpolicy):
        super().update_table(state, action, reward, time, td_target, td_error, onpolicy)
        self.action_count_table[state][action] += 1
        self.tmp[state][action] += self.learning_rate * td_error * self.action_count_table[state][action]
        # if not self.with_rho_trick or (self.with_rho_trick and onpolicy):
            # self.calc_new_rho(reward,time,td_target,td_error)

class GLearningAgent(ContinuousGLearningAgent):

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, 1.0)



