from .q_learning import ContinuousQLearningAgent
from .average_rates import hma, TIME_RATE, RATIO_RATE_EMA

class ContinuousGLearningAgent(ContinuousQLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate=exploration_rate, **kwargs)
        self.action_count_table = {}
        self.action_time_table = {}
        self.rho_table = {}
        self.tmp = {}

    def reset(self):
        super().reset()
        self.rho_table = {}
        self.tmp = {}
        self.action_count_table = {}
        self.action_time_table = {}

    def initialize_table(self, state):
        super().initialize_table(state)
        if state not in self.action_count_table:
            available_actions = self.get_available_actions(state)
            self.action_count_table[state] = {action: 0 for action in available_actions} 
            self.action_time_table[state] = {action: 0 for action in available_actions} 
            self.rho_table[state] = {action: 0 for action in available_actions} 
            self.tmp[state] ={action: 0 for action in available_actions}  

    def eval(self, state):
        # return super().eval(state)     # Simple version: Choose based on q alone. This will choose based on gain optimality rather than bias optimality
        max_gain = state
        # max_gain = max(self.rho_table[state], key=self.rho_table[state].get)
        max_bias = max(self.q_table[state], key=self.q_table[state].get)

        return max_bias

    def set_target(self, reward, time, state, action, next_state, next_action):
        # total_time= self.action_time_table[state][action]+self.action_time_table[next_state][next_action]+time
        
        # old_avg = self.rho_table[state][action]
        # target = old_avg + (reward - old_avg*time)/total_time # This is probably wrong because it doesn't do harmonic averaging
                
  
        # $$\mu_n = (n-1)*\mu_{n-1}/n + x_n/n$$
        # $$\mu_n = \mu_{n-1} + (x_n - \mu_{n-1})/n$$

        # n = self.action_count_table[state][action]+self.action_count_table[next_state][next_action]+1
        # n = self.action_count_table[state][action]+1
        n = self.action_count_table[next_state][next_action]+1

        if n==1:
            target = reward
            old_avg = self.rho_table[state][action]
        else:
            # q holds the average reward for the state-action pair, also with future q_max
            # old_avg = (self.q_table[state][action]*self.action_count_table[state][action] + self.q_table[next_state][next_action]*self.action_count_table[next_state][next_action]) / n
            # old_avg = self.q_table[state][action]
            old_avg = self.rho_table[next_state][next_action]
            
            target = old_avg + (reward-old_avg)/n
    
        print(f"r: {reward} n: {n} old_avg: {old_avg} target: {target} rho_table: {self.rho_table} q_table: {self.q_table}")

        
        
        # numerator =(reward/time) + self.q_table[next_state][next_action] * self.action_count_table[next_state][next_action]
        # denominator = 1 + self.action_count_table[state][action]+ self.action_count_table[next_state][next_action]
        # target = (numerator/denominator)

        return target

    def update_table(self, state, action, reward, time, td_target, td_error, onpolicy):
        update = self.learning_rate * td_error
        
        self._check_convergence(state, action, update)
        self.q_table[state][action] += update

        self.action_count_table[state][action] += 1
        self.action_time_table[state][action]  += time
        self.rho_table[state][action] = td_target
        self.rho = max(self.q_table[state].values())  # Once we separate rho (gain) and q (bias), this needs to change to rho_table

        # self.tmp[state][action] += self.learning_rate * td_error * self.action_count_table[state][action]
        # if not self.with_rho_trick or (self.with_rho_trick and onpolicy):
            # self.calc_new_rho(reward,time,td_target,td_error)

class GLearningAgent(ContinuousGLearningAgent):

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, 1.0)



