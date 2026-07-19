from .r_learning import ContinuousRLAgent

class SMARTRLAgent(ContinuousRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.total_time = 0
        self.total_reward = 0
        self.step_count = 0
        self.total_totals = 0
        self.beta = rho_learning_rate

    def reset(self):
        super().reset()
        self.total_time = 0
        self.total_reward = 0
        self.step_count = 0
        self.total_totals = 0

    def calc_new_rho(self,reward,time,td_target,td_error):
            self.step_count += 1
            self.total_time += time
            self.total_reward += reward
            # SMART
            self.rho = self.total_reward / self.total_time

            # AVG (sum_r/T): Option 1
            # self.beta=(1.0/(self.step_count+1))
            # self.rho = (1-self.beta)*self.rho + self.beta*(self.total_reward / self.total_time)

            # AVG (sum_r/T): Option 2
            # self.total_totals += (self.total_reward / self.total_time)
            # self.rho = self.total_totals / self.step_count

            # AVG(sum r) / T
            # self.total_totals += self.total_reward
            # self.rho = (self.total_totals / (self.step_count)) / self.total_time
            # print(f"total_totals {self.total_totals}, step_count {self.step_count} avg {(self.total_totals / (self.step_count))} total_reward {self.total_reward} reward {reward} total_time {self.total_time} rho {self.rho}")

class SMARTEMARLAgent(SMARTRLAgent):
    """
    Continuous Reinforcement Learning Agent based on Gostabi 2004: Relaxed SMART.
    This agent is designed for environments with continuous rewards.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3, **kwargs):
        super().__init__(
            name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs
        )
        self.rho_time = 0
        self.rho_reward = 0

    def reset(self):
        super().reset()
        self.rho_time = 0
        self.rho_reward = 0


    def calc_new_rho(self,reward,time,td_target,td_error):
        super().calc_new_rho(reward, time, td_target, td_error)  # Really, only needed to update the step count

        # Now override whatever super() did for self.rho
        b1 = self.rho_learning_rate
        b2 = self.rho_learning_rate

        # Option 1:  Relaxed smart (EMA r)/(EMA time)
        self.rho_time = (1 - b1) * self.rho_time + b1 * time
        self.rho_reward = (1 - b2) * self.rho_reward + b2 * reward

        # Option 2:  AVG(r)/AVG(time)
        # self.rho_time = self.total_time / self.step_count
        # self.rho_reward = self.total_reward / self.step_count

        self.rho = self.rho_reward / self.rho_time

## Experimental

class AdaptiveSMARTRLAgent(SMARTRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3, **kwargs):
        super().__init__(
            name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs
        )
        self.initial_learning_rate = learning_rate

    def reset(self):
        super().reset()
        self.learning_rate  = self.initial_learning_rate

    def learn(self, state, action, reward, next_state, time):
        if (self.rho != 0) or (reward != 0):
            deltarho = reward - self.rho*time
            self.learning_rate = 1 - (self.rho * time) / ((abs(deltarho) + self.rho * time))
        super().learn(state, action, reward, next_state, time)


## Yet to be rewritten using class inhertitence

class StateSMARTRLAgent(SMARTRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.03, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate,
                         **kwargs)
        self.time = {}
        self.reward = {}

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            available_actions = self.get_available_actions(next_state)
            self.q_table[next_state] = {action: reward for action in available_actions}
        if state not in self.q_table:
            available_actions = self.get_available_actions(state)
            self.q_table[state] = {action: reward for action in available_actions}

        if next_state not in self.time:
            self.time[next_state] = 0
            self.reward[next_state] = 0
        if state not in self.time:
            self.time[state] = 0
            self.reward[state] = 0

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)

        rho = (self.reward[state] / self.time[state]) if self.time[state] != 0 else 1
        delta = reward - rho * time
        new_q_state = delta + self.q_table[next_state][best_next_action]
        self._check_convergence(state, action, self.learning_rate * (new_q_state - self.q_table[state][action]))
        self.q_table[state][action] += self.learning_rate * (new_q_state - self.q_table[state][action])
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reward[state] += reward
            self.time[state] += time
            self.total_time += time
            self.total_reward += reward
            self.rho = self.total_reward / self.total_time


