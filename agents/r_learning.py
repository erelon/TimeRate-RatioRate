from .q_learning import QLearningAgent


class ContinuousRLAgent(QLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.03, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate=exploration_rate, **kwargs)
        self.rho = 0
        self.with_rho_trick = with_rho_trick
        self.rho_learning_rate = rho_learning_rate

    def reset(self):
        super().reset()
        self.rho = 0

    def calc_new_rho(self,delta,reward,time):
        self.rho += self.rho_learning_rate * delta

    def update_rho(self, onpolicy_action, delta, reward, time):
        if not self.with_rho_trick or (self.with_rho_trick and onpolicy_action):
            self.calc_new_rho(delta,reward,time)

    def set_target(self, reward, time, next):
        return (reward-self.rho*time+next)

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            available_actions = self.get_available_actions(next_state)
            self.q_table[next_state] = {action: 0 for action in available_actions}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)

        delta = self.set_target(reward,time,self.q_table[next_state][best_next_action]) - self.q_table[state][action]

        self._check_convergence(state, action, self.learning_rate * delta)
        self.q_table[state][action] += self.learning_rate * delta

        self.update_rho(action==best_current_action,delta,reward,time)


class RLAgent(ContinuousRLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.03, **kwargs):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate, **kwargs)

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, 1.0)



