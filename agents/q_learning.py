
from .base import Agent, MAX_REWARDS

class ContinuousQLearningAgent(Agent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, _lambda=0.01, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self._lambda = _lambda
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.policy_changed = False
        self.rho = 0
        self.q_table_past_a= []
        self.q_table_past_b= []

    def reset(self):
        self.q_table = {}
        self.rho = 0


    def initialize_table(self, state):
        if state not in self.q_table:
            available_actions = self.get_available_actions(state)
            self.q_table[state] = {action: MAX_REWARDS for action in available_actions}

    def eval(self, state): # returns best action in state
        return max(self.q_table[state], key=self.q_table[state].get)

    def act(self, state):
        self.initialize_table(state)
        if self.rng.random() < self.exploration_rate:
            available_actions = list(self.q_table[state].keys())
            return self.rng.choice(available_actions)
        return self.eval(state)

    def set_target(self, reward, time, next_q):
        # df = math.exp(-self._lambda * time * self.discount_factor)   # Gemini says this is to be used if discount is given as RATE, rather than factor. 
        # See The Continuous Rate Case: Use $e^{-\gamma \tau}$Use this if: Your $\gamma$ is actually a continuous discount rate (often denoted as $\beta$ or $\rho$ in literature).In continuous-time control or specific SMDP literature (like Bradtke & Duff, 1995), discounting is often defined by a rate parameter $\beta$. 

        df = self.discount_factor**time
        return reward + df * next_q

    def update_table(self, state, action, reward, time,td_target, td_error, onpolicy):
        self.q_table[state][action] += self.learning_rate * td_error
        self.q_table_past_a.append(self.q_table["s1"][0])
        self.q_table_past_b.append(self.q_table["s1"][1])

    def learn(self, state, action, reward, next_state, time):
        self.initialize_table(next_state)
        best_next_action = self.eval(next_state)
        best_old_action = self.eval(state)  # Sometimes, it matters to the table updates -- see r-learning on-policy updates
        td_target = self.set_target(reward,time, self.q_table[next_state][best_next_action])
        td_error = td_target - self.q_table[state][action]

        self._check_convergence(state, action, self.learning_rate * td_error)
        self.update_table(state,action,reward, time,td_target,td_error,(action==best_old_action))

class QLearningAgent(ContinuousQLearningAgent):

    # def set_target(self, reward, next_q):
        # return reward + self.discount_factor * next_q

    def learn(self, state, action, reward, next_state, time):
        super().learn(state, action, reward, next_state, 1.0)

## Have not yet been rewritten

class HarmonicQAgent(QLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1, **kwargs):
        super().__init__(name, action_space, **kwargs)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.rq_table = {}

    def act(self, state):
        if state not in self.q_table:
            available_actions = self.get_available_actions(state)
            self.q_table[state] = {action: MAX_REWARDS for action in available_actions}
            self.rq_table[state] = {action: 1.0 for action in available_actions}
        if self.rng.random() < self.exploration_rate:
            available_actions = list(self.q_table[state].keys())
            return self.rng.choice(available_actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def reset(self):
        super().reset()
        self.rq_table = {}

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            available_actions = self.get_available_actions(next_state)
            self.q_table[next_state] = {action: MAX_REWARDS for action in available_actions}
            self.rq_table[next_state] = {action: 1.0 for action in available_actions}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = time / reward + (self.discount_factor * 1 / (self.rq_table[next_state][best_next_action]))
        td_error = td_target - self.rq_table[state][action]
        self.rq_table[state][action] += self.learning_rate * td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]

