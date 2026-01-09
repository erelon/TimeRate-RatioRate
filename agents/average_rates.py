class AvgRate:
    def __init__(self):
        self.rho = 0.0

    def reset(self):
        self.rho = 0.0

    def update_rho(self, reward, time, weight) -> float:
        Throw(NotImplementedError)

    def get_rho(self) -> float:
        return self.rho



class HMA(AvgRate):
    def __init__(self, beta):
        super().__init__()
        self.rho_learning_rate = beta

        self.reciprocal_rho = 0.0
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w =0 
        self.pos_w =0 
        self.zero_w =0 

    def reset(self):
        self.reciprocal_rho = 0.0
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w =0 
        self.pos_w =0 
        self.zero_w =0 

    def update_rho(self, reward, time, weight) -> float:
        pos = 1 if reward > 0 else 0
        neg = 1 if reward < 0 else 0
        zero = 1 if reward == 0 else 0

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

        return self.rho

    def get_rho(self) -> float:
        return self.rho


class TIME_RATE(AvgRate):
    def __init__(self):
        super().__init__()
        self.total_time = 0.0
        self.total_reward = 0.0

    def reset(self):
        self.total_time = 0.0
        self.total_reward = 0.0

    def update_rho(self, reward, time, weight) -> float:
        self.total_time += time 
        self.total_reward += reward * weight
        self.rho = self.total_reward / self.total_time

class RATIO_RATE_EMA(AvgRate):
    def __init__(self, beta_1, beta_2 = None):
        super().__init__()
        self.b1 = beta_1
        if beta_2 is None:
            self.b2 = beta_1
        else: self.b2 = beta_2

        self.rho_time = 0
        self.rho_reward = 0

    def reset(self):
        self.rho_time = 0
        self.rho_reward = 0

    def update_rho(self, reward, time, weight) -> float:
        b1 = self.b1
        b2 = self.b2
        # Option 1:  Relaxed smart (EMA r)/(EMA time)
        self.rho_time = (1 - b1) * self.rho_time + b1 * time
        self.rho_reward = (1 - b2) * self.rho_reward + b2 * reward * weight

        # Option 2:  AVG(r)/AVG(time)
        # self.rho_time = self.total_time / self.step_count
        # self.rho_reward = self.total_reward / self.step_count

        self.rho = self.rho_reward / self.rho_time


