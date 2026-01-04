
class hma:
    def __init__(self, beta):
        self.reciprocal_rho = 0.0
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w =0 
        self.pos_w =0 
        self.zero_w =0 
        self.rho = 0.0
        self.rho_learning_rate = beta

    def reset(self):
        self.reciprocal_rho = 0.0
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w =0 
        self.pos_w =0 
        self.zero_w =0 
        self.rho = 0.0

    def update_hma(self, reward, time, weight) -> float:
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


