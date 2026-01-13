import numpy as np
import random

class NonStationaryProcess:
    def __init__(self, base_value, rng, stationarity, ensure_positive=False, reversion_speed=0.1):
        """
        :param base_value: The long-term mean (anchor).
        :param reversion_speed: The strength of the pull back to base (0 to 1).
        :param stationarity: float between 0 and 1 (0 = stationary, 1 = high drift/volatility).
        :param ensure_positive: If True, ensures the output is positive.
        :param rng: seeded random generator
        """
        self.base_value = base_value
        self.current_value = base_value
        self.reversion_speed = reversion_speed
        self.stationarity = stationarity
        self.ensure_positive = ensure_positive
        self.rng = rng

    def generate_next(self):
        Throw(Exception("This method should be overridden by subclasses."))

class DriftingProcess(NonStationaryProcess):
    

    def generate_next(self):
        """
        :return: float, the new (drifting) 

        """
        # 1. Mean Reversion Step
        # The further we are from base_value, the stronger the pull.
        reversion_pull = (self.base_value - self.current_value) * self.reversion_speed

        # We use the stationarity_param to scale the standard deviation of the drift.
        # If s=0, sigma=0 (no change). If s=1, sigma is high.
        sigma = self.stationarity * 1.0  # Adjust '1.0' to change the maximum volatility
        # TODO: Tweak multiples of self.stationarity until sufficiently large drift
        
        while True:  # really, do-until
            # 2. Stochastic Drift Step
            # s scales the standard deviation of the random noise.
            # If s=0, drift is 0.
            drift = np.random.normal(loc=0, scale=sigma) # TODO: change to self.rng ?

            # 3. Update the internal state
            observation = self.current_value + (reversion_pull + drift)
           
            # 4. Observation Noise (Optional)
            # Adds a tiny bit of jitter to the return value without affecting the state.
            # observation = self.current_value + np.random.normal(0, 0.05)

            if not self.ensure_positive:
                break
            else if observation > 0:
                break 

        self.current_value = observation 
        return observation

"""
# --- Testing the Logic ---
# Base value of 100, moderate reversion speed
process1 = NonStationaryProcess(base_value=100.0, rng, s=0.5, reversion_speed=0.1)
process2 = NonStationaryProcess(base_value=100.0, rng, s=0, reversion_speed=0.1)

print("Stationary (s=0):")
for _ in range(3):
    print(f"{process2.generate_next():.2f}")

print("\nNon-Stationary Drift (s=0.5):")
for _ in range(5):
    print(f"{process1.generate_next():.2f}")

printf("\nNon-statioary drift, all positive:")
for _ in range(10):
    print(f"{process1.generate_next(True):.2f}")

"""


class JumpProcess(NonStationaryProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls_until_jump = None

    def _reset_jump_counter(self):
        s = self.stationarity
        if s <= 0:
            self.calls_until_jump = float('inf')
            return
        # If s=1, range is [1, 3]. If s=0.1, range is [10, 30].
        low = int(1 / s)
        high = int(3 / s)
        self.calls_until_jump = random.randint(low, max(low, high))

    def generate_next(self):
        # 1. Initialize counter
        if self.calls_until_jump is None:
            self._reset_jump_counter()

        # 2. Apply Mean Reversion (The "Pull" back to base)
        # Formula: change = (target - current) * speed
        reversion_step = (self.base_value - self.current_value) * self.reversion_speed

        while True:  # do-until
            observation = self.current_value + reversion_step


            # 3. Check for Jump (The "Push" away from base)
            if self.stationarity > 0:
                self.calls_until_jump -= 1
                if self.calls_until_jump <= 0:
                    jump_magnitude = random.uniform(2.0, 5.0)
                    observation += jump_magnitude
                    self._reset_jump_counter(s)

            # 4. Generate observation with local noise
            # Note: Local noise doesn't affect the permanent state
            # noise = np.random.normal(loc=0, scale=0.1)
            # observation += noise

            if not ensure_positive:
                break
            else if observation > 0:
                break
        self.current_value = observation
        return observation

"""
# --- Example Usage ---
# We use a high s (1.0) to see frequent jumps and reversion in action
gen = MeanRevertingJumpProcess(base_value=10.0, reversion_speed=0.2)

print("Output (Base=10, s=1.0, Reversion=0.2):")
for i in range(15):
    val = gen.generate_next(s=1.0)
    print(f"Call {i+1:02d}: {val:.2f}")
"""
