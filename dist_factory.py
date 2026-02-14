from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Dict, Literal, Optional, Type

DriftOp = Literal["add", "mul"]


def _apply_drift(value: float, drift: float, op: DriftOp) -> float:
    if op == "add":
        return value + drift
    if op == "mul":
        return value * drift
    raise ValueError(f"Unknown drift op: {op!r}. Expected 'add' or 'mul'.")


class Distribution:
    """
    Base class for anything that produces a number on each call.

    - Supports reset()
    - Optional drift every `interval` calls
    - Optional hook() called on each sample (same behavior as your original code)
    """

    def __init__(self, *, interval: int = 1):
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.interval = interval
        self.step = 0
        self.hook: Optional[Callable[[bool], None]] = None

    def register_hook(self, hook: Callable[[bool], None]) -> None:
        self.hook = hook

    def reset(self) -> None:
        self.step = 0

    def __call__(self, call_hook: bool = True) -> float:
        self.step += 1
        if self.step % self.interval == 0:
            self._drift()

        if self.hook is not None and call_hook:
            # preserve original "hook(False)" behavior
            self.hook.dist(False)

        return self._sample()

    # --- subclass API ---
    def _sample(self) -> float:
        raise NotImplementedError

    def _drift(self) -> None:
        # default: no drift
        return


# "Reward" and "Duration" are just semantic wrappers around a Distribution
class Reward:
    def __init__(self, dist: Distribution):
        self.dist = dist

    def __call__(self) -> float:
        return self.dist()

    def reset(self) -> None:
        self.dist.reset()

    def register_hook(self, hook: Callable[[bool], None]) -> None:
        self.dist.register_hook(hook)


class Duration:
    # rule: duration can't be negative; bump to 0.001
    MIN_DURATION = 0.001

    def __init__(self, dist: Distribution):
        self.dist = dist

    def __call__(self) -> float:
        v = self.dist()
        return v if v >= self.MIN_DURATION else self.MIN_DURATION

    def reset(self) -> None:
        self.dist.reset()

    def register_hook(self, hook: Callable[[bool], None]) -> None:
        self.dist.register_hook(hook)


# -------------------------
# Concrete distributions
# -------------------------

@dataclass
class ConstantDist(Distribution):
    value: float

    def __init__(self, value: float):
        super().__init__(interval=1)
        self.value = value

    def _sample(self) -> float:
        return self.value


class LinearDriftDist(Distribution):
    """
    Deterministic drift of the output itself, but only every `interval` calls.
    """

    def __init__(self, start: float = 0.0, step: float = 1.0, *, interval: int = 1, drift_op: DriftOp = "add"):
        super().__init__(interval=interval)
        self.start = start
        self.step_value = step
        self.drift_op = drift_op
        self.current = start

    def reset(self) -> None:
        super().reset()
        self.current = self.start

    def _drift(self) -> None:
        self.current = _apply_drift(self.current, self.step_value, self.drift_op)

    def _sample(self) -> float:
        return self.current


class NormalDriftDist(Distribution):
    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 1.0,
        *,
        mean_drift: float = 0.0,
        drift_op: DriftOp = "add",
        seed: int = 42,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        if stddev < 0:
            raise ValueError("stddev must be >= 0")
        self.seed = seed
        self.start_mean = mean
        self.stddev = stddev
        self.mean_drift = mean_drift
        self.drift_op = drift_op
        self.rng = Random(seed)
        self.mean = mean

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.mean = self.start_mean

    def _drift(self) -> None:
        self.mean = _apply_drift(self.mean, self.mean_drift, self.drift_op)

    def _sample(self) -> float:
        return self.rng.gauss(self.mean, self.stddev)


class ExpDriftDist(Distribution):
    """
    Exponential distribution parameterized by mean (like your original code).
    Drift updates the mean.
    """

    def __init__(
        self,
        mean: float = 1.0,
        *,
        mean_drift: float = 0.0,
        drift_op: DriftOp = "add",
        seed: int = 42,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.seed = seed
        self.start_mean = mean
        self.mean_drift = mean_drift
        self.drift_op = drift_op
        self.rng = Random(seed)
        self.mean = mean

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.mean = self.start_mean

    def _drift(self) -> None:
        self.mean = _apply_drift(self.mean, self.mean_drift, self.drift_op)
        if self.mean <= 0:
            raise ValueError(f"Exp mean became non-positive after drift: {self.mean}")

    def _sample(self) -> float:
        return self.rng.expovariate(1.0 / self.mean)


class GammaDriftDist(Distribution):
    """
    Gamma distribution with shape k and scale theta.
    Drift updates scale by default (matches your original code).
    """

    def __init__(
        self,
        shape: float = 1.0,
        scale: float = 1.0,
        *,
        scale_drift: float = 0.0,
        drift_op: DriftOp = "add",
        seed: int = 42,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        if shape <= 0:
            raise ValueError("shape must be > 0")
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.seed = seed
        self.start_shape = shape
        self.start_scale = scale
        self.scale_drift = scale_drift
        self.drift_op = drift_op
        self.rng = Random(seed)
        self.shape = shape
        self.scale = scale

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.shape = self.start_shape
        self.scale = self.start_scale

    def _drift(self) -> None:
        self.scale = _apply_drift(self.scale, self.scale_drift, self.drift_op)
        if self.scale <= 0:
            raise ValueError(f"Gamma scale became non-positive after drift: {self.scale}")

    def _sample(self) -> float:
        return self.rng.gammavariate(self.shape, self.scale)


class UniformDriftDist(Distribution):
    """
    Uniform(low, high). Drift can be applied independently to low/high.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        *,
        low_drift: float = 0.0,
        high_drift: float = 0.0,
        drift_op: DriftOp = "add",
        seed: int = 42,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        if high <= low:
            raise ValueError("Uniform requires high > low")
        self.seed = seed
        self.start_low = low
        self.start_high = high
        self.low_drift = low_drift
        self.high_drift = high_drift
        self.drift_op = drift_op
        self.rng = Random(seed)
        self.low = low
        self.high = high
        if high <= low:
            raise ValueError("Uniform requires high > low")

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.low = self.start_low
        self.high = self.start_high

    def _drift(self) -> None:
        self.low = _apply_drift(self.low, self.low_drift, self.drift_op)
        self.high = _apply_drift(self.high, self.high_drift, self.drift_op)
        if self.high <= self.low:
            raise ValueError(f"Uniform became invalid after drift: low={self.low}, high={self.high}")

    def _sample(self) -> float:
        return self.rng.uniform(self.low, self.high)


# -------------------------
# Mean-reverting non-stationary processes (translated)
# -------------------------

class MeanRevertingDriftDist(Distribution):
    """
    DriftingProcess translation:
      x_{t+1} = x_t + (base - x_t)*reversion_speed + N(0, stationarity*max_volatility)
    """

    def __init__(
        self,
        base_value: float,
        *,
        stationarity: float,
        reversion_speed: float = 0.1,
        ensure_positive: bool = False,
        max_volatility: float = 1.0,
        seed: int = 42,
        interval: int = 1,
        max_resample_tries: int = 10_000,
        min_positive: float = 0.001,
    ):
        super().__init__(interval=interval)
        if not (0.0 <= stationarity <= 1.0):
            raise ValueError("stationarity must be in [0, 1]")
        if reversion_speed < 0:
            raise ValueError("reversion_speed must be >= 0")
        if max_volatility < 0:
            raise ValueError("max_volatility must be >= 0")

        self.seed = seed
        self.rng = Random(seed)

        self.base_value = base_value
        self.current_value = base_value

        self.stationarity = stationarity
        self.reversion_speed = reversion_speed
        self.ensure_positive = ensure_positive

        self.max_volatility = max_volatility
        self.max_resample_tries = max_resample_tries
        self.min_positive = min_positive

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.current_value = self.base_value

    def _sample(self) -> float:
        reversion_pull = (self.base_value - self.current_value) * self.reversion_speed
        sigma = self.stationarity * self.max_volatility

        # do-until (with safety cap)
        observation = self.current_value
        for _ in range(self.max_resample_tries):
            drift = self.rng.gauss(0.0, sigma)
            observation = self.current_value + reversion_pull + drift
            if not self.ensure_positive or observation > 0.0:
                break
        else:
            # fallback: avoid infinite loops if base is <= 0 etc.
            if self.ensure_positive:
                observation = max(self.min_positive, observation)

        self.current_value = observation
        return observation


class MeanRevertingJumpDist(Distribution):
    """
    JumpProcess translation:
    - Mean reversion each step
    - Occasional jump away from base with frequency controlled by stationarity:
        calls_until_jump ~ randint(int(1/s), int(3/s)) ; s=0 => no jumps
    """

    def __init__(
        self,
        base_value: float,
        *,
        stationarity: float,
        reversion_speed: float = 0.1,
        ensure_positive: bool = False,
        jump_low: float = 2.0,
        jump_high: float = 5.0,
        seed: int = 42,
        interval: int = 1,
        max_resample_tries: int = 10_000,
        min_positive: float = 0.001,
    ):
        super().__init__(interval=interval)
        if not (0.0 <= stationarity <= 1.0):
            raise ValueError("stationarity must be in [0, 1]")
        if reversion_speed < 0:
            raise ValueError("reversion_speed must be >= 0")
        if jump_high < jump_low:
            raise ValueError("jump_high must be >= jump_low")

        self.seed = seed
        self.rng = Random(seed)

        self.base_value = base_value
        self.current_value = base_value

        self.stationarity = stationarity
        self.reversion_speed = reversion_speed
        self.ensure_positive = ensure_positive

        self.jump_low = jump_low
        self.jump_high = jump_high

        self.calls_until_jump: Optional[float] = None

        self.max_resample_tries = max_resample_tries
        self.min_positive = min_positive

    def reset(self) -> None:
        super().reset()
        self.rng = Random(self.seed)
        self.current_value = self.base_value
        self.calls_until_jump = None

    def _reset_jump_counter(self) -> None:
        s = self.stationarity
        if s <= 0.0:
            self.calls_until_jump = float("inf")
            return
        low = max(1, int(1.0 / s))
        high = max(low, int(3.0 / s))
        self.calls_until_jump = self.rng.randint(low, high)

    def _sample(self) -> float:
        if self.calls_until_jump is None:
            self._reset_jump_counter()

        reversion_pull = (self.base_value - self.current_value) * self.reversion_speed

        observation = self.current_value
        for _ in range(self.max_resample_tries):
            observation = self.current_value + reversion_pull

            if self.stationarity > 0.0:
                self.calls_until_jump -= 1  # type: ignore[operator]
                if self.calls_until_jump <= 0:  # type: ignore[operator]
                    observation += self.rng.uniform(self.jump_low, self.jump_high)
                    self._reset_jump_counter()

            if not self.ensure_positive or observation > 0.0:
                break
        else:
            if self.ensure_positive:
                observation = max(self.min_positive, observation)

        self.current_value = observation
        return observation


# -------------------------
# Trigonometric & Log-Scaled Distributions
# -------------------------

import math


class SinDist(Distribution):
    """
    Returns sin(step * frequency + phase) * amplitude + offset.
    Useful for periodic/oscillating rewards/durations.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        *,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def reset(self) -> None:
        super().reset()

    def _sample(self) -> float:
        return self.amplitude * math.sin(self.step * self.frequency + self.phase) + self.offset


class CosDist(Distribution):
    """
    Returns cos(step * frequency + phase) * amplitude + offset.
    Useful for periodic/oscillating rewards/durations.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        *,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def reset(self) -> None:
        super().reset()

    def _sample(self) -> float:
        return self.amplitude * math.cos(self.step * self.frequency + self.phase) + self.offset


class LogMultiplierDist(Distribution):
    """
    Wraps another distribution and multiplies its output by a logarithmic scaling factor.
    output = base_dist() * log_base^(step * scale + start_exp)

    This creates exponentially growing/shrinking values similar to np.logspace behavior.
    """

    def __init__(
        self,
        base_dist: Distribution,
        *,
        log_base: float = 10.0,
        start_exp: float = 0.0,
        scale: float = 0.001,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        if log_base <= 0 or log_base == 1:
            raise ValueError("log_base must be > 0 and != 1")
        self.base_dist = base_dist
        self.log_base = log_base
        self.start_exp = start_exp
        self.scale = scale

    def reset(self) -> None:
        super().reset()
        self.base_dist.reset()

    def _sample(self) -> float:
        base_value = self.base_dist(call_hook=False)
        exponent = self.start_exp + self.step * self.scale
        log_multiplier = self.log_base ** exponent
        return base_value * log_multiplier


class SinLogDist(Distribution):
    """
    sin(step * frequency + phase) * log_base^(step * scale + start_exp)
    Combines sinusoidal oscillation with exponential growth (like tmp.py: sin_line = np.sin(X) * Y).
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        *,
        log_base: float = 10.0,
        start_exp: float = 0.0,
        log_scale: float = 0.001,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        if log_base <= 0 or log_base == 1:
            raise ValueError("log_base must be > 0 and != 1")
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset
        self.log_base = log_base
        self.start_exp = start_exp
        self.log_scale = log_scale

    def reset(self) -> None:
        super().reset()

    def _sample(self) -> float:
        sin_value = self.amplitude * math.sin(self.step * self.frequency + self.phase) + self.offset
        exponent = self.start_exp + self.step * self.log_scale
        log_multiplier = self.log_base ** exponent
        return sin_value * log_multiplier


class CosLogDist(Distribution):
    """
    cos(step * frequency + phase) * log_base^(step * scale + start_exp)
    Combines cosine oscillation with exponential growth (like tmp.py: cos_line = np.cos(X) * Y).
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        *,
        log_base: float = 10.0,
        start_exp: float = 0.0,
        log_scale: float = 0.001,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        if log_base <= 0 or log_base == 1:
            raise ValueError("log_base must be > 0 and != 1")
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset
        self.log_base = log_base
        self.start_exp = start_exp
        self.log_scale = log_scale

    def reset(self) -> None:
        super().reset()

    def _sample(self) -> float:
        cos_value = self.amplitude * math.cos(self.step * self.frequency + self.phase) + self.offset
        exponent = self.start_exp + self.step * self.log_scale
        log_multiplier = self.log_base ** exponent
        return cos_value * log_multiplier


# -------------------------
# Factory
# -------------------------

class DistFactory:
    _REGISTRY: Dict[str, Type[Distribution]] = {
        "constant": ConstantDist,
        "linear": LinearDriftDist,
        "normal": NormalDriftDist,
        "gauss": NormalDriftDist,
        "exp": ExpDriftDist,
        "exponential": ExpDriftDist,
        "gamma": GammaDriftDist,
        "uniform": UniformDriftDist,

        # mean-reverting non-stationary processes
        "mr_drift": MeanRevertingDriftDist,
        "mean_reverting_drift": MeanRevertingDriftDist,
        "mr_jump": MeanRevertingJumpDist,
        "mean_reverting_jump": MeanRevertingJumpDist,

        # trigonometric & log-scaled distributions
        "sin": SinDist,
        "cos": CosDist,
        "sin_log": SinLogDist,
        "cos_log": CosLogDist,
    }

    @classmethod
    def create(cls, kind: str, **kwargs) -> Distribution:
        key = kind.strip().lower().replace("-", "_")
        if key not in cls._REGISTRY:
            supported = ", ".join(sorted(cls._REGISTRY.keys()))
            raise KeyError(f"Unknown dist kind {kind!r}. Supported: {supported}")
        return cls._REGISTRY[key](**kwargs)


def make_reward(kind: str, **kwargs) -> Reward:
    return Reward(DistFactory.create(kind, **kwargs))


def make_duration(kind: str, **kwargs) -> Duration:
    return Duration(DistFactory.create(kind, **kwargs))


# -------------------------
# Examples
# -------------------------
if __name__ == "__main__":
    # Mean-reverting drift: base=100, moderate stationarity drift
    r = make_reward(
        "mr_drift",
        base_value=100.0,
        stationarity=0.5,
        reversion_speed=0.1,
        ensure_positive=False,
        max_volatility=1.0,
        seed=1,
    )

    # Mean-reverting jump duration: base=10, frequent jumps, clamp duration to >= 0.001 automatically
    d = make_duration(
        "mr_jump",
        base_value=10.0,
        stationarity=1.0,
        reversion_speed=0.2,
        ensure_positive=True,  # optional; Duration also clamps anyway
        seed=2,
    )

    for _ in range(10):
        print("reward:", r(), "duration:", d())
