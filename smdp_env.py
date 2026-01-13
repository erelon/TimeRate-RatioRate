from dataclasses import dataclass
from symtable import Function
from typing import Any, Dict, List, Tuple, Hashable, Optional
from non-stationary import *
import random

State = Hashable
Action = int

class Reward:  # TODO: rewrite as a non-stochastic class -- see "non-stationary"
    def __init__(self):
        self.reset()

    def __call__(self):
        self.current *= 1.5
        return self.current

    def reset(self):
        self.current = 1.0

class Duration:
    def __init__(self):
        self.reset()

    def __call__(self):
        self.current *= 2
        return self.current

    def reset(self):
        self.current = 1.0


@dataclass
class Transition:
    def __init__(self, next_state, prob, reward, duration):
        self._next_state = next_state
        self._prob = prob
        self._reward = reward
        self._duration = duration

    def next_state(self) -> State:
        if isinstance(self._next_state, Function):
            return self._next_state()
        return self._next_state

    def prob(self) -> float:
        if isinstance(self._prob, Function):
            return self._prob()
        return self._prob

    def reward(self) -> float:
        if callable(self._reward):
            return self._reward()
        return self._reward

    def duration(self) -> float:
        if callable(self._duration):
            return self._duration()
        return self._duration


@dataclass
class SMDPConfig:
    states: List[State]
    actions: List[Action]
    # mapping: (state, action) -> list[Transition]
    transitions: Dict[Tuple[State, Action], List[Transition]]
    start_state: State
    terminal_states: Optional[List[State]] = None


class SMDPEnvironment:
    """Simple configurable SMDP environment.

    step(action) returns (next_state, reward, duration, done, info).
    """

    def __init__(self, config: SMDPConfig, seed: int = 42):
        self.config = config
        self.state: State = config.start_state
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.time_elapsed: float = 0.0
        self.total_reward: float = 0.0
        self.terminal_states = set(config.terminal_states or [])

        # For convenience when constructing agents
        self.states = list(config.states)
        self.action_space = sorted(set(config.actions))

        self.reset(seed)

    def reset(self, seed: int = 42) -> State:
        self.state = self.config.start_state
        self.time_elapsed = 0.0
        self.total_reward = 0.0
        self.rng = random.Random(seed)

        for tl in self.config.transitions.values():
            for t in tl:
                for ob in t.__dict__:
                    if "reset" in dir(t.__dict__[ob]):
                        t.__dict__[ob].reset()

        return self.state

    def get_available_actions(self, state: State) -> List[Action]:
        """Return the list of actions available from the given state."""
        available = []
        for action in self.action_space:
            if (state, action) in self.config.transitions:
                available.append(action)
        return available

    def step(self, action: Action):
        key = (self.state, action)
        if key not in self.config.transitions:
            raise ValueError(f"No transition defined for state {self.state!r}, action {action!r}")

        transitions = self.config.transitions[key]
        # sample according to probs
        probs = [t.prob() for t in transitions]
        r = self.rng.random()
        cumulative = 0.0
        chosen: Transition = transitions[-1]
        for t, p in zip(transitions, probs):
            cumulative += p
            if r <= cumulative:
                chosen = t
                break

        self.state = chosen.next_state()
        duration = chosen.duration()
        reward = chosen.reward()

        self.time_elapsed += duration
        self.total_reward += reward

        done = self.state in self.terminal_states
        info = {"prob": p}
        return self.state, reward, duration, done, info


def bonus_unichain_smdp_config() -> SMDPConfig:
    """
    from s1 can take several actions.
    action a leads to s2 with p = 1.0, tau = 1, r=0
    action b leads to s2 with p = 1.0, tau = 1, r=100

    from s2: 
      action a leads to s2 w p=1.0, tau=1, r=10


    """
    transitions: Dict[Tuple[State, Action], List[Transition]] = {}
    s1, s2, s3 = "s1", "s2", "s3"
    states = list([s1, s2])

    A, B = 0, 1  # 0: action a, 1: action b
    actions = list([A])

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=1, reward=0.0, duration=1.0),
    ]

    # s1, action b
    transitions[(s1, B)] = [
        Transition(next_state=s2, prob=1, reward=100.0, duration=1.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s2, prob=1.0, reward=10.0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=[s1, s2],
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def hellorheaven_unichain_smdp_config() -> SMDPConfig:
    """
    from s1 can take several actions.
    action a leads to s2 with p = 1.0, tau = 1, r=0
    action b leads to s3 with p = 1.0, tau = 1, r=100

    from s2: 
      action a leads to s2 w p=1.0, tau=1, r=1

    from s3: 
      action a leads to s3 w p=1.0, tau=1, r=-1


    """
    transitions: Dict[Tuple[State, Action], List[Transition]] = {}
    s1, s2, s3 = "s1", "s2", "s3"
    states = list([s1, s2])

    A, B = 0, 1  # 0: action a, 1: action b
    actions = list([A, B])

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=1, reward=0.0, duration=1.0),
    ]

    # s1, action b
    transitions[(s1, B)] = [
        Transition(next_state=s3, prob=1, reward=1000.0, duration=1.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s2, prob=1.0, reward=1.0, duration=1.0),
    ]

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state=s3, prob=1.0, reward=-1.0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=[s1, s2, s3],
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def noisy_hellorheaven_unichain_smdp_config(noise_factor: float) -> SMDPConfig:
    """
    from s1 can take several actions.
    action a leads to s2 with p = 1.0, tau = 1, r=0
    action b leads to s3 with p = 1.0, tau = 1, r=100

    from s2: 
      action a leads to s2 w p=1.0, tau=1, r=1
      action b leads to s2 with tau=0, r = 1 or 2 at 0.5 prob

    from s3: 
      action a leads to s3 w p=1.0, tau=1, r=-1
      action b leads to


    """
    transitions: Dict[Tuple[State, Action], List[Transition]] = {}
    s1, s2, s3 = "s1", "s2", "s3"
    states = list([s1, s2])

    A, B = 0, 1  # 0: action a, 1: action b
    actions = list([A, B])

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=1, reward=0.0, duration=1.0),
    ]

    # s1, action b
    transitions[(s1, B)] = [
        Transition(next_state=s3, prob=1, reward=100.0, duration=1.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s2, prob=1.0, reward=1.0, duration=1.0),
    ]

    transitions[(s2, B)] = [
        Transition(next_state=s2, prob=0.5, reward=1.0, duration=1.0),
        Transition(next_state=s2, prob=0.5, reward=2.0, duration=1.0),
    ]

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state=s3, prob=1.0, reward=-1.0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=[s1, s2, s3],
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def feinberg1_three_state_smdp_config() -> SMDPConfig:
    """Return the SMDPConfig that matches the provided 3-state diagram.

    States: s1, s2, s3
    Actions: 0 -> action a

    - At s1:
        * action a leads to s2 with p=0.5, tau=1, r=0
        * action a leads to s3 with p=0.5, tau=1, r=0
    - At s2:
        * action a leads to s2 with p=1.0, tau=1, r=10
    - At s3:
        * action a leads to s3 with p=1.0, tau=2, r=10

    policy a@s1 should yield 7.5 according to time average
    policy a@s1 yields 6.6666 under ratio of expectations 

    """

    s1, s2, s3 = "s1", "s2", "s3"
    A, B = 0, 1  # 0: action a, 1: action b

    transitions: Dict[Tuple[State, Action], List[Transition]] = {}

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=0.5, reward=0.0, duration=1.0),
        Transition(next_state=s3, prob=0.5, reward=0.0, duration=1.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s2, prob=1.0, reward=10.0, duration=1.0),
    ]

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state=s3, prob=1.0, reward=10.0, duration=2.0),
    ]

    cfg = SMDPConfig(
        states=[s1, s2, s3],
        actions=[A],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def gemini_three_state_smdp_config() -> SMDPConfig:
    """Return the SMDPConfig that matches the provided 3-state diagram.

    States: s1, s2, s3
    Actions: 0 -> action a, 1 -> action b

    From the gemini conversation at: https://gemini.google.com/app/02e42239d8664b7d 
    - At s1:
        * action a leads to s2 with p=0.5, tau=1, r=20
        * action a leads to s3 with p=0.5, tau=19, r=0
        * action b leads to s1 with p=1.0, tau=1, r=4
    - At s2:
        * action a leads to s2 with p=1.0, tau=1, r=20
    - At s3:
        * action a leads to s3 with p=1.0, tau=19, r=0

    policy b@s1 yields 4 both for time rate as well as for ratio rate.

    policy a@s1 should yield 10 according to time average, better than b
    policy a@s1 yields 1 under ratio of expectations, worse than b 

    """

    s1, s2, s3 = "s1", "s2", "s3"
    A, B = 0, 1  # 0: action a, 1: action b

    transitions: Dict[Tuple[State, Action], List[Transition]] = {}

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=0.5, reward=20.0, duration=1.0),
        Transition(next_state=s3, prob=0.5, reward=0.0, duration=19.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s2, prob=1.0, reward=20.0, duration=1.0),
    ]

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state=s3, prob=1.0, reward=0.0, duration=19.0),
    ]

    # s1, action b (self-loop)
    transitions[(s1, B)] = [
        Transition(next_state=s1, prob=1.0, reward=4.0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=[s1, s2, s3],
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def long_three_state_smdp_config(k: int) -> SMDPConfig:
    """Return the SMDPConfig that matches the provided 3-state diagram.

    States: s1, s2_1,... s_2_k, s3_i, ... s_3_k
    Actions: 0 -> action a, 1 -> action b

    From the diagram (as interpreted):
    - At s1:
        * action a leads to s2 with p=0.5, tau=1, r=0
        * action a leads to s3 with p=0.5, tau=1, r=0
        * action b leads to s1 with p=1.0, tau=1, r=2/5
    - At s2_i:
        * action a leads to s2_{i+1} with p=1.0, tau=1, r=1
        * s2_k: action leads to s2_k
    - At s3:
        * action a leads to s3 with p=1.0, tau=2, r=0

    This can be easily modified in code if you want to try other structures.
    """

    s1, s2, s3 = "s1", "s2", "s3"
    A, B = 0, 1  # 0: action a, 1: action b

    states = list([s1, s2, s3])
    actions = list([A])
    transitions: Dict[Tuple[State, Action], List[Transition]] = {}

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=0.5, reward=0.0, duration=0.000001),
        Transition(next_state=s3, prob=0.5, reward=0.0, duration=0.000001),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state="s2_1", prob=1.0, reward=1.0, duration=1.0),
    ]

    for i in range(1, k + 1):
        si = f"s2_{i}"
        transitions[(si, A)] = [
            Transition(next_state=f"s2_{i + 1}", prob=1.0, reward=1.0 if i % 2 else 1.0,
                       duration=1.0 if i % 2 else 1.0),
        ]
        states.append(si)

    transitions[(f"s2_{k + 1}", A)] = [
        Transition(next_state=f"s2_{k + 1}", prob=1.0, reward=1.0, duration=1.0),
    ]
    states.append(f"s2_{k + 1}")

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state="s3_1", prob=1.0, reward=0.0, duration=2.0),
    ]

    for i in range(1, k + 1):
        si = f"s3_{i}"
        transitions[(si, A)] = [
            Transition(next_state=f"s3_{i + 1}", prob=1.0, reward=0 if i % 2 else 0, duration=2.0 if i % 2 else 2.0),
        ]
        states.append(si)

    transitions[(f"s3_{k + 1}", A)] = [
        Transition(next_state=f"s3_{k + 1}", prob=1.0, reward=0, duration=2.0),
    ]
    states.append(f"s3_{k + 1}")

    # Action B
    actions.append(B)

    # s1, action b (self-loop)
    transitions[(s1, B)] = [
        Transition(next_state=s1, prob=1.0, reward=0.4, duration=1.0),  # reward = 2.0/5.0 = 0.4
    ]

    cfg = SMDPConfig(
        states,
        actions,
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def loopy_long_three_state_smdp_config(k: int) -> SMDPConfig:
    """Return the SMDPConfig that matches the provided 3-state diagram.

    States: s1, s2_1,... s_2_k, s3_i, ... s_3_k
    Actions: 0 -> action a, 1 -> action b

    From the diagram (as interpreted):
    - At s1:
        * action a leads to s2 with p=0.5, tau=1, r=0
        * action a leads to s3 with p=0.5, tau=1, r=0
        * action b leads to s1 with p=1.0, tau=1, r=2/5
    - At s2_i:
        * action a leads to s2_{i+1} with p=1.0, tau=1, r=1
        * s2_k: action leads to s1
    - At s3:
        * action a leads to s3 with p=1.0, tau=2, r=0
        * s3_k: action leads to s1

    This can be easily modified in code if you want to try other structures.
    """

    s1, s2, s3 = "s1", "s2", "s3"
    A, B = 0, 1  # 0: action a, 1: action b

    states = list([s1, s2, s3])
    actions = list([A])
    transitions: Dict[Tuple[State, Action], List[Transition]] = {}

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=0.5, reward=0.0, duration=1.0),
        Transition(next_state=s3, prob=0.5, reward=0.0, duration=1.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state="s2_1", prob=1.0, reward=1.0, duration=8.0),
    ]

    for i in range(1, k + 1):
        si = f"s2_{i}"
        transitions[(si, A)] = [
            Transition(next_state=f"s2_{i + 1}", prob=1.0, reward=1.0 if i % 2 else 1.0,
                       duration=8.0 if i % 2 else 8.0),
        ]
        states.append(si)

    transitions[(f"s2_{k + 1}", A)] = [
        Transition(next_state=s2, prob=1.0, reward=1.0, duration=8.0),
    ]
    states.append(f"s2_{k + 1}")

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state="s3_1", prob=1.0, reward=10.0, duration=2.0),
    ]

    for i in range(1, k + 1):
        si = f"s3_{i}"
        transitions[(si, A)] = [
            Transition(next_state=f"s3_{i + 1}", prob=1.0, reward=10.0 if i % 2 else 10.0,
                       duration=2.0 if i % 2 else 2.0),
        ]
        states.append(si)

    transitions[(f"s3_{k + 1}", A)] = [
        Transition(next_state=s3, prob=1.0, reward=10.0, duration=2.0),
    ]
    states.append(f"s3_{k + 1}")

    # Action B
    actions.append(B)

    # s1, action b (self-loop)
    transitions[(s1, B)] = [
        Transition(next_state=s1, prob=1.0, reward=0.4, duration=1.0),  # reward = 2.0/5.0 = 0.4
    ]

    cfg = SMDPConfig(
        states,
        actions,
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def loopy_three_state_smdp_config() -> SMDPConfig:
    """Return the SMDPConfig that matches the provided 3-state diagram.

    States: s1, s2_1,... s_2_k, s3_i, ... s_3_k
    Actions: 0 -> action a, 1 -> action b

    - At s1:
        * action a leads to s2 with r=100, tau=1 or 9, p=0.5
        * action b leads to s3 with r=120, tau=4, p=1
    - At s2:
        * action a leads back to s1
    - At s3:
        * action leads to s1

    This can be easily modified in code if you want to try other structures.
    """

    s1, s2, s3 = "s1", "s2", "s3"
    A, B = 0, 1  # 0: action a, 1: action b

    states = list([s1, s2, s3])
    actions = list([A])
    transitions: Dict[Tuple[State, Action], List[Transition]] = {}

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=0.5, reward=100.0, duration=1.0),
        Transition(next_state=s2, prob=0.5, reward=100.0, duration=4.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state="s1", prob=1.0, reward=0.0, duration=1.0),
    ]

    # Action B
    actions.append(B)

    # s1, action b (self-loop)
    transitions[(s1, B)] = [
        Transition(next_state=s3, prob=1.0, reward=120.0, duration=4.0),
    ]

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state="s1", prob=1.0, reward=0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states,
        actions,
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def schwartz_first_loop_smdp_config() -> SMDPConfig:
    """Return the SMDPConfig that matches the Schwartz first loop example.

    States: s1, s2
    Actions: 0 -> action a, 1 -> action b

    - At s1:
        * action a leads to s2 with p=1.0, tau=1, r=0
    - At s2:
        * action a leads to s1 with p=1.0, tau=2, r=10

    """

    A, B = 0, 1
    loop_for = 49

    transitions: Dict[Tuple[State, Action], List[Transition]] = {}
    states = ["s0"]

    def one_or_minus_one():
        return 1 if random.random() < 0.5 else -1

    # one_or_minus_one = 1
    for l in range(loop_for):
        states.append(f"s{l + 1}")
        transitions[(f"s{l}", B)] = [
            Transition(next_state=f"s{l + 1}", prob=1.0, reward=one_or_minus_one, duration=1.0),
        ]
        transitions[(f"s{l}", A)] = [
            Transition(next_state=f"s{l}", prob=1.0, reward=one_or_minus_one, duration=1.0),
        ]

    transitions[(f"s{l + 1}", B)] = [
        Transition(next_state="s0", prob=1.0, reward=50.0, duration=1.0),
    ]
    transitions[(f"s{l + 1}", A)] = [
        Transition(next_state=f"s{l + 1}", prob=1.0, reward=one_or_minus_one, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=states,
        actions=[A, B],
        transitions=transitions,
        start_state=states[0],
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg


def non_stationary_simple_unichain() -> SMDPConfig:
    """Return the SMDPConfig that matches a non-stationary simple unichain example.

    States: s1, s2
    Actions: 0 -> action a, 1 -> action b

    - At s1:
        * action a leads to s2 with p=1.0, tau=1, r=0
    - At s2:
        * action a leads to s1 with p=1.0, tau=2, r=10

    """

    A, B = 0, 1

    transitions: Dict[Tuple[State, Action], List[Transition]] = {}
    s1, s2 = "s1", "s2"
    states = [s1, s2]


    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=1.0, reward=Reward(), duration=Duration()),
    ]
    transitions[(s1, B)] = [
        Transition(next_state=s2, prob=1.0, reward=10.0, duration=1.0),
    ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s1, prob=1.0, reward=0.0, duration=1.0),
    ]

    cfg = SMDPConfig(
        states=states,
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg
