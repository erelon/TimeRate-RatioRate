from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Hashable, Optional
import random


State = Hashable
Action = int


@dataclass
class Transition:
    next_state: State
    prob: float
    reward: float
    duration: float  # tau


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
        self.rng = random.Random(seed)
        self.state: State = config.start_state
        self.time_elapsed: float = 0.0
        self.total_reward: float = 0.0
        self.terminal_states = set(config.terminal_states or [])

        # For convenience when constructing agents
        self.states = list(config.states)
        self.action_space = sorted(set(config.actions))

    def reset(self) -> State:
        self.state = self.config.start_state
        self.time_elapsed = 0.0
        self.total_reward = 0.0
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
        probs = [t.prob for t in transitions]
        r = self.rng.random()
        cumulative = 0.0
        chosen: Transition = transitions[-1]
        for t, p in zip(transitions, probs):
            cumulative += p
            if r <= cumulative:
                chosen = t
                break

        self.state = chosen.next_state
        self.time_elapsed += chosen.duration
        self.total_reward += chosen.reward

        done = self.state in self.terminal_states
        info = {"prob": chosen.prob}
        return self.state, chosen.reward, chosen.duration, done, info


def default_three_state_smdp_config() -> SMDPConfig:
    """Return the SMDPConfig that matches the provided 3-state diagram.

    States: s1, s2, s3
    Actions: 0 -> action a, 1 -> action b

    From the diagram (as interpreted):
    - At s1:
        * action a leads to s2 with p=0.5, tau=1, r=0
        * action a leads to s3 with p=0.5, tau=1, r=0
        * action b leads to s1 with p=1.0, tau=1, r=2/5
    - At s2:
        * action a leads to s2 with p=1.0, tau=1, r=1
    - At s3:
        * action a leads to s3 with p=1.0, tau=2, r=0

    This can be easily modified in code if you want to try other structures.
    """

    s1, s2, s3 = "s1", "s2", "s3"
    A, B = 0, 1  # 0: action a, 1: action b

    transitions: Dict[Tuple[State, Action], List[Transition]] = {}

    # s1, action a
    transitions[(s1, A)] = [
        Transition(next_state=s2, prob=0.5, reward=0.0, duration=1.0),
        Transition(next_state=s3, prob=0.5, reward=0.0, duration=1.0),
    ]

    # s1, action b (self-loop)
    # transitions[(s1, B)] = [
    #     Transition(next_state=s1, prob=1.0, reward=2.0 / 5.0, duration=1.0),
    # ]

    # s2, action a
    transitions[(s2, A)] = [
        Transition(next_state=s2, prob=1.0, reward=1.0, duration=1.0),
    ]

    # s3, action a
    transitions[(s3, A)] = [
        Transition(next_state=s3, prob=1.0, reward=50.0, duration=100.0),
    ]

    cfg = SMDPConfig(
        states=[s1, s2, s3],
        actions=[A, B],
        transitions=transitions,
        start_state=s1,
        terminal_states=[],  # continuing task; episodes cut off in runner
    )
    return cfg
