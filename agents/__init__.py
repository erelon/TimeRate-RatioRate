# Agents package initialization
from .base import Agent, MAX_REWARDS
from .oracle import OracleAgent
from .random_agent import RandomAgent
from .q_learning import QLearningAgent, ContinuousQLearningAgent, HarmonicQAgent
from .r_learning import ContinuousRLAgent,RLAgent 
from .smart_r import SMARTRLAgent, SMARTEMARLAgent, AdaptiveSMARTRLAgent
from .harmonic_r import HarmonicRLAgent, HarmonicRLAgent2, AdaptiveHarmonicRLAgent, AdaptiveHarmonicRLAgent2
from .bandits import MAB, ContinuesMAB, UCB, ContinuosUCB
from .multirho import StateSMARTRLAgent
from .harmonic_average import hma

__all__ = [
    'Agent', 'hma', 'MAX_REWARDS', 'OracleAgent', 'RandomAgent', 'QLearningAgent', 'ContinuousQLearningAgent', 'HarmonicQAgent',
    'RLAgent', 'ContinuousRLAgent', 'SMARTRLAgent', 'SMARTEMARLAgent', 'StateSMARTRLAgent', 'AdaptiveSMARTRLAgent',
    'HarmonicRLAgent', 'HarmonicRLAgent2', 'AdaptiveHarmonicRLAgent', 'AdaptiveHarmonicRLAgent2', 'MAB', 'ContinuesMAB', 'UCB', 'ContinuosUCB'
]

