"""Blackjack with double, split and a finite shoe, as a Gymnasium environment.

    import gymnasium as gym
    import blackjack_rl  # registers the env

    env = gym.make(blackjack_rl.ENV_ID, num_decks=6)
"""

import gymnasium as gym

from blackjack_rl.env import BlackjackSplitDoubleEnv
from blackjack_rl.rules import (
    ACTION_NAMES,
    DEFAULT_RULES,
    DOUBLE,
    HIT,
    PYGAME_TABLE_RULES,
    SPLIT,
    STICK,
    Rules,
    State,
)

ENV_ID = "BlackjackSplitDouble-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="blackjack_rl.env:BlackjackSplitDoubleEnv")

__all__ = [
    "ACTION_NAMES", "DEFAULT_RULES", "DOUBLE", "ENV_ID", "HIT", "PYGAME_TABLE_RULES", "SPLIT",
    "STICK", "BlackjackSplitDoubleEnv", "Rules", "State",
]  # fmt: skip
