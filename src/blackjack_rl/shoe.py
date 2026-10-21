"""A finite shoe is a shuffled list of card values, dealt from the end with pop()."""

import numpy as np
from gymnasium.envs.toy_text.blackjack import deck

from blackjack_rl.rules import Rules

CARDS_PER_DECK = 52


def new_shoe(num_decks: int, np_random: np.random.Generator) -> list[int]:
    """num_decks shuffled decks. Gymnasium's `deck` is one suit, so a full deck is `deck * 4`."""
    return [int(c) for c in np_random.permutation(deck * 4 * num_decks)]


def needs_shuffle(shoe: list[int], rules: Rules) -> bool:
    """True once `rules.penetration` of the shoe has been dealt."""
    return len(shoe) <= (1 - rules.penetration) * rules.num_decks * CARDS_PER_DECK
