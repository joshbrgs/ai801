"""Fixed policies, used as baselines and for watching the pygame table.

A policy is any callable `(obs, info) -> action`. It reads only the first five observation
fields, so it also works under wrappers that append to the observation, such as card counting.
"""

from collections.abc import Callable

import numpy as np

from blackjack_rl.rules import DOUBLE, HIT, SPLIT, STICK

Policy = Callable[[tuple, dict], int]

_rng = np.random.default_rng()


def random_policy(obs: tuple, info: dict) -> int:
    """A uniformly random legal action."""
    return int(_rng.choice(np.flatnonzero(info["action_mask"])))


def stick_on_17(obs: tuple, info: dict) -> int:
    return STICK if obs[0] >= 17 else HIT


def double_and_split(obs: tuple, info: dict) -> int:
    """stick_on_17, plus doubling on 10 and 11 and splitting aces and 8s."""
    player_sum, dealer, ace, can_dbl, can_spl = obs[:5]
    if can_spl and (ace or player_sum == 16):  # a pair with a usable ace is A-A; 16 is 8-8
        return SPLIT
    if can_dbl and player_sum in (10, 11) and dealer not in (1, 10):
        return DOUBLE
    return stick_on_17(obs, info)


# Basic strategy, ported from blackjack_pygame/basic_strategy.py. The dealer's ace counts as 11.
TWO_TO = {n: set(range(2, n + 1)) for n in range(2, 11)}  # TWO_TO[6] == {2, 3, 4, 5, 6}

SPLIT_AGAINST = {1: TWO_TO[10] | {11}, 8: TWO_TO[10] | {11}, 2: TWO_TO[7], 3: TWO_TO[7],
                 6: TWO_TO[6], 7: TWO_TO[7], 9: TWO_TO[6] | {8, 9}}  # fmt: skip
SOFT_DOUBLE_AGAINST = {13: {5, 6}, 14: {5, 6}, 15: {4, 5, 6}, 16: {4, 5, 6},
                       17: {3, 4, 5, 6}, 18: TWO_TO[6]}  # fmt: skip
HARD_DOUBLE_AGAINST = {9: {3, 4, 5, 6}, 10: TWO_TO[9], 11: TWO_TO[10]}


def should_stand(player_sum: int, up: int, soft: bool) -> bool:
    if soft:
        return player_sum >= 19 or (player_sum == 18 and up <= 8)
    if player_sum >= 17:
        return True
    if player_sum >= 13:
        return up <= 6
    return player_sum == 12 and 4 <= up <= 6


def basic_strategy(obs: tuple, info: dict) -> int:
    player_sum, dealer, soft, can_dbl, can_spl = obs[:5]
    up = 11 if dealer == 1 else dealer
    pair = 1 if soft else player_sum // 2
    if can_spl and up in SPLIT_AGAINST.get(pair, ()):
        return SPLIT
    double_table = SOFT_DOUBLE_AGAINST if soft else HARD_DOUBLE_AGAINST
    if can_dbl and up in double_table.get(player_sum, ()):
        return DOUBLE
    return STICK if should_stand(player_sum, up, bool(soft)) else HIT
