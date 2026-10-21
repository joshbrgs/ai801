"""Blackjack-v1 extended with double, split and an optional finite shoe."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.toy_text.blackjack import draw_card

from blackjack_rl.rules import (
    ACTION_NAMES,
    DEFAULT_RULES,
    Obs,
    Rules,
    State,
    actions,
    active,
    dealer_play,
    initial_state,
    is_terminal,
    newly_revealed,
    observation,
    result,
    utility,
    visible_cards,
)
from blackjack_rl.shoe import needs_shuffle, new_shoe


def observation_space() -> spaces.Tuple:
    """(player_sum, dealer_card, usable_ace, can_double, can_split)"""
    return spaces.Tuple(
        (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2),
         spaces.Discrete(2), spaces.Discrete(2))
    )  # fmt: skip


def action_mask(legal: list[int]) -> np.ndarray:
    mask = np.zeros(4, dtype=np.int8)
    mask[legal] = 1
    return mask


class BlackjackSplitDoubleEnv(gym.Env):
    """Actions 0-3 are stick, hit, double, split. Rewards are in units of the original bet.

    `info` on every step carries:
      action_mask  legal actions, for `action_space.sample(mask=...)` or a DQN
      revealed     cards that became visible this step (for card counting)
      shuffled     True if a fresh shoe was opened this step
      cards_left   cards left in the shoe (None for an infinite deck)
      hand         index of the active hand (None once the round is over)
      state        the full State, hole card included: for debugging, not for agents
    """

    metadata = {"render_modes": []}

    def __init__(self, rules: Rules = DEFAULT_RULES, **overrides):
        self.rules = replace(rules, **overrides)
        self.action_space = spaces.Discrete(4)
        self.observation_space = observation_space()
        self.shoe: list[int] = []
        self.shuffled = False

    def draw(self) -> int:
        if self.rules.num_decks is None:
            return draw_card(self.np_random)
        if not self.shoe:  # a long split round emptied it
            self.shuffle()
        return self.shoe.pop()

    def shuffle(self) -> None:
        self.shoe = new_shoe(self.rules.num_decks, self.np_random)
        self.shuffled = True

    def legal_actions(self) -> list[int]:
        return actions(self.state, self.rules)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.shuffled, self.seen = False, []
        if self.rules.num_decks and (seed is not None or needs_shuffle(self.shoe, self.rules)):
            self.shuffle()
        self.state = initial_state(self.draw)
        return self._observe()

    def step(self, action: int):
        action = int(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {ACTION_NAMES.get(action, action)} in {self.state}")

        self.shuffled = False
        self.state = result(self.state, action, self.draw, self.rules)
        terminated = is_terminal(self.state)
        reward = 0.0
        if terminated:
            self.state = dealer_play(self.state, self.draw)
            reward = utility(self.state, self.rules)
        obs, info = self._observe()
        return obs, reward, terminated, False, info

    def _observe(self) -> tuple[Obs, dict]:
        """Observation and info for the current state, recording which cards are now visible."""
        seen = visible_cards(self.state)
        revealed, self.seen = newly_revealed(self.seen, seen), seen
        mask = action_mask(self.legal_actions())
        info = {
            "action_mask": mask,
            "revealed": revealed,
            "shuffled": self.shuffled,
            "cards_left": len(self.shoe) if self.rules.num_decks else None,
            "hand": active(self.state),
            "state": self.state,
        }
        return observation(self.state, mask), info
