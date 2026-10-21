"""The pygame table as a Gymnasium env, so any agent can play it on screen at a watchable pace.

It has the same observation, actions and `info` as `blackjack_rl`'s simulator, so a policy or
wrapper (card counting, flattening for a DQN) built against the simulator works here unchanged.
The table's own rules are `PYGAME_TABLE_RULES`. Train with those to match what the agent sees.

Rewards come from `blackjack_rl.rules.utility` applied to the cards on the table, not from the
game's chip count, so they are in units of the original bet like the simulator's.
"""

import random
import time
from dataclasses import replace

import gymnasium as gym
import pygame
from gymnasium import spaces

from blackjack_rl.env import action_mask, observation_space
from blackjack_rl.rules import (
    ACTION_NAMES,
    DOUBLE,
    HIT,
    PYGAME_TABLE_RULES,
    SPLIT,
    STICK,
    Obs,
    State,
    newly_revealed,
    observation,
    utility,
    visible_cards,
)
from deck import Card
from main import Game

BET = 25
STARTING_CHIPS = 500
MAX_ANIMATION_WAIT = 5.0  # seconds, in case an animation never reports that it finished


class TableClosed(Exception):
    """The pygame window was closed while an agent was playing."""


def card_value(card: Card) -> int:
    """Blackjack-v1's card value: 1 for an ace, 10 for any ten-card."""
    return 1 if card.rank == "ace" else card.value()


def table_state(game: Game) -> State:
    """The cards on the table as a blackjack_rl State."""
    hands = game.split_hands if game.is_split else [game.player_hand]
    if game.is_split:
        bets, done = (1.0, 1.0), tuple(game.split_hands_complete)  # the table has one split only
    else:
        bets, done = (2.0 if game.double_down_used else 1.0,), (game.game_over,)
    if game.game_over:
        done = (True,) * len(hands)
    hole, up, *drawn = game.dealer_hand  # the table hides dealer_hand[0]
    return State(
        hands=tuple(tuple(map(card_value, hand)) for hand in hands),
        bets=bets,
        done=done,
        dealer=tuple(map(card_value, (up, hole, *drawn))),
    )


class PygameBlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        delay: float = 1.0,
        num_decks: int = 1,
        penetration: float = 0.75,
        animate: bool = True,
    ):
        self.game = Game(num_decks, penetration)
        self.game.animations_enabled = animate
        self.rules = replace(PYGAME_TABLE_RULES, num_decks=num_decks, penetration=penetration)
        self.delay = delay
        self.clock = pygame.time.Clock()
        self.action_space = spaces.Discrete(4)
        self.observation_space = observation_space()
        self.moves = {
            STICK: self.game.stand,
            HIT: self.game.hit,
            DOUBLE: self.game.double_down,
            SPLIT: self.game.split_hand,
        }

    def legal_actions(self) -> list[int]:
        g = self.game
        if g.game_over:
            return []
        return [STICK, HIT] + [DOUBLE] * g.can_double_down() + [SPLIT] * g.can_split()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        g = self.game
        shuffles = g.deck.shuffles
        if seed is not None:
            random.seed(seed)  # the table's deck uses Python's random module
            g.deck.shuffle()
        if g.player_chips < BET:  # top up so an agent can keep playing after going broke
            g.player_chips = STARTING_CHIPS
        g.player_bet, g.agent_label = BET, ""
        g.reset_game()
        self.shuffled, self.seen = g.deck.shuffles != shuffles, []
        self._show()
        return self._observe()

    def step(self, action: int):
        action = int(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {ACTION_NAMES.get(action, action)} on the table")

        g = self.game
        g.agent_label = f"Agent: {ACTION_NAMES[action]}"
        shuffles = g.deck.shuffles
        self.moves[action]()
        self.shuffled = g.deck.shuffles != shuffles
        self._show()

        terminated = g.game_over
        reward = utility(table_state(g), self.rules) if terminated else 0.0
        obs, info = self._observe()
        return obs, reward, terminated, False, info

    def close(self) -> None:
        pygame.quit()

    def _observe(self) -> tuple[Obs, dict]:
        state = table_state(self.game)
        seen = visible_cards(state)
        revealed, self.seen = newly_revealed(self.seen, seen), seen
        mask = action_mask(self.legal_actions())
        info = {
            "action_mask": mask,
            "revealed": revealed,
            "shuffled": self.shuffled,
            "cards_left": len(self.game.deck.cards),
            "hand": self.game.active_split_hand if not self.game.game_over else None,
            "state": state,
        }
        return observation(state, mask), info

    def _show(self) -> None:
        """Draw frames until the animations finish and `delay` seconds have passed."""
        start = time.monotonic()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise TableClosed
                if event.type != pygame.MOUSEBUTTONDOWN:  # the agent has the controls
                    self.game.handle_event(event)
            self.game.draw()
            pygame.display.flip()
            self.clock.tick(60 if self.delay else 0)

            elapsed = time.monotonic() - start
            animating = self.game.animations_enabled and self.game.card_animator.has_animations()
            if elapsed >= self.delay and (not animating or elapsed > MAX_ANIMATION_WAIT):
                return
