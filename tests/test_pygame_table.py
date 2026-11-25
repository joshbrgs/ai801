"""Plays the pygame table headless and checks it against the game's own results."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).parents[1] / "blackjack_pygame"))

from blackjack_rl.play import play_episode  # noqa: E402
from blackjack_rl.policies import basic_strategy, random_policy  # noqa: E402
from game_env import PygameBlackjackEnv  # noqa: E402

ONE_HAND_OUTCOMES = {"Player Wins!": 1, "Push! It's a tie.": 0, "Dealer Wins.": -1,
                     "Bust! Dealer Wins.": -1}  # fmt: skip


@pytest.fixture(scope="module")
def table():
    env = PygameBlackjackEnv(delay=0, num_decks=2, animate=False)
    yield env
    env.close()


@pytest.mark.parametrize("policy", [basic_strategy, random_policy])
def test_rewards_match_the_tables_own_result(table, policy):
    for _ in range(300):
        reward = play_episode(table, policy)
        g = table.game
        if not g.is_split:
            stake = 2 if g.double_down_used else 1
            assert reward == stake * ONE_HAND_OUTCOMES[g.winner], (g.winner, reward)


def test_the_table_reveals_every_card_it_deals(table):
    obs, info = table.reset(seed=1)
    revealed = list(info["revealed"])
    for _ in range(200):
        terminated = False
        while not terminated:
            obs, _, terminated, _, info = table.step(basic_strategy(obs, info))
            revealed = [] if info["shuffled"] else revealed
            revealed += info["revealed"]
        deck = table.game.deck
        assert len(revealed) == 52 * deck.num_decks - len(deck.cards)
        obs, info = table.reset()
        revealed = [] if info["shuffled"] else revealed
        revealed += info["revealed"]
