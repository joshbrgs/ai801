from collections import Counter

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.toy_text.blackjack import deck
from gymnasium.wrappers import FlattenObservation

from blackjack_rl import ENV_ID, BlackjackSplitDoubleEnv
from blackjack_rl.play import evaluate, play_episode
from blackjack_rl.policies import (
    basic_strategy,
    double_and_split,
    random_policy,
    stick_on_17,
)
from blackjack_rl.shoe import CARDS_PER_DECK


def play_random_steps(env: gym.Env, n: int):
    obs, info = env.reset(seed=0)
    for _ in range(n):
        yield obs, info
        obs, _, terminated, _, info = env.step(env.action_space.sample(mask=info["action_mask"]))
        if terminated:
            yield obs, info
            obs, info = env.reset()


@pytest.mark.parametrize("num_decks", [None, 1, 6])
def test_observations_stay_in_the_space(num_decks):
    env = gym.make(ENV_ID, num_decks=num_decks)
    for obs, _ in play_random_steps(env, 5_000):
        assert env.observation_space.contains(obs), obs


def test_illegal_actions_raise():
    env = BlackjackSplitDoubleEnv()
    _, info = env.reset(seed=0)
    illegal = np.flatnonzero(info["action_mask"] == 0)
    with pytest.raises(ValueError):
        env.step(int(illegal[0]))


def test_one_deck_shoe_holds_one_deck_and_reshuffles_at_penetration():
    env = BlackjackSplitDoubleEnv(num_decks=1, penetration=0.75)
    _, info = env.reset(seed=0)
    s = info["state"]
    dealt = [c for hand in s.hands for c in hand] + list(s.dealer)
    assert Counter(env.shoe + dealt) == Counter(deck * 4)

    cards_left = [env.reset()[1]["cards_left"] for _ in range(24)]  # 4 cards per round
    assert min(cards_left) >= 0.25 * CARDS_PER_DECK - 4
    assert cards_left.count(CARDS_PER_DECK - 4) >= 2


def test_every_card_dealt_from_a_shoe_is_revealed_exactly_once():
    env = BlackjackSplitDoubleEnv(num_decks=8, penetration=1.0)  # no reshuffle in 60 steps
    revealed: list[int] = []
    for _, info in play_random_steps(env, 60):
        assert not (info["shuffled"] and revealed)
        revealed += info["revealed"]
        if info["hand"] is None:
            dealt = 8 * CARDS_PER_DECK - len(env.shoe)
            assert len(revealed) == dealt


def test_flattened_observations_are_ready_for_a_dqn():
    env = FlattenObservation(gym.make(ENV_ID, num_decks=6))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (32 + 11 + 2 + 2 + 2,)


@pytest.mark.parametrize(
    "policy", [random_policy, stick_on_17, double_and_split, basic_strategy]
)
def test_policies_only_choose_legal_actions(policy):
    env = gym.make(ENV_ID, num_decks=6, max_hands=2, double_after_split=False)
    for _ in range(2_000):
        play_episode(env, policy)  # the env raises on an illegal action


def test_basic_strategy_beats_the_simpler_policies():
    env = gym.make(ENV_ID, natural=True)
    scores = {p.__name__: evaluate(env, p, n=50_000).mean for p in (stick_on_17, basic_strategy)}
    assert scores["basic_strategy"] > scores["stick_on_17"]
