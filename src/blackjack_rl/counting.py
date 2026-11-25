"""Hi-Lo card counting as a Gymnasium wrapper.

    env = HiLoCountWrapper(gym.make(blackjack_rl.ENV_ID, num_decks=6))

The wrapper adds the count bucket as a 6th observation value and puts the running and true count
in `info`. The first five values are unchanged, so `basic_strategy` still works on the wrapped env.
"""

import math

import gymnasium as gym
from gymnasium import spaces

# How much each card adds to the Hi-Lo count when it is seen (1 = Ace, 10 = ten/J/Q/K).
HI_LO_VALUES = {1: -1, 2: +1, 3: +1, 4: +1, 5: +1, 6: +1, 7: 0, 8: 0, 9: 0, 10: -1}
CARDS_PER_DECK = 52


class HiLoCountWrapper(gym.Wrapper):
    """Adds the Hi-Lo true count, rounded down and capped into a bucket, as a 6th number
    the agent sees (its observation)."""

    def __init__(self, env: gym.Env, min_bucket: int = -1, max_bucket: int = 1):
        super().__init__(env)
        if env.unwrapped.rules.num_decks is None:
            raise ValueError("Card counting needs a finite shoe: pass num_decks=... to gym.make")
        if min_bucket > 0 or max_bucket < 0:
            raise ValueError("The bucket range must include 0")
        self.min_bucket = min_bucket
        self.max_bucket = max_bucket
        self.running_count = 0

        # Discrete(n, start=s) is a whole number from s to s + n - 1, so the bucket can be negative
        num_buckets = max_bucket - min_bucket + 1
        self.observation_space = spaces.Tuple(
            tuple(env.observation_space.spaces) + (spaces.Discrete(num_buckets, start=min_bucket),)
        )

    def _update_count(self, info: dict) -> None:
        """Start again at 0 on a new shoe, then add the Hi-Lo value of every newly visible card."""
        if info["shuffled"]:
            self.running_count = 0
        for card in info["revealed"]:
            self.running_count += HI_LO_VALUES[card]

    def true_count(self, cards_left: int) -> float:
        """Running count divided by the number of decks still in the shoe."""
        decks_left = max(cards_left, 1) / CARDS_PER_DECK  # never fewer than 1 card
        return self.running_count / decks_left

    def bucket(self, true_count: float) -> int:
        """Round down, then cap between min_bucket and max_bucket.
        With the defaults (-1 to +1): 2.7 -> 1,  0.5 -> 0,  -0.4 -> -1,  -3.0 -> -1."""
        return int(min(max(math.floor(true_count), self.min_bucket), self.max_bucket))

    def _add_count(self, obs: tuple, info: dict) -> tuple[tuple, dict]:
        """Append the bucket to the observation and put the raw numbers in info."""
        true_count = self.true_count(info["cards_left"])
        info["running_count"] = self.running_count
        info["true_count"] = true_count
        return tuple(obs) + (self.bucket(true_count),), info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # A bet is chosen before the cards are dealt, so save the true count from just before
        # this deal (0 if a new shoe was opened). cards_left no longer includes the 4 dealt cards.
        if info["shuffled"]:
            info["true_count_before_deal"] = 0.0
        else:
            info["true_count_before_deal"] = self.true_count(info["cards_left"] + 4)

        # The dealer's face down card is counted later, once it is revealed
        self._update_count(info)
        return self._add_count(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_count(info)
        obs, info = self._add_count(obs, info)
        return obs, reward, terminated, truncated, info


# For autoplay.py's --wrapper, which takes a callable env -> env
def hilo_3(env: gym.Env) -> HiLoCountWrapper:
    """The 3 count buckets (-1 to +1) the default agents trained with."""
    return HiLoCountWrapper(env)


def hilo_9(env: gym.Env) -> HiLoCountWrapper:
    """The 9 count buckets (-3 to +5) the fine-bucket Q-table trained with."""
    return HiLoCountWrapper(env, min_bucket=-3, max_bucket=5)
