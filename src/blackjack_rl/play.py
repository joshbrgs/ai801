"""Running a policy in any of the envs: the simulator, a wrapped one, or the pygame table."""

from typing import NamedTuple

import gymnasium as gym
import numpy as np

from blackjack_rl.policies import Policy
from blackjack_rl.rules import ACTION_NAMES


class Estimate(NamedTuple):
    mean: float
    stderr: float

    def __str__(self) -> str:
        return f"{self.mean:+.4f} ± {self.stderr:.4f}"


def play_episode(
    env: gym.Env, policy: Policy, seed: int | None = None, verbose: bool = False
) -> float:
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(obs, info)
        if verbose:
            print(f"hand {info['hand']}  obs={obs[:5]}  -> {ACTION_NAMES[action]}")
        obs, reward, terminated, truncated, info = env.step(action)
    if verbose:
        s = info["state"]
        print(f"hands={s.hands} bets={s.bets} dealer={s.dealer}  reward={reward:+}")
    return reward


def evaluate(env: gym.Env, policy: Policy, n: int = 100_000, seed: int = 0) -> Estimate:
    """Mean reward per round over n rounds, with its standard error."""
    env.reset(seed=seed)
    rewards = np.array([play_episode(env, policy) for _ in range(n)])
    return Estimate(float(rewards.mean()), float(rewards.std(ddof=1) / np.sqrt(n)))
