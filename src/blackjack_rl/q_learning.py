"""Tabular Q-learning for the BlackjackSplitDouble environment."""

from collections import defaultdict
from collections.abc import Mapping
from typing import TypeAlias

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from blackjack_rl.policies import Policy

StateKey: TypeAlias = tuple[int, int, int, int, int]
QTable: TypeAlias = defaultdict[StateKey, NDArray[np.float64]]


def _state_key(observation: tuple) -> StateKey:
    """Convert the environment's five observable values to a hashable Q-table key."""
    return (
        int(observation[0]),
        int(observation[1]),
        int(observation[2]),
        int(observation[3]),
        int(observation[4]),
    )


def _choose_action(
    q_table: Mapping[StateKey, NDArray[np.float64]],
    observation: tuple,
    info: dict,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Choose an epsilon-greedy move while respecting the environment's legal-action mask."""
    legal_actions = np.flatnonzero(info["action_mask"])
    if legal_actions.size == 0:
        raise ValueError("The environment provided no legal actions for a non-terminal state.")

    if rng.random() < epsilon:
        return int(rng.choice(legal_actions))

    values = q_table.get(_state_key(observation))
    if values is None:
        values = np.zeros(len(info["action_mask"]), dtype=np.float64)
    legal_values = values[legal_actions]
    best_actions = legal_actions[legal_values == np.max(legal_values)]
    return int(rng.choice(best_actions))


def train_q_learning(
    env: gym.Env,
    episodes: int = 300_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    seed: int | None = 42,
) -> tuple[QTable, NDArray[np.float64]]:
    """Train a tabular agent and return its Q-table and reward from each episode."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not 0 <= epsilon_min <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= epsilon_min <= epsilon_start <= 1")
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError("Q-learning requires an environment with a discrete action space")
    number_of_actions = action_space.n

    q_table: QTable = defaultdict(lambda: np.zeros(number_of_actions, dtype=np.float64))
    rng = np.random.default_rng(seed)
    episode_rewards = np.zeros(episodes, dtype=np.float64)
    decay_episodes = max(1, int(0.8 * episodes))

    for episode in range(episodes):
        progress = min(episode / decay_episodes, 1.0)
        epsilon = epsilon_start + progress * (epsilon_min - epsilon_start)
        observation, info = env.reset(seed=seed if episode == 0 else None)
        terminated = truncated = False

        while not (terminated or truncated):
            state = _state_key(observation)
            action = _choose_action(q_table, observation, info, epsilon, rng)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            target = float(reward)
            if not done:
                next_state = _state_key(next_observation)
                next_legal = np.flatnonzero(next_info["action_mask"])
                target += gamma * np.max(q_table[next_state][next_legal])

            q_table[state][action] += alpha * (target - q_table[state][action])
            episode_rewards[episode] += reward
            observation, info = next_observation, next_info

    return q_table, episode_rewards


def make_q_policy(q_table: Mapping[StateKey, NDArray[np.float64]]) -> Policy:
    """Create a greedy policy compatible with ``blackjack_rl.play.evaluate``."""
    rng = np.random.default_rng(0)

    def policy(observation: tuple, info: dict) -> int:
        return _choose_action(q_table, observation, info, epsilon=0.0, rng=rng)

    return policy
