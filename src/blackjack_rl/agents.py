"""The agents trained in notebooks/AI_801_Blackjack.ipynb, as policies `(obs, info) -> action`.

Each one needs the env it trained on: 6 decks, and the counting wrapper for the agents that count.

    ql_no_count    no wrapper
    ql_3_buckets   counting.hilo_3
    ql_9_buckets   counting.hilo_9
    dqn            counting.hilo_3 or hilo_9 (it reads info['true_count'], not the bucket)

A saved model is loaded the first time its policy is called.
"""

import pickle
from functools import cache
from pathlib import Path

import numpy as np
import torch

from blackjack_rl.policies import basic_strategy

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
TRAIN_ROUNDS = 3_000_000


@cache
def load(name: str):
    """A saved agent: a Q-table dict or a torch network. The pickle also holds its learning curve."""
    with open(MODEL_DIR / f"{name}_{TRAIN_ROUNDS}.pkl", "rb") as f:
        agent, _curve = pickle.load(f)
    return agent


def encode(observation: tuple, info: dict) -> np.ndarray:
    """Turn the observation and true count into the network's 46 inputs."""
    player_sum, dealer_card, usable_ace, can_double, can_split = observation[:5]
    x = np.zeros(46, dtype=np.float32)
    x[min(player_sum, 31)] = 1  # slots 0-31: player total (a bust can pass 21)
    x[32 + dealer_card - 1] = 1  # slots 32-41: dealer card, A to 10
    x[42:45] = usable_ace, can_double, can_split  # slots 42-44: yes/no flags
    x[45] = np.clip(info["true_count"], -6, 6) / 6  # slot 45: true count, scaled to [-1, 1]
    return x


def best_legal(values: np.ndarray, info: dict) -> int:
    legal = np.flatnonzero(info["action_mask"])
    return int(legal[np.argmax(values[legal])])


def q_table_action(name: str, key_length: int, observation: tuple, info: dict) -> int:
    if len(observation) != key_length:
        raise ValueError(f"{name} needs a {key_length}-value observation, got {len(observation)}: "
                         "check the counting wrapper (see this module's docstring)")
    values = load(name).get(tuple(int(v) for v in observation[:key_length]))
    if values is None:  # a state never seen in training, e.g. under the table's split rules
        return basic_strategy(observation, info)
    return best_legal(values, info)


def ql_no_count(observation: tuple, info: dict) -> int:
    return q_table_action("ql_no_count", 5, observation, info)


def ql_3_buckets(observation: tuple, info: dict) -> int:
    return q_table_action("ql_3_buckets", 6, observation, info)


def ql_9_buckets(observation: tuple, info: dict) -> int:
    return q_table_action("ql_9_buckets", 6, observation, info)


def dqn(observation: tuple, info: dict) -> int:
    if "true_count" not in info:
        raise ValueError("dqn reads info['true_count']: wrap the env with blackjack_rl.counting")
    with torch.no_grad():
        q_values = load("dqn")(torch.from_numpy(encode(observation, info))).numpy()
    return best_legal(q_values, info)
