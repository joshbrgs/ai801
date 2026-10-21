"""Watch an agent play the pygame table.

    uv run python autoplay.py                                   # basic strategy, 1 move a second
    uv run python autoplay.py --policy blackjack_rl.policies:random_policy --delay 0.5
    uv run python autoplay.py --decks 6 --wrapper my_agents:flatten --policy my_agents:dqn

--policy is a `module:name` callable `(obs, info) -> action`. Each --wrapper is a `module:name`
callable `env -> env`, applied in order, e.g. a card-counting or flattening wrapper the agent was
trained with. Closing the window stops the run.
"""

import argparse
import importlib
import itertools

from blackjack_rl.play import play_episode
from game_env import PygameBlackjackEnv, TableClosed


def load(spec: str):
    module, _, name = spec.partition(":")
    if not name:
        raise argparse.ArgumentTypeError(f"expected module:name, got {spec!r}")
    return getattr(importlib.import_module(module), name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", type=load, default="blackjack_rl.policies:basic_strategy")
    parser.add_argument("--wrapper", type=load, action="append", default=[])
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between moves")
    parser.add_argument("--decks", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=0, help="0 plays until the window closes")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = PygameBlackjackEnv(delay=args.delay, num_decks=args.decks)
    for wrap in args.wrapper:
        env = wrap(env)

    rounds = itertools.count(1) if args.rounds == 0 else range(1, args.rounds + 1)
    total = 0.0
    try:
        for n in rounds:
            reward = play_episode(env, args.policy, seed=args.seed if n == 1 else None)
            total += reward
            print(f"round {n}: {reward:+.1f}  total {total:+.1f}  ({total / n:+.3f} per round)")
    except TableClosed:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
