# blackjack_rl

A blackjack game for reinforcement learning! It takes Gymnasium's `Blackjack-v1`, which only lets
you **stick** or **hit**, and adds **double**, **split** and a **finite shoe** of real decks.

Notebooks and the pygame table in `blackjack_pygame/` all import this package, so they all play
by exactly the same rules.

## Quick start

```python
import gymnasium as gym
import blackjack_rl  # registers the environment
from blackjack_rl.play import evaluate
from blackjack_rl.policies import basic_strategy

env = gym.make(blackjack_rl.ENV_ID, num_decks=6)
print(evaluate(env, basic_strategy))  # e.g. -0.0022 ± 0.0036 per round
```

## How it fits together

```
rules.py     the game itself: what you can do, and what happens when you do it
shoe.py      where the cards come from when you play with real decks
env.py       wraps the rules up as a Gymnasium environment
policies.py  ready-made players to compare against
play.py      plays rounds and measures how well a player does
```

The rules don't know about Gymnasium, and the environment doesn't know any blackjack. It asks
`rules.py` what's legal and what happens next.

## The files

### `rules.py`: the game

Everything about a round lives in one **`State`**: the player's hands, the bet on each hand,
which hands are finished, and the dealer's cards. A `State` never changes. Each move makes a new
one.

Cards are numbers, the same as in `Blackjack-v1`: **1 is an ace** and **10 is any ten, jack,
queen or king**.

The main functions:

| Function | What it answers |
|---|---|
| `initial_state(draw)` | Deal a new round: two cards each for the player and the dealer. |
| `actions(s, rules)` | Which moves are legal right now? |
| `result(s, action, draw, rules)` | What does the table look like after this move? |
| `is_terminal(s)` | Is every hand finished? |
| `dealer_play(s, draw)` | Let the dealer draw until they reach 17. |
| `utility(s, rules)` | How much did the player win or lose this round? |

Smaller helpers:

- `active(s)` is the hand currently being played. After a split there can be several.
- `can_double(s, rules)` and `can_split(s, rules)` check those two moves.
- `hand_payoff(...)` settles one hand against the dealer. A bust always loses, even if the
  dealer busts too.
- `visible_cards(s)` lists the cards a player can see. The dealer's second (hole) card stays
  hidden until the round is over.
- `newly_revealed(before, after)` gives the cards that came into view since last time.
- `observation(s, mask)` summarises the state for an agent (see below).

`draw` is just "a function that hands you the next card". That means the rules never deal with
randomness themselves, and a test can pass in a fixed deck to check a scenario exactly.

**House rules** are set with a `Rules` value rather than by editing code:

| Setting | Default | Meaning |
|---|---|---|
| `natural` | `False` | Pay 3:2 for blackjack on the first two cards |
| `max_hands` | `4` | How many hands splitting can make (1 turns splitting off) |
| `double_after_split` | `True` | Allow doubling on a hand that came from a split |
| `hit_split_aces` | `False` | Let split aces take more than one card |
| `num_decks` | `None` | `None` is an endless deck like `Blackjack-v1`; a number uses a real shoe |
| `penetration` | `0.75` | How much of the shoe is dealt before it is reshuffled |

`PYGAME_TABLE_RULES` matches the pygame game. 
### `shoe.py`: real decks

- `new_shoe(num_decks, rng)` builds 52-card decks and shuffles them together.
- `needs_shuffle(shoe, rules)` says when enough of the shoe has been dealt to reshuffle.

### `env.py`: the Gymnasium environment

`BlackjackSplitDoubleEnv` is registered as `BlackjackSplitDouble-v0`. Any setting from the
`Rules` table can be passed straight to `gym.make`:

```python
env = gym.make(blackjack_rl.ENV_ID, num_decks=6, natural=True, max_hands=2)
```

**Actions:** `0` stick, `1` hit, `2` double, `3` split. Stick and hit keep the same numbers as
in `Blackjack-v1`. An illegal move raises an error instead of being quietly ignored.

**What the agent sees** is five numbers about the hand being played:

```
(player_total, dealer_up_card, has_usable_ace, can_double, can_split)
```

**Reward** is counted in bets. Winning a normal hand is `+1`, losing a doubled hand is `-2`,
and so on. It arrives on the step that ends the round, and every step before that pays `0`.

**`info`** comes with every step:

| Key | What it is |
|---|---|
| `action_mask` | 1 for each legal move. Pass it to `env.action_space.sample(mask=...)`. |
| `revealed` | Cards that came into view on this step |
| `shuffled` | `True` if a fresh shoe was opened |
| `cards_left` | Cards left in the shoe (`None` with an endless deck) |
| `hand` | Which hand is being played (`None` once the round is over) |
| `state` | The full `State`, including the dealer's hidden card. For debugging only! |

### `policies.py`: ready-made players

A policy is any function `(obs, info) -> action`. Four are included as baselines:

| Policy | How it plays |
|---|---|
| `random_policy` | Picks any legal move at random |
| `stick_on_17` | Hits until 17, then stops |
| `double_and_split` | `stick_on_17`, plus doubling on 10 or 11 and splitting aces and 8s |
| `basic_strategy` | The standard blackjack strategy chart, the same as the pygame game's hints |

They only read the first five observation values, so they keep working if a wrapper adds more.

### `play.py`: running rounds

- `play_episode(env, policy)` plays one round and returns the reward. Pass `verbose=True` to
  print each move.
- `evaluate(env, policy, n)` plays `n` rounds and returns the average reward with its
  **standard error**. That's how much the average could wobble by chance, so you can tell a
  real improvement from luck.

### `__init__.py`

Registers the environment and offers the most-used names at the top level: `ENV_ID`, `Rules`,
`State`, `PYGAME_TABLE_RULES` and the action constants `STICK`, `HIT`, `DOUBLE`, `SPLIT`.

## Testing

```sh
uv run pytest
```

The tests check the rules with fixed decks, the shoe's card counts, the environment's
observations, and that every policy only makes legal moves. They also play rounds on the pygame
table without opening a window.

