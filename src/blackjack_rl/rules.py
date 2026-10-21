from collections import Counter
from dataclasses import dataclass
from typing import Callable, NamedTuple

from gymnasium.envs.toy_text.blackjack import cmp, is_bust, is_natural, score, sum_hand, usable_ace

STICK, HIT, DOUBLE, SPLIT = range(4)
ACTION_NAMES = {STICK: "stick", HIT: "hit", DOUBLE: "double", SPLIT: "split"}

Hand = tuple[int, ...]
Draw = Callable[[], int]
Obs = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Rules:
    """House rules for a table. The defaults are the notebook's rules."""

    natural: bool = False  # pay 3:2 on a natural in an unsplit hand
    max_hands: int = 4  # hands a round can split into (1 turns splitting off)
    double_after_split: bool = True
    hit_split_aces: bool = False  # False: split aces get one card each and stand
    num_decks: int | None = None  # None: infinite deck, as in Blackjack-v1
    penetration: float = 0.75  # fraction of a finite shoe dealt before it is reshuffled

    def __post_init__(self) -> None:
        if self.max_hands < 1:
            raise ValueError(f"max_hands must be >= 1, got {self.max_hands}")
        if self.num_decks is not None and self.num_decks < 1:
            raise ValueError(f"num_decks must be >= 1 or None, got {self.num_decks}")
        if not 0 < self.penetration <= 1:
            raise ValueError(f"penetration must be in (0, 1], got {self.penetration}")


DEFAULT_RULES = Rules()

# The vendored pygame table: one split, no double after it, split aces may be hit, 1:1 naturals
PYGAME_TABLE_RULES = Rules(max_hands=2, double_after_split=False, hit_split_aces=True, num_decks=1)


class State(NamedTuple):
    hands: tuple[Hand, ...]  # player hands, in play order
    bets: tuple[float, ...]  # stake on each hand
    done: tuple[bool, ...]  # has each hand finished?
    dealer: Hand  # dealer[0] is the up card, dealer[1] the hole card


def initial_state(draw: Draw) -> State:
    return State(hands=((draw(), draw()),), bets=(1.0,), done=(False,), dealer=(draw(), draw()))


def active(s: State) -> int | None:
    """Index of the hand being played, or None once every hand is finished."""
    return next((i for i, d in enumerate(s.done) if not d), None)


def is_terminal(s: State) -> bool:
    return active(s) is None


def replace_at(t: tuple, i: int, *items) -> tuple:
    """Copy of t with t[i] swapped for items (one item replaces, two items splits)."""
    return t[:i] + items + t[i + 1 :]


def can_double(s: State, rules: Rules = DEFAULT_RULES) -> bool:
    is_split = len(s.hands) > 1
    return len(s.hands[active(s)]) == 2 and (rules.double_after_split or not is_split)


def can_split(s: State, rules: Rules = DEFAULT_RULES) -> bool:
    hand = s.hands[active(s)]
    return len(hand) == 2 and hand[0] == hand[1] and len(s.hands) < rules.max_hands


def actions(s: State, rules: Rules = DEFAULT_RULES) -> list[int]:
    if is_terminal(s):
        return []
    return [STICK, HIT] + [DOUBLE] * can_double(s, rules) + [SPLIT] * can_split(s, rules)


def result(s: State, a: int, draw: Draw, rules: Rules = DEFAULT_RULES) -> State:
    i = active(s)
    hand, bet = s.hands[i], s.bets[i]

    if a == STICK:
        return s._replace(done=replace_at(s.done, i, True))

    if a == HIT:
        hand = hand + (draw(),)
        return s._replace(
            hands=replace_at(s.hands, i, hand), done=replace_at(s.done, i, is_bust(hand))
        )

    if a == DOUBLE:
        return s._replace(
            hands=replace_at(s.hands, i, hand + (draw(),)),
            bets=replace_at(s.bets, i, 2 * bet),
            done=replace_at(s.done, i, True),
        )

    if a == SPLIT:
        card = hand[0]
        stand = card == 1 and not rules.hit_split_aces
        return s._replace(
            hands=replace_at(s.hands, i, (card, draw()), (card, draw())),
            bets=replace_at(s.bets, i, bet, bet),
            done=replace_at(s.done, i, stand, stand),
        )

    raise ValueError(f"unknown action {a}")


def dealer_play(s: State, draw: Draw) -> State:
    """The dealer draws to 17 and stands on soft 17, as in Blackjack-v1."""
    dealer = s.dealer
    while sum_hand(dealer) < 17:
        dealer = dealer + (draw(),)
    return s._replace(dealer=dealer)


def hand_payoff(hand: Hand, bet: float, dealer: Hand, natural_bonus: bool) -> float:
    if is_bust(hand):
        return -bet
    if natural_bonus and is_natural(hand) and not is_natural(dealer):
        return 1.5 * bet
    return bet * cmp(score(hand), score(dealer))


def utility(s: State, rules: Rules = DEFAULT_RULES) -> float:
    """Total payoff of a finished round, in units of the original bet."""
    bonus = rules.natural and len(s.hands) == 1
    return sum(hand_payoff(h, b, s.dealer, bonus) for h, b in zip(s.hands, s.bets))


def visible_cards(s: State) -> list[int]:
    """Cards a player at the table can see: the hole card stays face down until the round ends."""
    dealer = s.dealer if is_terminal(s) else s.dealer[:1]
    return [card for hand in s.hands for card in hand] + list(dealer)


def newly_revealed(before: list[int], after: list[int]) -> list[int]:
    """Cards visible now that weren't before. A split reorders hands, so compare as multisets."""
    return list((Counter(after) - Counter(before)).elements())


def observation(s: State, mask) -> Obs:
    """Blackjack-v1's (player_sum, dealer_card, usable_ace) for the active hand, plus whether
    doubling and splitting are legal. After the round it describes the last hand."""
    i = active(s)
    hand = s.hands[i if i is not None else -1]
    return (
        sum_hand(hand), s.dealer[0], int(usable_ace(hand)), int(mask[DOUBLE]), int(mask[SPLIT])
    )
