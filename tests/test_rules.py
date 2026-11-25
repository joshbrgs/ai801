import pytest

from blackjack_rl.rules import (
    DOUBLE,
    HIT,
    PYGAME_TABLE_RULES,
    SPLIT,
    STICK,
    Rules,
    State,
    actions,
    active,
    dealer_play,
    hand_payoff,
    is_terminal,
    newly_revealed,
    result,
    utility,
    visible_cards,
)


def deck_of(*cards: int):
    return iter(cards).__next__


def start(hand, dealer=(10, 7)) -> State:
    return State(hands=(hand,), bets=(1.0,), done=(False,), dealer=dealer)


def test_split_then_double_then_stick():
    s = result(start((8, 8)), SPLIT, deck_of(3, 10))
    assert s.hands == ((8, 3), (8, 10)) and active(s) == 0

    s = result(s, DOUBLE, deck_of(10))  # 8+3 doubled, draw 10 -> 21 for 2 bets
    s = result(s, STICK, deck_of())  # 18 stands
    assert is_terminal(s) and s.bets == (2.0, 1.0)
    assert utility(dealer_play(s, deck_of())) == 2 + 1  # 21 and 18 both beat 17


def test_split_aces_stand_and_a_split_21_is_not_a_natural():
    s = result(start((1, 1), dealer=(10, 9)), SPLIT, deck_of(10, 5))
    assert s.hands == ((1, 10), (1, 5)) and is_terminal(s)
    assert utility(s, Rules(natural=True)) == 1 - 1  # 21 beats 19 at 1:1, soft 16 loses


def test_split_aces_can_be_hit_when_the_rules_allow():
    s = result(start((1, 1)), SPLIT, deck_of(2, 3), PYGAME_TABLE_RULES)
    assert active(s) == 0 and HIT in actions(s, PYGAME_TABLE_RULES)


def test_bust_loses_even_if_the_dealer_busts():
    assert hand_payoff((10, 6, 9), 1.0, (10, 6, 8), natural_bonus=False) == -1


def test_natural_pays_3_to_2_only_when_enabled():
    s = start((1, 10), dealer=(10, 7))._replace(done=(True,))
    assert utility(s, Rules(natural=True)) == 1.5
    assert utility(s, Rules(natural=False)) == 1


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        (Rules(), [STICK, HIT, DOUBLE, SPLIT]),
        (Rules(max_hands=1), [STICK, HIT, DOUBLE]),
    ],
)
def test_legal_actions_on_a_pair(rules, expected):
    assert actions(start((8, 8)), rules) == expected


def test_house_rules_limit_resplitting_and_doubling_after_a_split():
    s = result(start((8, 8)), SPLIT, deck_of(8, 2), PYGAME_TABLE_RULES)
    assert actions(s, PYGAME_TABLE_RULES) == [STICK, HIT]  # no resplit, no double after split
    assert actions(s, Rules()) == [STICK, HIT, DOUBLE, SPLIT]


def test_hole_card_is_hidden_until_the_round_ends():
    s = start((8, 3), dealer=(10, 6))
    assert visible_cards(s) == [8, 3, 10]
    s = result(s, STICK, deck_of())
    assert sorted(visible_cards(s)) == [3, 6, 8, 10]


def test_newly_revealed_survives_the_reordering_of_a_split():
    before = visible_cards(start((8, 8)))
    after = visible_cards(result(start((8, 8)), SPLIT, deck_of(3, 4)))
    assert sorted(newly_revealed(before, after)) == [3, 4]


@pytest.mark.parametrize(
    "kwargs", [{"max_hands": 0}, {"num_decks": 0}, {"penetration": 0}, {"penetration": 1.5}]
)
def test_bad_rules_fail_loudly(kwargs):
    with pytest.raises(ValueError):
        Rules(**kwargs)
