# deck.py
# NathanGr33n
# June 20, 2025

import random  # Used to shuffle the deck
import os      # Used for file path operations

from paths import ASSETS_DIR

# Suits and ranks used in a standard deck
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']

# Represents a single playing card
class Card:
    def __init__(self, suit, rank):
        self.suit = suit  # e.g., "hearts"
        self.rank = rank  # e.g., "queen"
        # Create the path to the image file for this card
        self.image_path = os.path.join(ASSETS_DIR, "cards", "png", f"{rank}_of_{suit}.png")

    def value(self):
        """Returns the blackjack value of the card"""
        if self.rank in ['jack', 'queen', 'king']:
            return 10
        elif self.rank == 'ace':
            return 11
        else:
            return int(self.rank)

# Represents a shoe of one or more 52-card decks, kept between rounds
class Deck:
    def __init__(self, num_decks=1, penetration=0.75):
        self.num_decks = num_decks
        self.penetration = penetration  # Fraction of the shoe dealt before a reshuffle
        self.shuffles = 0               # Counts fresh shoes, so card counters know to reset
        self.shuffle()

    def shuffle(self):
        """Refills the shoe with every card and shuffles it"""
        self.cards = [Card(suit, rank) for _ in range(self.num_decks)
                      for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)
        self.shuffles += 1

    def needs_shuffle(self):
        """True once the penetration point of the shoe has been dealt"""
        return len(self.cards) <= (1 - self.penetration) * 52 * self.num_decks

    def deal(self):
        """Deals (removes and returns) the top card, opening a fresh shoe if this one ran out"""
        if not self.cards:
            self.shuffle()
        return self.cards.pop()
