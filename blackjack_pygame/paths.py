# paths.py
# Absolute paths anchored to this package, so the game runs from any working directory.

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
