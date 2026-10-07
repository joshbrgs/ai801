#!/bin/sh
set -e

# Screen matches the game's default 800x600 window
Xvfb "$DISPLAY" -screen 0 800x600x24 -nolisten tcp &
until [ -e /tmp/.X11-unix/X99 ]; do sleep 0.2; done

x11vnc -display "$DISPLAY" -forever -shared -nopw -localhost -quiet &
websockify --web /usr/share/novnc 6080 localhost:5900 &

# Relaunch the game if a player quits or goes bankrupt, so the page never goes blank
cd /app/blackjack_pygame
while true; do
    python main.py || true
    sleep 1
done
