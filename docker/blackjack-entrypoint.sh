#!/bin/sh
set -e

# Screen matches the game's default 800x600 window
Xvfb "$DISPLAY" -screen 0 800x600x24 -nolisten tcp &
until [ -e /tmp/.X11-unix/X99 ]; do sleep 0.2; done

x11vnc -display "$DISPLAY" -forever -shared -nopw -localhost -quiet &
websockify --web /usr/share/novnc 6080 localhost:5900 &

# Relaunch the game if a player quits or goes bankrupt, so the page never goes blank.
# With AGENT_POLICY set (module:name), an agent plays instead of a person.
cd /app/blackjack_pygame
while true; do
    if [ -n "$AGENT_POLICY" ]; then
        python autoplay.py --policy "$AGENT_POLICY" \
            --delay "${AGENT_DELAY:-1.0}" --decks "${AGENT_DECKS:-1}" || true
    else
        python main.py || true
    fi
    sleep 1
done
