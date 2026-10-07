FROM python:3.12-slim-bookworm AS base

# uv binary only — nobody on the team needs it installed locally
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Keep the venv outside /app so the project bind mount doesn't hide it
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install deps first so this layer is cached until pyproject/lock change
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project


FROM base AS jupyter

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--notebook-dir=/app/notebooks", \
     "--IdentityProvider.token=", "--ServerApp.password="]


# Pygame needs a display: run it on a virtual X server and stream it to the browser via noVNC
FROM base AS blackjack

RUN apt-get update \
    && apt-get install -y --no-install-recommends xvfb x11vnc novnc python3-websockify fontconfig fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99 \
    SDL_AUDIODRIVER=dummy

COPY docker/blackjack-entrypoint.sh /usr/local/bin/blackjack-entrypoint
RUN chmod +x /usr/local/bin/blackjack-entrypoint

EXPOSE 6080

CMD ["blackjack-entrypoint"]
