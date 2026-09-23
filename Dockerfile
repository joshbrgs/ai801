FROM python:3.12-slim-bookworm

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

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--notebook-dir=/app/notebooks", \
     "--IdentityProvider.token=", "--ServerApp.password="]
