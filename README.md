# ai801

JupyterLab in Docker. Notebooks are saved to `./notebooks`.

## Run

```sh
docker compose up -d --build
```

Open http://localhost:8888 (no token; bound to localhost only).

Stop with `docker compose down`.

## Adding dependencies

Deps are managed with [uv](https://docs.astral.sh/uv/) inside the container — you don't need uv installed.

```sh
docker compose exec jupyter uv add pandas matplotlib
```

This updates `pyproject.toml` and `uv.lock` in the repo and installs the package immediately
(restart the kernel to pick it up). Commit both files. Teammates pulling the change run:

```sh
docker compose up -d --build
```
