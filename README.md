# ai801

JupyterLab in Docker. Notebooks are saved to `./notebooks`.

## Run

```sh
docker compose up -d --build
```

- JupyterLab: http://localhost:8888 (no token; bound to localhost only)
- Blackjack (pygame): http://localhost:6080/vnc.html?autoconnect=true&resize=scale

The game runs in its own container on a virtual display and is streamed to the browser via noVNC,
so nothing needs to be installed locally beyond Docker. It relaunches automatically if you quit
or go bankrupt. Restart it after code changes with `docker compose restart blackjack`.

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

## Credits

`blackjack_pygame/` is vendored from [NathanGr33n/blackjack_pygame](https://github.com/NathanGr33n/blackjack_pygame)
(commit `cc65c0d`) by NathanGr33n, used under the MIT License (full text in
`blackjack_pygame/README.md`), with local modifications.
