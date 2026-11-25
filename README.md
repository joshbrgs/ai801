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

## Blackjack environment

`src/blackjack_rl/` is the shared game environment: Gymnasium's `Blackjack-v1` extended with
double, split and a finite shoe, plus baseline policies. Notebooks import
it (`import blackjack_rl`), and `notebooks/blackjack_split_double.ipynb` walks through it.

```sh
uv run pytest            # tests, including headless rounds on the pygame table
```

### Watching an agent play

Any policy `(obs, info) -> action` can play the pygame table, one move per `--delay` seconds:

```sh
cd blackjack_pygame
uv run python autoplay.py --policy blackjack_rl.policies:basic_strategy --decks 6
```

or in Docker, at the same noVNC URL as above:

```sh
AGENT_POLICY=blackjack_rl.policies:basic_strategy AGENT_DECKS=6 docker compose up -d blackjack
```

Leave `AGENT_POLICY` unset to play the game yourself.

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
