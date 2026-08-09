# Docker

Two unrelated things live here: `compose.yaml`, the local service stack you run *next to* a
checkout while developing, and `Dockerfile`, an image that packages the library itself.

## The local dev stack (`compose.yaml`)

```
make stack-up      # docker compose -f docker/compose.yaml up -d
make stack-logs    # tail all three services
make stack-down    # stop them; named volumes survive
```

Three services on small official images (Postgres and Redis pinned to a major, Qdrant tracking
`latest`):

| Service | Image | Ports | What in the library uses it |
|---|---|---|---|
| `qdrant` | `qdrant/qdrant:latest` | 6333 (HTTP + dashboard), 6334 (gRPC) | `strata_forge.rag.QdrantVectorStore`, and through it `strata_forge.agents.EpisodicMemory` — both need the `[rag]` extra |
| `redis` | `redis:7-alpine` | 6379 | `strata_forge.llm.RedisCache`, the cross-process response cache — needs the `[redis]` extra |
| `postgres` | `postgres:16-alpine` | 5432 | Nothing in `strata_forge`. It is here as the database a self-hosted Langfuse deployment expects |

Credentials are deliberately trivial: Postgres comes up as `forge` / `forge` / database `forge`,
and neither Qdrant nor Redis is authenticated. That is fine for a laptop and unfit for anything
reachable from a network.

The defaults in `strata_forge.config` already point at this stack — `QDRANT_URL` defaults to
`http://localhost:6333` and `REDIS_URL` to `redis://localhost:6379/0` — so no configuration is
needed once the containers are up. `strata-forge doctor` prints a reachability probe for both,
which is the fastest way to confirm the stack is actually listening.

State lives in the named volumes `qdrant_data`, `redis_data` and `postgres_data`. `make stack-down`
leaves them in place; `docker compose -f docker/compose.yaml down -v` throws them away.

### Langfuse is not in this file

Tracing, the Langfuse-backed prompt store, and the Langfuse-backed dataset store all talk to a
Langfuse instance, and this compose file does not provide one. Point `LANGFUSE_HOST`,
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` at Langfuse Cloud or at a deployment you run
yourself (that is what the `postgres` service above is sized for). Without those keys the tracing
layer stays a silent no-op and the Langfuse-backed stores fail on first use with an explicit
"Langfuse is not configured" error rather than falling back to something local.

### Which tests need the stack

Unit tests never do — Qdrant and Redis are faked in-process, so `make test` passes with nothing
running. Neither do the cross-module suites under `tests/integration/*/test_cross_module.py`, which
are unmarked and run under `make test-all`.

The `integration` marker is what signals "needs a live service", and today exactly two tests carry
it, both against Langfuse:

```
make stack-up                         # if you are pointing Langfuse at a local deployment
export LANGFUSE_HOST=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
make integration                      # pytest -m integration
```

Both skip with a clear reason when the keys are absent, so the target is safe to run either way.

## The image (`Dockerfile`)

A two-stage build that produces a runnable container with the library installed: the builder stage
installs the locked dependency set into `/app/.venv` with `uv sync --frozen --no-dev`, and the
runtime stage copies that virtualenv onto a bare `python:3.14-slim` layer with a non-root `forge`
user. Dependencies install before the source is copied, so editing `src/` reuses the dependency
layer.

It expects the repository root as the build context, because it copies `pyproject.toml`, `uv.lock`
and `src/`:

```
docker build -f docker/Dockerfile -t strata-forge .
```

The image is a convenience for running the CLI somewhere hermetic; the library itself is
distributed on PyPI as `strata-forge`, and nothing here is needed to consume it. No CI workflow
builds this image, so it is not covered by the checks that guard the rest of the repo.

> **Known issue.** The `ENTRYPOINT` and `HEALTHCHECK` invoke a `forge` binary. The package installs
> exactly one console script, `strata-forge`, so a container built from this file fails to start
> with `exec: "forge": executable file not found`. Both lines need to name `strata-forge` before
> the image is usable.
