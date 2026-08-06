# Agent rules — strata_forge.config

`strata_forge.config` is the single source of truth for runtime configuration. It depends only on `strata_forge.core`.

## Purpose

Hold the Pydantic Settings root, the YAML overlay loader, the `.env` integration, and a cached `get_settings()` accessor that every other module uses. Direct env-var reads (`os.environ[...]`) elsewhere in `forge` are a code smell — go through this module.

## Boundaries

- **Owns:** `settings.py` (the root `Settings` + sub-models), `overlays.py` (YAML overlay loader, deep merge), `env.py` (`.env` loading helper), `get_settings()` / `reset_settings()` accessors.
- **Imports from inside `forge`:** `strata_forge.core` only.
- **Allowed external deps:** `pydantic`, `pydantic-settings`, `pyyaml`, `python-dotenv`.
- **Does NOT:** construct provider clients, manage caches, read provider-specific env vars except through declared sub-models, make network calls.

## Public API

- `Settings` (Pydantic `BaseSettings`) — root config with sub-models per concern (`providers`, `langfuse`, `redis`, `qdrant`, `storage`, `logging`, `diagnostic`).
- `get_settings() -> Settings` — cached accessor; same instance for the lifetime of the process.
- `reset_settings() -> None` — clear the cache; tests only.
- `load_overlay(path: Path) -> dict[str, Any]` — load + parse a YAML overlay file.

## Internal patterns

- Loading order (lowest → highest precedence):
  1. Pydantic field defaults
  2. YAML overlay (`configs/<FORGE_PROFILE>.yaml` if `FORGE_PROFILE` set; otherwise `configs/dev.yaml` if it exists)
  3. `.env` file in repo root
  4. Process environment variables
  5. In-code overrides passed to `Settings(...)`
- Sub-models are nested Pydantic models, NOT separate `BaseSettings` (so they validate as part of the root).
- Env-var prefixes per sub-model are explicit (e.g. `LANGFUSE_*`, `REDIS_*`) — declared via `model_config = SettingsConfigDict(env_prefix="LANGFUSE_")` on each sub-model.
- Secrets are kept as `pydantic.SecretStr` so they don't accidentally `str()` into logs.
- `get_settings()` uses `@functools.lru_cache(maxsize=1)`.

## Test expectations

- Unit tests under `tests/unit/config/`.
- Coverage target: ≥ 85 % line.
- Tests cover loading-order precedence (each higher source overrides the one below).
- YAML overlay tests cover deep-merge (nested sub-models merge correctly, not replace wholesale).
- `reset_settings()` resets the cache between tests via a fixture in `tests/conftest.py`.
- Secret fields use `pytest.MonkeyPatch` for env-var injection; tests assert that secrets don't appear in `Settings.model_dump_json()` output without explicit reveal.

## Gotchas

- Pydantic `BaseSettings` reads env vars at instantiation. Tests that mutate the environment between assertions must reset the cache.
- Deep-merge for YAML overlays — a missing field in the overlay must NOT erase the default, only override what's explicitly set.
- When adding a new setting:
  1. Add the field to the appropriate sub-model with a sensible default.
  2. Document it in `.env.example` with a leading-comment description.
  3. Add a `forge doctor` check for it (if it's required for any feature to work).
  4. Update tests covering its env-var and overlay-source precedence.

## When to update this file

- Adding a new sub-model (e.g. `strata_forge.config.settings.RagConfig`).
- Changing the loading-order precedence.
- Renaming or removing env vars.
- Changing the `get_settings()` caching strategy.
- Introducing a settings hot-reload mechanism (would require an ADR).
