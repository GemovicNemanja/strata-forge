# strata_forge.config

Configuration root for the whole project. Pydantic `BaseSettings` with sub-models per concern (providers, Langfuse, Redis, Qdrant, storage, logging, diagnostic), layered with YAML profile overlays (`FORGE_PROFILE`) and `.env` files. Loaded once at startup; `strata-forge doctor` validates consistency.

> Implementation pending. See `docs/roadmap.md` for current status.
