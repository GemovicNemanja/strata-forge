# strata_forge.config

The single configuration entry point. `Settings` is a Pydantic Settings root with a sub-model per
concern (`providers`, `langfuse`, `redis`, `qdrant`, `storage`, `huggingface`, `logging`,
`diagnostic`); `get_settings()` returns the cached instance and `reset_settings()` clears it for
tests. Values resolve from field defaults, then `.env`, then process environment, then any in-code
override, and secrets are held as `SecretStr` so they do not leak into logs.

`load_overlay`, `deep_merge` and `overlay_path_for_profile` are standalone YAML helpers for callers
that want profile overlays; `Settings` itself does not apply them. No optional extra is required.

Reference:
[docs/modules/config.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/config.md).
