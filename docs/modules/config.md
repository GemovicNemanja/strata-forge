# `strata_forge.config` — the configuration entry point

`strata_forge.config` holds the Pydantic Settings root that every other module reads instead of
touching `os.environ`. It sits one level above [`strata_forge.core`](core.md) in the dependency
graph and imports nothing else from the package, so configuration can be constructed before any
heavy module is loaded. Its external dependencies are `pydantic`, `pydantic-settings`, `pyyaml`,
and `python-dotenv`.

Four things live here:

- **`Settings`** — the root model, plus one sub-model per concern (Langfuse, Redis, Qdrant,
  Hugging Face, storage, logging, diagnostics). Each sub-model is a `BaseSettings` in its own
  right with its own env prefix.
- **`get_settings()` / `reset_settings()`** — a process-wide cached accessor and the test-only
  cache clear.
- **`load_env_file()`** — an explicit `.env` loader for entry points that want to control when
  dotenv values reach the process environment.
- **Overlay primitives** — `load_overlay`, `deep_merge`, `overlay_path_for_profile`,
  `DEFAULT_PROFILE_DIR`: standalone helpers for layering YAML profile files. They are building
  blocks, not an automatic mechanism; see
  [Profiles and YAML overlays](#profiles-and-yaml-overlays).

Every variable documented here is also listed in
[`.env.example`](https://github.com/GemovicNemanja/strata-forge/blob/main/.env.example), which is
the file to copy to `.env` when setting up a working tree.

Module rules:
[`src/strata_forge/config/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/config/CLAUDE.md).
Source:
[`src/strata_forge/config/`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/config/).

---

## Contents

- [Quickstart](#quickstart)
- [The Settings tree](#the-settings-tree)
- [Environment variables](#environment-variables)
- [Loading order and precedence](#loading-order-and-precedence)
- [Profiles and YAML overlays](#profiles-and-yaml-overlays)
- [The cached accessor](#the-cached-accessor)
- [Secrets](#secrets)
- [Reading configuration from your own module](#reading-configuration-from-your-own-module)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from strata_forge.config import get_settings

settings = get_settings()
print(settings.profile)           # "dev" unless FORGE_PROFILE is set
print(settings.logging.level)     # FORGE_LOG_LEVEL, default "INFO"
print(settings.langfuse.enabled)  # True only when both Langfuse keys are present
print(settings.qdrant.url)        # QDRANT_URL, default "http://localhost:6333"
```

Every function in this module is synchronous. Configuration resolution is pure local I/O — reading
process environment and at most one `.env` file — so it is a deliberate exception to the
async-first rule that governs the I/O-bearing modules.

---

## The Settings tree

`Settings` owns exactly one field of its own, `profile`, and aggregates eight sub-models through
`default_factory`. Each sub-model is instantiated when `Settings` is constructed, and each reads
its own environment variables under its own prefix.

| Field | Type | Env prefix | Purpose |
|---|---|---|---|
| `profile` | `str` | `FORGE_PROFILE` (or bare `PROFILE`) | Free-form deployment label. Default `"dev"`. |
| `providers` | `ProvidersConfig` | none | Empty by design — see the note below. |
| `langfuse` | `LangfuseConfig` | `LANGFUSE_` | Tracing host, keys, environment label. |
| `redis` | `RedisConfig` | `REDIS_` | Redis URL for the LLM response cache. |
| `qdrant` | `QdrantConfig` | `QDRANT_` | Vector-store URL and API key. |
| `storage` | `StorageConfig` | `FORGE_STORAGE_` | Storage-gateway defaults. |
| `huggingface` | `HuggingFaceConfig` | `HF_` | Hub token and endpoint. |
| `logging` | `LoggingConfig` | `FORGE_LOG_` | structlog level and renderer. |
| `diagnostic` | `DiagnosticConfig` | `FORGE_DIAGNOSTIC_` | NDJSON dump of completed LLM calls. |

`ProvidersConfig` is a plain `BaseModel` with no fields, and provider credentials are **not**
stored under it. They live in `strata_forge.llm.providers.config` as their own `BaseSettings`
models (`OpenAIConfig`, `AnthropicConfig`, `VertexConfig`, `BedrockConfig`, `AzureConfig`,
`OpenAICompatConfig`), because they are an LLM-module concern and the dependency arrow only runs
`llm` → `config`. `settings.providers` is therefore an empty object; reach for the LLM module's
configs instead.

`LangfuseConfig` exposes one computed property, `enabled`, which is `True` only when both
`public_key` and `secret_key` are set. The tracing module short-circuits to a no-op on that flag,
which is why an unconfigured install traces nothing rather than failing.

`HuggingFaceConfig` is importable from the submodule but is currently absent from the package's
re-export list, so use `from strata_forge.config.settings import HuggingFaceConfig` if you need
the type. The `settings.huggingface` field itself works exactly like the others.

`StorageConfig.default_backend` is read into settings and displayed by `strata-forge doctor`, but
`StorageGateway` derives the protocol from the URL scheme of each path it is given, so this value
does not change where data goes.

---

## Environment variables

**Read by `Settings`.** These are the variables the configuration root resolves. All of them are
optional; each has a working default or a documented "unset" behaviour.

| Variable | Field | Default |
|---|---|---|
| `FORGE_PROFILE` | `profile` | `dev` |
| `FORGE_LOG_LEVEL` | `logging.level` | `INFO` (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) |
| `FORGE_LOG_FORMAT` | `logging.format` | `auto` (`auto`/`pretty`/`json`) |
| `FORGE_DIAGNOSTIC_ENABLED` | `diagnostic.enabled` | `false` |
| `FORGE_DIAGNOSTIC_PATH` | `diagnostic.path` | `./forge-diagnostic.ndjson` |
| `FORGE_STORAGE_DEFAULT_BACKEND` | `storage.default_backend` | `local` (`local`/`s3`/`gcs`/`azure`/`hf`) |
| `LANGFUSE_HOST` | `langfuse.host` | `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | `langfuse.public_key` | unset |
| `LANGFUSE_SECRET_KEY` | `langfuse.secret_key` | unset |
| `LANGFUSE_TRACING_ENVIRONMENT` | `langfuse.tracing_environment` | unset (SDK default) |
| `REDIS_URL` | `redis.url` | `redis://localhost:6379/0` |
| `QDRANT_URL` | `qdrant.url` | `http://localhost:6333` |
| `QDRANT_API_KEY` | `qdrant.api_key` | unset |
| `HF_TOKEN` | `huggingface.token` | unset |
| `HF_ENDPOINT` | `huggingface.endpoint` | unset |

Values are coerced by Pydantic, so `FORGE_DIAGNOSTIC_ENABLED` accepts `1`, `true`, `yes`, and
their negations. Unknown variables are ignored (`extra="ignore"` on every model) — a typo in a
variable name is silent, so check the spelling against this table when a setting seems inert.

**Read by the LLM module's own settings models.** These belong to
`strata_forge.llm.providers.config` rather than `Settings`, and they read the process environment
directly (they do not declare an `env_file`, so a `.env` only reaches them if something else has
already loaded it into the environment).

| Variable | Config | Notes |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_ORG_ID` | `OpenAIConfig` | `enabled` when the key is set. |
| `ANTHROPIC_API_KEY` | `AnthropicConfig` | |
| `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT`, `GCP_REGION` | `VertexConfig` | Region defaults to `us-central1`. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | `BedrockConfig` | Region defaults to `us-east-1`; boto3's own credential chain still applies. |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | `AzureConfig` | |
| `FORGE_OPENAI_COMPAT_BASE_URL`, `FORGE_OPENAI_COMPAT_API_KEY` | `OpenAICompatConfig` | Self-hosted OpenAI-compatible servers. |

**Read directly by a handful of call sites.** A small number of variables are read with
`os.environ` outside the settings layer, either because the value is a per-invocation input rather
than configuration or because the code path already accepts explicit constructor arguments:

| Variable | Read by | Purpose |
|---|---|---|
| `FORGE_PROGRESS_PATH` | `strata_forge.training.progress`, `strata_forge.pipelines` | Fallback path for the JSONL progress stream. |
| `STRATA_RUN_CONFIG` | `strata_forge.pipelines.inference_runner` | The JSON run spec handed to the entrypoint. |
| `HF_WRITE_TOKEN` | `strata_forge.pipelines.inference_runner` | Token for uploading results to the Hub. |
| `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | the Langfuse prompt and dataset stores | Last-resort fallback when no explicit value was passed to the store. |

---

## Loading order and precedence

`Settings` resolves each field from the first source that supplies it, in this order, highest
precedence first:

1. **In-code overrides** — anything passed to `Settings(...)`.
2. **Process environment** — `os.environ` at the moment the model is instantiated.
3. **`.env`** — a file named `.env` in the current working directory.
4. **Field defaults** — the values declared on the model.

That resolution runs independently for every model in the tree. Because each sub-model is its own
`BaseSettings` with `env_file=".env"`, a sub-model picks up its variables even though the root
knows nothing about them.

The consequence worth internalizing is that overrides are **per field, not per sub-model**:

```python
from strata_forge.config import Settings

# With FORGE_LOG_FORMAT=json in the environment:
settings = Settings(logging={"level": "DEBUG"})
assert settings.logging.level == "DEBUG"   # from the override
assert settings.logging.format == "json"   # still resolved from the environment
```

Passing a partial dict for `logging` does not reset the fields it omits to their defaults — the
sub-model is constructed from that dict and fills the rest from its own sources. The same is true
when you pass a fully constructed `LoggingConfig(...)`: its unspecified fields were themselves
resolved from the environment at construction time.

Two further details:

- `profile` accepts either `FORGE_PROFILE` or a bare `PROFILE`, via `AliasChoices`. The prefixed
  form is the documented one.
- `.env` is resolved relative to the current working directory. Running a script from a different
  directory silently changes which file is read — or whether one is read at all.

Note also that importing `strata_forge.llm` pulls in LiteLLM, which calls `load_dotenv()` at
import time. That search is not limited to the working directory, so a `.env` elsewhere on the
path to the filesystem root can be loaded into `os.environ` as a side effect of an import. Values
already present in the environment win, so this only ever fills gaps — but it does mean a variable
can appear to come from nowhere. `strata_forge.config.load_env_file(path)` is the explicit
alternative: it loads exactly the file you name, returns whether one existed, and leaves
already-set variables alone unless you pass `override=True`.

---

## Profiles and YAML overlays

`overlays.py` provides the primitives for profile-based YAML configuration:

| Function | Signature | Behaviour |
|---|---|---|
| `overlay_path_for_profile` | `(profile: str, *, base_dir: PathLike \| None = None) -> Path` | `configs/<profile>.yaml` by default. |
| `load_overlay` | `(path: PathLike) -> dict[str, Any]` | Parses a YAML mapping. A missing or empty file yields `{}`. |
| `deep_merge` | `(base: dict, overlay: dict) -> dict` | Recursive merge, returning a new dict. |
| `DEFAULT_PROFILE_DIR` | `Path("configs")` | The conventional overlay directory. |

`load_overlay` treats a missing file as an empty overlay rather than an error, because overlays
are optional by design. Malformed YAML, or a top-level value that is not a mapping, raises
`strata_forge.core.errors.ConfigError` with `source` set to the offending path. `deep_merge`
merges nested dicts key by key and replaces lists and scalars wholesale — element-wise list
merging is almost never what a reader expects.

**These primitives are not wired into `Settings`.** Constructing `Settings()` reads defaults,
`.env`, and the process environment; it never looks for a `configs/` directory, and
`FORGE_PROFILE` does nothing beyond populating the `profile` string. If you want profile overlays,
compose them at your own entry point:

```python
from strata_forge.config import Settings, deep_merge, load_overlay, overlay_path_for_profile

profile = Settings().profile
overlay = load_overlay(overlay_path_for_profile(profile))  # {} when the file is absent
settings = Settings(**deep_merge({}, overlay))
```

Because overlay values arrive as in-code overrides, they take precedence over environment
variables for the keys they name. If you want the opposite — environment wins over the YAML file —
apply the overlay only to keys that are absent from the environment, or keep profile files
restricted to settings you never expect to override per shell session.

Keys in the overlay must match the `Settings` field structure (`{"logging": {"level": "WARNING"}}`,
not `{"FORGE_LOG_LEVEL": "WARNING"}`), and unknown keys are ignored rather than rejected.

---

## The cached accessor

`get_settings()` is wrapped in `functools.lru_cache(maxsize=1)`, so the first call constructs
`Settings` and every later call returns that same instance. Module-level code across the library
calls it freely on the assumption that it is cheap.

The cost of that cache is that mutating the environment after the first call has no effect:

```python
import os

from strata_forge.config import get_settings, reset_settings

os.environ["FORGE_LOG_LEVEL"] = "DEBUG"
reset_settings()                      # drop the cached instance
assert get_settings().logging.level == "DEBUG"
```

`reset_settings()` exists for tests and for entry points that mutate the environment during
startup. It is not a hot-reload mechanism: code that captured `get_settings()` into a local
variable keeps the old object.

Constructing `Settings()` directly bypasses the cache entirely, which is the right move when you
need a throwaway instance with overrides and do not want to disturb the process-wide one — that
is exactly what `strata-forge doctor` does.

---

## Secrets

Credential fields are typed `pydantic.SecretStr`, not `str`. They render as `**********` in
`repr`, in `str`, and in `model_dump_json()`, so a settings object that ends up in a log line or
an error message does not leak. Read the real value explicitly with `.get_secret_value()`:

```python
from strata_forge.config import get_settings

langfuse = get_settings().langfuse
print(langfuse.model_dump_json())  # a configured secret_key renders as "**********"
if langfuse.enabled:
    key = langfuse.secret_key.get_secret_value()  # explicit, greppable
```

The fields typed this way are `langfuse.public_key`, `langfuse.secret_key`, `qdrant.api_key`,
`huggingface.token`, and the provider API keys in `strata_forge.llm.providers.config`. When you
add a new credential, type it `SecretStr` too — the masking is the whole point, and a plain `str`
silently opts out of it.

---

## Reading configuration from your own module

The house convention is that `strata_forge.config` is the only place that turns environment
variables into typed values. In practice that means:

- Read configuration through `get_settings()`, not `os.environ`.
- Accept explicit constructor arguments for anything a caller might reasonably want to override
  per instance, and fall back to settings only when the argument is `None`. `HFHubClient` is the
  model to copy: explicit argument first, then `HuggingFaceConfig`, then the underlying SDK's own
  resolution.
- When a new setting appears, add a field to the appropriate sub-model with a sensible default,
  document it in
  [`.env.example`](https://github.com/GemovicNemanja/strata-forge/blob/main/.env.example), and add
  a test that covers its precedence.

Adding a whole new sub-model is the right move when a concern owns several related variables;
adding a stray `os.environ` read in a module that already has a settings path is not.

---

## Troubleshooting

**I changed an environment variable and `get_settings()` still returns the old value.**
The accessor is cached for the life of the process. Call `reset_settings()` after mutating the
environment, or construct `Settings()` directly for a fresh read.

**My `.env` is being ignored.**
`Settings` resolves `.env` relative to the current working directory, so running from a
subdirectory or from an IDE with a different working directory reads a different file, or none.
Confirm with `pathlib.Path(".env").exists()` from the same process, or load the file explicitly
with `load_env_file("/absolute/path/.env")` before constructing settings.

**A variable is set even though nothing in my project sets it.**
LiteLLM calls `load_dotenv()` when it is imported, and that search is not confined to the working
directory. A `.env` further up the filesystem can therefore populate the environment as a side
effect of `import strata_forge.llm`. Print `os.environ` around the import to confirm which values
appeared.

**I set `FORGE_PROFILE=prod` and nothing changed.**
`FORGE_PROFILE` sets the `profile` string and nothing else. No file is loaded from it. See
[Profiles and YAML overlays](#profiles-and-yaml-overlays) for the wiring you have to supply.

**My `configs/prod.yaml` has no effect.**
`Settings` never reads it. `load_overlay` is a primitive you call yourself; the repository has no
`configs/` directory and no code path that scans for one.

**`from strata_forge.config import HuggingFaceConfig` raises `ImportError`.**
The class is not in the package's re-export list. Import it from the submodule:
`from strata_forge.config.settings import HuggingFaceConfig`. The `settings.huggingface` field is
unaffected.

**`FORGE_LOG_LEVEL` doesn't change my log output.**
Nothing calls `configure_logging` for you. Wire the two together at your entry point:
`configure_logging(level=get_settings().logging.level)`. Note also that `logging.format` has three
values (`auto`, `pretty`, `json`) while `configure_logging` takes a boolean `json` argument — map
between them explicitly.

**Setting `FORGE_STORAGE_DEFAULT_BACKEND` doesn't change where files go.**
`StorageGateway` picks the backend from the URL scheme of each path (`s3://`, `gs://`, a bare
local path). The setting is surfaced by `strata-forge doctor` but does not steer I/O.

**My secret prints as `**********`.**
That is `SecretStr` doing its job. Call `.get_secret_value()` where you genuinely need the raw
value, and keep that call as close to the point of use as possible.

**A variable I invented is silently ignored.**
Every model uses `extra="ignore"`, so an unrecognized or misspelled variable produces no error.
Check the name against the [environment variables](#environment-variables) tables.
