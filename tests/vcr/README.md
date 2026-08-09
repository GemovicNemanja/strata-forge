# VCR cassette tests

Provider-touching integration coverage that replays previously-recorded HTTP
exchanges via [`pytest-recording`](https://pytest-recording.readthedocs.io/).
The goal: keep CI hermetic (no live keys) while still exercising the real
LiteLLM payloads each provider expects.

## Layout

```
tests/vcr/
├── __init__.py
├── conftest.py             # vcrpy record mode + scrubbing policy
├── test_providers.py       # the test matrix (provider × scenario)
└── cassettes/
    ├── _cross/             # cross-provider scenarios
    ├── anthropic/
    ├── azure/
    ├── bedrock/
    ├── openai/
    ├── openai_compat/
    └── vertex/
```

Cassettes live under `cassettes/<provider>/<scenario>.yaml`. Tests skip
gracefully when the matching cassette is missing, so the whole directory is
green on a fresh checkout. No cassettes are committed today — every provider
directory holds a `.gitkeep` and nothing else — which means this suite proves
nothing until somebody records against live keys.

`test_providers.py` pins one registry model per provider (`gpt-5.5` for
OpenAI and Azure, `claude-opus-4-7` for Anthropic and Bedrock,
`gemini-3.1-pro` for Vertex). Update that map when the registry's models
change, or the recordings drift from what the library actually dispatches.

## Scenarios

Every provider gets a cassette per scenario where the scenario applies:

- `basic_completion` — a small non-streaming call returns a non-empty response.
- `streaming` — chunks accumulate into a coherent reply.
- `structured_output` — Pydantic schema dispatch returns a parsed instance.
- `multimodal` — image input round-trips; recorded only for the vision-capable
  providers (OpenAI, Anthropic, Vertex, Bedrock).
- `single_tool_call` — model emits one tool call; caller invokes it manually.
- `multi_turn_tool_loop` — `run_tool_loop` survives two tool iterations.
- `rate_limit_retry` — the `@retry` policy catches a 429 and succeeds on retry.
- `content_filter` — `ProviderContentFilterError` short-circuits the chain.

Cross-provider scenarios (in `cassettes/_cross/`):

- `provider_fallthrough` — same logical model, one provider 429s, the next succeeds.
- `model_fallthrough` — first model exhausted, second model picks up.

## Recording new cassettes

You need live keys for every provider you want to cover. Set them in `.env`
or your shell, then:

```
make vcr-record
```

The target sets `RECORD=1` for you, which flips the record mode from `"none"`
to `"once"`. Every header named in `_HEADERS_TO_SCRUB` is replaced with
`[REDACTED]` *before* the cassette is written. If you record without
scrubbing — visible API keys, AWS sigv4 signatures, GCP bearer tokens —
**treat them as leaked and rotate immediately**.

After recording, inspect the produced YAML for any field you don't
recognize (model-specific headers, request IDs that might still be
sensitive) before committing.

The nightly workflow re-records the whole matrix when provider secrets are
configured: it deletes the existing `*.yaml` files, runs the suite with
`RECORD=1`, and opens a PR if anything changed.

## Replay

```
make vcr-replay
```

This runs `tests/vcr/` with `record_mode="none"`. Tests fail loudly if the
request shape drifts from what the cassette captured — that's the early
warning that an upstream library (LiteLLM, the provider SDK) changed its
wire format and the LLM module needs an adjustment.

## Adding a new scenario

1. Add the test function to `test_providers.py`.
2. Decide which providers are eligible (some scenarios don't apply to
   every provider — e.g. multimodal needs vision support).
3. Record cassettes for those providers with `make vcr-record`.
4. Commit the resulting YAMLs and update this README's scenario list.

The scrubbing policy is declared twice: `conftest.py` holds the `vcr_config`
fixture that `pytest-recording`'s marker consumes, and `test_providers.py`
builds its own `VCR` instance so each test can name its cassette path
explicitly. Both carry the same `_HEADERS_TO_SCRUB` tuple — extend both, or a
recording made through one path keeps the credential the other one strips.
