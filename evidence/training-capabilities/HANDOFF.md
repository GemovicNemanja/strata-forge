# Inert fine-tuning declaration projection

## Source and ownership

- Checkout: `/Users/nemanjagemovic/Documents/Personal/Code/.strata-workspaces/platform-20261110/agents/training-capabilities`
- Branch: `feat/training-capabilities`
- Base: `86ed9d2bdb6a8be49f7e811cbbdbf010264a50fa` (selected runtime remains `b4e956992c47b1f7bf9f61ab4d64041419b12597`).
- Head: `d3d618983e08cc1ff3fc6bb3e1dc9d242c4bff15`
- Tree: `16e54f219add8500985005f5bdeffaeaec21083b`
- Author and committer: `Nemanja Gemovic <gemovic.nemanja@gmail.com>`.
- No remotes, no GitHub/auth/network Git operation. No source changes after this commit.

The six changed files are `training/capabilities.py`, its unit tests and exact JSON fixture,
plus the training reference, README and module rules. Shared exports, registries, existing
serialized specs, runners, workers, package metadata and dependency/image pins are unchanged.
`source-manifest.json` records exact changed-source hashes and retained-byte comparisons.

## Implemented boundary

The direct library leaf returns frozen strict catalog/request/decision models. Catalogs derive
methods and dataset roles from `METHODS`/`FORMATS`; ordered roles and knobs retain their order,
while named entries and format sets sort. Collections are immutable tuples and detached from
the mutable registries. Schema identity covers the actual generated catalog/request/decision
JSON Schemas; body identity hashes every canonical catalog field, including schema identity.

- Schema SHA256: `ccb5be4e552c6e3e542f282835566af31b3dfd157f1c361dd517a37ba8b50b8c`
- Catalog SHA256: `a3d41914c887df59ec0457ff013f104f5f2f47dd58548774204e81a0d25aae90`

The `inert_finetune_declaration` profile recognizes the existing pretrained causal-LM
SFT/DPO/ORPO/KTO method/shape cells, `none`/`lora`/`qlora` adapters and existing merge constraint.
`none` is pretrained full-weight tuning. GRPO remains declined with registry metadata. Other
heads, scratch initialization and explicit distribution are outside this producer profile;
this is not a statement about other trusted-Python/custom-task interfaces.

All decisions retain unknown installed/environment/hardware status. `distribution=unspecified`
does not imply single-GPU, DDP or FSDP support. Declared selectors never imply model/backend/
hardware suitability, authorization, trusted source/output custody, valid actual dataset rows,
parameter admission, allocation or execution. The declared `finetuning` extra is a primitive
packaging requirement, not a complete worker/publication deployment profile.

Malformed request selectors return fixed `invalid_request`; other refusals use fixed codes.
Raw Pydantic diagnostics and rejected values do not enter the decision. Copied models are
revalidated from their field dictionary so malformed nested/extra values cannot bypass checks
or trigger serializer warnings. The leaf does not mount or replace any consumer transport.

## Validation and preserved failures

All Python checks used the existing read-only Server virtual environment, Python3.14.5,
Pydantic2.13.4 and isolated `PYTHONPATH=<checkout>/src`. `env -i` retained actual HOME and a fixed
PATH, disabled dotenv/telemetry and selected the verified installed LiteLLM local cost map.
No installation or optional ML/model/worker execution occurred. `environment.json` pins versions
and the actual LiteLLM flag implementation source hash.

1. `initial-focused.log`: 185 passed, one cold-checker failure. The original checker incorrectly
   denied core LiteLLM/httpx as optional dependencies. Its actual traceback and script remain
   retained. This was not optional ML activation by the producer. Full pre-freeze source snapshots
   were not captured for these development attempts; do not treat them as immutable Git ancestors.
2. `cold-successor.log`: 186 passed after correcting only that classification and using the
   documented `LITELLM_LOCAL_MODEL_COST_MAP=True`. The cold subprocess denied actual ML extras
   and every socket/process audit event, and asserted its attempt counters were empty.
3. `final-focused.log`: 351 passed, one 15-second cold-import subprocess timeout in36.90s.
   This run covered the final runtime, complete188 new controls and164 existing method,
   dataset-format, PEFT and fine-tune-runner controls. The timed-out subprocess was reaped by
   `subprocess.run`'s timeout handling; its `-9` is retained as a failed test probe, not normal
   service/native teardown or a passing cold check.
4. `final-cold-only.log`: the exact cold-import assertions passed in32.31s with a45-second
   setup bound. No other assertions changed and no broad suite was repeated to conceal the
   earlier timeout. No optional ML import or socket/DNS/process attempt occurred inside that
   cold subprocess. Outer pytest launching this fixed subprocess is intentionally separate.
5. `final-type-controls.log`: two affected role/snapshot-immutability controls passed in0.88s
   after test-only typing corrections. Those corrections did not change runtime source.
6. `frozen-pyright.log`: final new source and tests,0 errors/0 warnings. Actual Node24.22.1.
   Earlier Node26 failed its dynamic loader; an incompatible two-option Pyright invocation and
   initial test typing findings remain in their original logs. No tool install or repair occurred.
7. `final-ruff-successor.log` and `frozen-format.log`: final affected-file lint/format pass.
   Earlier lint findings remain in the retained logs. `git diff --check` passed before commit.

The initial fixture-generation invocation omitted the LiteLLM local-cost flag. Its audit hook
denied `socket.getaddrinfo`; LiteLLM caught that denial and warned before using its bundled map.
That warning was observed in the tool transcript but was not originally captured to an evidence
file, so this paragraph is a retained observation rather than an invented original log.
`final-golden-generation.json` records the corrected generation's empty attempt counters.
The existing containing `training.__init__` core LLM import graph was deliberately unchanged;
the public module reference documents the offline flag requirement.

## Commands

The combined test invocation was:

```text
python -B -m pytest tests/unit/training/test_capabilities.py tests/unit/training/test_methods.py tests/unit/training/test_dataset_format.py tests/unit/training/test_peft.py tests/unit/pipelines/test_finetune_runner.py -q -W error
```

Only the failed cold test was rerun after changing its bound15→45; only the two edited test
controls were rerun after typing corrections. No claim is made that352 tests passed together
in one fresh process. The retained logs identify the actual cohorts.

Static analysis used installed Ruff and Node24 with installed Pyright's fixed entrypoint,
`--venvpath <retained-root>/strata-server`, and exactly the new source/test paths. The parent
interpreter and tools did not receive provider credentials or user-profile environment.

## Integration and remaining work

This is a partial FORGE-102 producer. FORGE-101 baseline reproduction and CONTRACT-104/106
remain open; the fixture is not cross-repository schema/wire sign-off. No App/server/CLI/HTTP/
MCP consumer, worker dispatch or model capability was activated. No version/release bump is
included because there is no publishing in this local implementation lease; release ownership
must review it before distribution. Consumer adapters must preserve these explicit limits.

Root can review/merge the original commit ancestry locally. Runtime rollback is omitting or
reverting this leaf; no existing spec digest, worker/image, user data or service state changed.
No listener, database, browser, GPU, native worker, cloud/provider job or production resource
was created; all recorded scoped command sessions completed. Original dirty checkouts,
paused dataset draft, current human preview and research/model data were untouched.
