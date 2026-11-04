# Scoring foundation review evidence

Recorded 2026-11-03. Task `INFER-001` is a private draft under review, not completed `f:SCORE`
and not an enabled inference feature. The implementation is preserved in
[drafts/scoring-foundation.patch](drafts/scoring-foundation.patch); no model/data/secret file
is included. [README.md](README.md) contains implementation and downstream acceptance briefs.

## Location and provenance

- Base: audited Forge `origin/dev` `28a5bcda9a3e013509ca8763f7699d55192a0db7`.
- Draft checkout: `/private/tmp/strata-scoring-foundation-20261103`.
- Local branch: `codex/scoring-foundation`.
- Source local object clone: `/private/tmp/strata-forge-audit-20261103`; no remote fetch,
  original checkout/source edit or Claude branch change was needed for this draft.
- Portable patch: [drafts/scoring-foundation.patch](drafts/scoring-foundation.patch).
- Untouched comparison worktree: `/private/tmp/strata-scoring-baseline-20261103`, detached
  at the same base, created inside the isolated clone's worktree metadata.
- Changes: private scoring module 477 lines; offline fake tests 540 lines; compute reference
  documentation 59 added lines. Three files, 1,076 insertions. No commits/push/merge/release.

The original Forge checkout still has an older ref and another agent's documentation branch.
Tomorrow's coordinator must refresh source refs, preserve that ownership and review whether
the patch applies to the actual new integration base. A locally stored `origin/dev` name is
not itself evidence of freshness.

## Specification read and conservative resolution

Read `~/.claude/plans/logic-verifier/impl/spec-forge.md` SCORE and relevant phase-P3 text.
The implemented subset provides exact classifier and label-token forward-pass scoring,
numeric `id2label` order and aliases, single-token preflight, full-vocabulary outside mass,
length batching, explicit overflow accounting and safe loaders.

The signed requirement says `truncation="only_first"`, preserving the full hypothesis, but
does not define a hypothesis that consumes the entire token budget by itself. The draft
refuses the row when hypothesis+pair specials exceed the limit, or equal it while a nonempty
premise would need complete removal. This avoids tokenizer-dependent failure or truncation
of the hypothesis. The row stays in output with `hypothesis_too_long`, no probabilities and
honest overflow/failed counts. A truncated first sequence can otherwise succeed and has a
separate `truncated` count. Decoder prompts never truncate.

The foundation interprets `label_alias` as source config class name→output name; generic
`LABEL_n` requires explicit aliases. Decoder verbalizer strings are literal, single-token
and distinct. Later phase-P5 proposes leading-space normalization for decoder training;
one shared reviewed rule and final token IDs must be recorded before training integration.

## Checks actually run

Tool source: `/Users/nemanjagemovic/Documents/Personal/Code/strata-forge/.venv`, read-only.
Python: 3.14.5. Tests execute the isolated checkout via `PYTHONPATH=src`; neither torch nor
transformers is installed in that reused environment. All new forward passes use fakes.

| Check | Result |
| --- | --- |
| `PYTHONPATH=src …/.venv/bin/python -m pytest tests/unit/pipelines/test_scoring.py -q` | **39 passed**, latest run 0.99s |
| `…/.venv/bin/ruff check src/strata_forge/pipelines/_scoring.py tests/unit/pipelines/test_scoring.py` | Passed |
| Ruff format on the two Python files | Applied; subsequent check clean |
| Pyright with explicit reused interpreter, source module only | **0 errors, 0 warnings**; notes local `.venv` absent |
| Pyright with source and test files | **0 errors, 8 warnings**; seven `pytest.approx` incomplete-stub warnings and one fake tensor index unknown-type warning |
| Earlier 25-test checkpoint with coverage | 25 passed; reported combined statement/branch coverage 85%; coverage was not rerun after the additional 14 refusal/output tests |
| Broader `tests/unit/pipelines` at the 25-test checkpoint | 317 passed, **75 failed**, one skipped |
| Same broader suite on untouched base | 292 passed, **the same 75 failed**, one skipped |

The latest 39-test suite covers classification class ordering, mandatory generic aliases,
invalid maps/collisions, safe explicit token/revision loading for both engines, safetensors
and remote-code flags, first-only truncation, refused hypotheses, length-sort order recovery,
step counters, exact conditional probabilities and outside mass, last-real-token indexing in
mixed-length padding, decoder overflow refusal, bad verbalizers before model loading,
finite/shape/vocabulary checks, absent padding tokens, empty input, invalid engine/options,
lazy imports and dependency installation hints.

### Why the broader suite is not green

The reused venv's installed distribution metadata says **strata-forge 0.0.1**, while the
audited source and `SPEC_VERSION` are **0.3.0**. Existing engine handshake tests correctly
refuse that mismatch; dependent runner-spec tests fail before exercising their intended path.
An existing clean-install metadata test also cannot find the base's bitsandbytes declaration
in the older installed distribution. The skipped test requires `datasets.packaged_modules`,
which is absent. Independent untouched-base execution reproduced the same failure set.

No handshake assertion, source version or installed metadata was changed to make this pass.
This establishes no observed new broader regression in that reused environment; it does not
substitute for a complete green suite under a correctly installed isolated current package.
`INFER-002` must create that environment and rerun the affected suites and clean-install checks.

## Not run and not claimed

- No downloaded or real model weights, tokenizer downloads, CUDA/CPU reference inference,
  real HF reads/writes, live provider calls or paid operations.
- No fresh `[serving,storage,hf]`, `[finetuning,storage,hf]` or `[all]` Linux dependency resolve.
- No worker VM, locally run server, app, remote MCP, endpoint or production acceptance.
- No 1k ModernBERT/Qwen reference comparison within 1e-4, Parquet artifact or Viewer smoke.
- No performance/latency/VRAM/quantization/model-calibration result.
- No trained verifier, model novelty or commercial usefulness claim.
- No integrated `Hyperparams.engine`, pair-template field, dependency extra, public SDK export,
  runner dispatch, artifact commit ref or lifecycle verdict change.

## Review and integration gaps

1. Review the actual patch against the signed semantics, not only this evidence summary.
2. Rebase/transplant in an assigned isolated checkout from refreshed `origin/dev`; retain
   the base SHA and code review evidence. The portable patch is for review, not auto-apply.
3. Resolve pair-template character limits without the existing renderer silently cutting a
   hypothesis. Choose supported model token limits and identical train/eval verbalizer IDs.
4. Add actual runner dispatch, input IDs/refs, Parquet/error persistence, all-failed verdict,
   start/end/progress artifacts and redaction through shared runner plumbing.
5. Declare bounded direct serving dependencies and prove clean installs; later create the
   signed `[scoring]` extra with eval integration. Do not advertise a phantom extra today.
6. Complete engine/alias/pair-template agent wire-design sign-off before any merge that makes
   the existing hyperparams passthrough able to launch these engines.
7. Verify cancellation through supervised process termination: a cancelled to-thread await
   does not terminate an in-flight model forward. Evaluate memory use of full decoder logits
   and equivalent last-position optimizations before larger batches/contexts.
8. Do approved real-model/reference and cross-repo acceptance. G1 waits for the Strata scoring/metrics/durable-evaluation path; there is no E0/E0+ execution alternative.

The public source patch carries generic primitives and synthetic test strings only. Internal
plan paths and research decisions stay here under `plans/`, outside the package's source-only
sdist allow-list. Re-check wheel/sdist contents when publishing; do not add this plan directory
to package build includes.
