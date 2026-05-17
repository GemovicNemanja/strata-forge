# `forge.evals` — typed experiments, graders, runner, reports, CI gate

`forge.evals` is the evaluation runner: it composes models, prompts,
datasets, and graders into a typed matrix, runs the LLM calls
concurrently, grades each response, and produces results downstream
tools (reports, CI gate) consume. See
[ADR 0010](../architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md)
for the experiment-as-data + pluggable-graders design.

Integration points:

- **Data shapes:** `Experiment`, `Trial`, `GraderResult`,
  `Outcome`, `SamplingParams` — frozen Pydantic v2 with tuple-typed
  collections.
- **Graders:** `Grader` Protocol; deterministic (`ExactMatch`,
  `Regex`, `JSONStructure`, `JSONField`); LLM-driven
  (`LLMJudge`, `PairwiseGrader`); embedding-based
  (`SemanticSimilarity`).
- **Runner:** `run_experiment` — async, parallelism-bounded.
- **Metrics:** `pass_rate`, `mean_score`, `pass_rate_by_model`,
  `pass_rate_by_grader`; `bleu` and `rouge` behind the
  `[evals]` extra.
- **Sweeps:** `sweep_sampling`, `sweep` — expand a base
  `Experiment` into variants.
- **Reports:** `render_markdown`, `render_html`.
- **CI gate:** `evaluate_ci_gate` with Wilson-CI lower bounds and a
  cost cap.
- **Trace replay:** `replay_trace` — fetch a Langfuse trace, rewrite
  optional system / user messages or sampling params, re-run through
  any `LLMClient`.

Module rules: [`src/forge/evals/CLAUDE.md`](../../src/forge/evals/CLAUDE.md).
Source: [`src/forge/evals/`](../../src/forge/evals/).

---

## Contents

- [Quickstart](#quickstart)
- [Experiments and outcomes](#experiments-and-outcomes)
- [Graders](#graders)
- [The runner](#the-runner)
- [Metrics](#metrics)
- [Parameter sweeps](#parameter-sweeps)
- [Reports](#reports)
- [The CI gate](#the-ci-gate)
- [Trace replay](#trace-replay)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
import asyncio
from forge.datasets import Dataset, DatasetItem
from forge.evals import (
    ExactMatch,
    Experiment,
    SamplingParams,
    render_markdown,
    run_experiment,
)
from forge.llm.client import LLMClient
from forge.llm.messages import UserMessage

async def main() -> None:
    dataset = Dataset(
        name="trivia",
        items=(
            DatasetItem.from_input({"q": "Capital of France?"}, expected_output="paris"),
            DatasetItem.from_input({"q": "2 + 2?"}, expected_output="4"),
        ),
    )
    client = LLMClient(model="claude-opus-4-7", provider="anthropic")
    experiment = Experiment(
        name="trivia-baseline",
        models=("claude-opus-4-7",),
        dataset_name=dataset.name,
        grader_names=("exact_match",),
        sampling=SamplingParams(temperature=0.0, max_tokens=50),
    )
    outcomes = await run_experiment(
        experiment,
        dataset=dataset,
        clients={"claude-opus-4-7": client},
        graders=[ExactMatch(case_sensitive=False)],
    )
    print(render_markdown(outcomes))

asyncio.run(main())
```

End-to-end demos:

- [`examples/20_eval_basic.py`](../../examples/20_eval_basic.py) —
  full experiment with `ExactMatch` + Markdown report.
- [`examples/21_eval_llm_judge.py`](../../examples/21_eval_llm_judge.py)
  — `LLMJudge` grading open-ended summaries.
- [`examples/22_eval_ci_gate.py`](../../examples/22_eval_ci_gate.py)
  — CI-gate workflow on synthetic outcomes (no provider keys).

---

## Experiments and outcomes

```python
class Experiment(BaseModel):       # frozen, extra="forbid"
    name: str
    models: tuple[str, ...]
    prompts: tuple[str, ...] = ()
    dataset_name: str
    dataset_version: str | None = None
    grader_names: tuple[str, ...]
    sampling: SamplingParams = SamplingParams()
    description: str = ""
    metadata: dict[str, Any] = {}
```

`Experiment` carries *names* (model name, prompt name, grader name,
dataset name) — never live objects. The runner resolves names
against registries at run time so a serialized `Experiment` is
portable across processes.

```python
class Trial(BaseModel):            # one (model, prompt, item) + response
    experiment_name: str; model: str; provider: str
    prompt_name: str | None; item_id: str
    response_text: str; usage: Usage; cost_usd: float
    cache_hit: bool; latency_ms: float

class GraderResult(BaseModel):     # one grader's verdict on one trial
    grader_name: str; score: float; passed: bool
    explanation: str = ""

class Outcome(BaseModel):          # trial + all grader verdicts for it
    trial: Trial
    grader_results: tuple[GraderResult, ...]
```

The runner returns `tuple[Outcome, ...]` — that's the shape every
downstream consumer (metrics, reports, CI gate) iterates over.

---

## Graders

The `Grader` Protocol is async and `runtime_checkable`:

```python
@runtime_checkable
class Grader(Protocol):
    @property
    def name(self) -> str: ...
    async def grade(
        self, *, item: DatasetItem, response: LLMResponse,
    ) -> GraderResult: ...
```

### Deterministic graders

| Grader | What it checks |
|---|---|
| `ExactMatch(case_sensitive=, strip=)` | `response.text == str(item.expected_output)` |
| `Regex(pattern, expected_match=True)` | regex matches `response.text` |
| `JSONStructure(schema=)` | `response.text` parses as `schema` (Pydantic v2) |
| `JSONField(path=, expected=)` | dotted JSON path equals `expected` (or item label) |

### LLM-driven graders

`LLMJudge(client, criteria, pass_threshold=0.7)` asks an LLM via
`complete_structured` to score the response on a `JudgeVerdict`
schema (`score: float in [0,1]`, `reasoning: str`). Pass a custom
`verdict_schema` for richer rubrics; override `system_prompt` /
`user_template` to control wording.

`PairwiseGrader(client, criteria)` puts the candidate side-by-side
with the item's `expected_output` and asks the judge to pick
`A` / `B` / `tie`. Position bias is mitigated by
`secrets.randbelow`-based slot randomization (toggle with
`randomize_positions=False` for deterministic tests).

### Embedding-based grader

`SemanticSimilarity(embed, pass_threshold=0.8)` runs cosine
similarity between embeddings of the response and the reference.
Pass a user-supplied async `embed(text) -> Sequence[float]` —
typically a thin wrapper over your provider's embedding endpoint.
The raw cosine is stored in `result.metadata`; the `score` field is
clamped to `[0, 1]` for metric composability.

`cosine_similarity(a, b) -> float` is exposed if you want to compute
similarities yourself.

---

## The runner

```python
outcomes = await run_experiment(
    experiment,
    dataset=dataset,                                 # resolved Dataset
    clients={model_name: LLMClient(...), ...},
    graders=[ExactMatch(), Regex(r"..."), ...],
    prompt_renderers={"my-prompt": lambda item: [...]},
    concurrency=10,
    continue_on_error=False,
)
```

The runner generates the cross-product across
`models × prompts × items`, dispatches LLM calls under an
`asyncio.Semaphore(concurrency)`, and applies every grader to every
trial. When `experiment.prompts` is empty, the runner uses a default
renderer that emits one `UserMessage` with the item's `input` dict
JSON-encoded. Validation runs before any LLM call: every named model
must have a client, every prompt must have a renderer, every grader
name must have an instance.

`continue_on_error=True` yields a synthetic `Outcome` with
`provider="<failed>"` and a failed grader result instead of
propagating the exception — handy for runs where one item shouldn't
sink the whole batch.

---

## Metrics

```python
from forge.evals import (
    pass_rate,                  # fraction passing one grader
    mean_score,                 # average score across one grader
    pass_rate_by_model,         # {model -> rate}
    pass_rate_by_grader,        # {grader -> rate}
    bleu, rouge,                # text-similarity metrics — [evals] extra
)
```

Pure-Python metrics are always available. `bleu` and `rouge` live
behind the `[evals]` extra (`nltk` + `rouge-score`); they look at
`trial.metadata["expected_text"]` for the reference text, so
downstream code that wants text-similarity metrics should populate
that field when constructing the runner's input.

---

## Parameter sweeps

```python
from forge.evals import sweep_sampling, sweep

# Sweep sampling axes — Cartesian product over temperature × top_p × max_tokens.
variants = sweep_sampling(
    base_experiment,
    temperatures=[0.0, 0.5, 1.0],
    top_p=[0.9, 1.0],
)

# Generic sweep — any Experiment field can be an axis.
model_variants = sweep(base_experiment, models=[("m1",), ("m2",), ("m3",)])
```

Each variant gets a deterministic name suffix
(`base.name/axis=value/...`) so reports trace results back to
configurations.

---

## Reports

```python
from forge.evals import render_markdown, render_html

print(render_markdown(outcomes, title="Today's eval"))

with open("report.html", "w") as f:
    f.write(render_html(outcomes, title="Today's eval"))
```

Both renderers emit a summary, per-grader and per-model pass-rate
tables, and a per-trial section with verdicts. The Markdown variant
uses ✓/✗ glyphs and is suitable for PR comments / CI logs; the HTML
variant is a standalone page with inline CSS for browser viewing.
Both escape user content — XSS isn't a risk in the HTML report.

---

## The CI gate

```python
from forge.evals import CIGateThresholds, evaluate_ci_gate

result = evaluate_ci_gate(
    outcomes,
    thresholds=CIGateThresholds(
        min_pass_rate=0.85,
        max_cost_usd=2.50,
        per_grader_pass_rate={"strict_match": 0.95},
        use_wilson_ci=True,         # default: conservative for small samples
    ),
)
if not result.passed:
    for issue in result.issues:
        print(f"::error::{issue}")
    sys.exit(1)
```

`evaluate_ci_gate` consults the Wilson 95 % lower bound by default
(`use_wilson_ci=True`) so small samples don't pass simply by
happening to score high. Threshold misses surface as
human-readable `issues` strings plus machine-readable rate /
Wilson-bound dicts on the `CIGateResult`.

`wilson_lower_bound(passed, total, z=1.96) -> float` is exposed for
custom rules.

---

## Trace replay

```python
from forge.evals import ReplayOverrides, replay_trace

result = await replay_trace(
    trace_id="abc-123",
    client=LLMClient(model="claude-opus-4-7", provider="anthropic"),
    overrides=ReplayOverrides(
        system_message="rewritten system prompt",
        temperature=0.0,
    ),
)
print(result.original_response_text, "->", result.new_response.text)
```

Pulls the input message list from a Langfuse trace, optionally
rewrites the system message or the last user message, optionally
swaps in different sampling parameters, and re-runs through any
`LLMClient`. The returned `ReplayResult` bundles the original
trace id, the original response text, and the fresh `LLMResponse`
so callers can diff them without round-tripping Langfuse twice.

Requires the `[langfuse]` extra. The Langfuse SDK is sync; the
call is wrapped in `asyncio.to_thread`.

---

## Lazy-import contract

`import forge.evals` works without any optional extras installed.
SDK imports happen lazily inside the functions that need them:

- `bleu` / `rouge` → `[evals]` extra (`nltk`, `rouge-score`).
- `replay_trace` (when no explicit `langfuse_client` is passed) →
  `[langfuse]` extra.

`ImportError` surfaces only when a caller actually invokes the
relevant function without the extra installed; the message names
the extra explicitly so the fix is obvious.

---

## Troubleshooting

- **"Experiment models without clients"**: the runner found a
  model name in `experiment.models` that isn't a key in the
  `clients` mapping. Add an `LLMClient` for each model — or remove
  the model from the experiment.
- **"Grader names must be unique"**: two grader instances share a
  `.name`. Override one with `name="..."` in its constructor so
  reports can address them separately.
- **CI gate flips small samples to failures**: that's the Wilson
  lower bound working as intended — with only a handful of trials,
  even a 100 % pass rate has a low confidence floor. Either
  increase the sample size (more dataset items, multiple runs) or
  set `use_wilson_ci=False` for point-estimate comparisons.
- **`LLMJudge` is too lenient / too strict**: tune `pass_threshold`
  on the verdict score, or sharpen the `criteria` text. The judge's
  reasoning is in the explanation field — read a few results to
  calibrate.
- **`replay_trace` raises `ValueError: Unsupported trace.input shape`**:
  the Langfuse trace's input doesn't match the supported shapes
  (`list[{"role": ..., "content": ...}]` or a single string). The
  helper `extract_messages_from_trace` is exposed if you need to
  pre-process traces with a different shape before replay.
