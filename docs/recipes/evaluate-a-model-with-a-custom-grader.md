# Evaluate a model with a custom grader

The graders that ship with `strata_forge.evals` cover the mechanical cases — exact match, regex,
JSON shape, embedding similarity, LLM judge. Real evaluations almost always need one more: a
rule that only makes sense for your task. This recipe builds an evaluation from a dataset, a
hand-written grader, and an LLM judge, then reads the report and turns it into a CI gate.

Modules touched: `strata_forge.datasets` (the data), `strata_forge.llm` (the calls),
`strata_forge.evals` (everything else).

**Prerequisites.** The base install plus one configured provider (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, …). The `bleu` and `rouge` metrics need the `[evals]` extra; nothing else on
this page does.

---

## 1. Describe the data

An evaluation set is a `Dataset` of frozen `DatasetItem`s. `input` is a free-form dict that your
renderer turns into messages; `expected_output` is what deterministic graders compare against;
`metadata` is where you put anything a custom grader needs.

```python
from strata_forge.datasets import Dataset, DatasetItem

dataset = Dataset(
    name="support-replies",
    description="Customer questions that must be answered with specific facts named.",
    items=(
        DatasetItem.from_input(
            {"question": "My order hasn't shipped. What now?"},
            expected_output="Check the tracking page; contact support if it is late by 48 hours.",
            metadata={"keywords": ["tracking", "48 hours"]},
        ),
        DatasetItem.from_input(
            {"question": "Can I change the delivery address after ordering?"},
            expected_output="Only before the parcel is dispatched, from the order detail page.",
            metadata={"keywords": ["dispatch", "order detail"]},
        ),
    ),
)
```

`DatasetItem.from_input` derives a stable content-hash `id`, so re-running the same eval against
the same items produces the same `Trial.item_id`. Persist the set through a `DatasetStore` when
you want versioning — see [`docs/modules/datasets.md`](../modules/datasets.md).

## 2. Turn each item into messages

The runner needs a `PromptRenderer`: a plain function from one `DatasetItem` to a message list.
Register it under a name and list that name in `Experiment.prompts`.

```python
from strata_forge.llm import SystemMessage, UserMessage

def render_support(item: DatasetItem) -> list:
    return [
        SystemMessage(content="Answer in two sentences. Name the exact policy details."),
        UserMessage(content=str(item.input["question"])),
    ]
```

If `Experiment.prompts` is empty the runner skips rendering entirely and sends
`json.dumps(item.input)` as a single user message — fine for smoke tests, rarely what you want
for a real eval.

## 3. Write the grader

`Grader` is a runtime-checkable Protocol, not a base class: anything with a `name` property and
an `async grade(*, item, response) -> GraderResult` is a grader. Subclassing buys you nothing.

```python
from strata_forge.datasets import DatasetItem
from strata_forge.evals import GraderResult
from strata_forge.llm import LLMResponse


class ContainsAll:
    """Pass only when every keyword listed on the item appears in the response."""

    def __init__(self, *, metadata_key: str = "keywords", name: str = "contains_all") -> None:
        self._metadata_key = metadata_key
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        required = [str(k).lower() for k in item.metadata.get(self._metadata_key, ())]
        text = response.text.lower()
        missing = sorted(k for k in required if k not in text)
        score = 0.0 if not required else (len(required) - len(missing)) / len(required)
        return GraderResult(
            grader_name=self.name,
            score=score,
            passed=bool(required) and not missing,
            explanation="" if not missing else f"missing keywords: {missing}",
            metadata={"missing": missing},
        )
```

Three conventions worth following, because the metrics and reports assume them:

- `score` is a float; binary graders use `0.0` / `1.0`. `pass_rate` counts `passed`;
  `mean_score` averages `score`, so they can disagree by design.
- `explanation` is what lands in the report's Failures section. Make it say what went wrong.
- `metadata` is free-form and survives into the `Outcome`, which is where per-grader diagnostics
  belong (the built-in judges put their own cost and latency there).

The Protocol is `runtime_checkable` and exported: `from strata_forge.evals import Grader`, then
`isinstance(ContainsAll(), Grader)` is `True`. An assertion on that in your own tests catches
signature drift early.

## 4. Add an LLM judge

Keyword coverage says nothing about whether the answer is any good. `LLMJudge` asks a model —
ideally a stronger one than the candidate — to score the response against a free-form criterion
and returns a `GraderResult` with the judge's reasoning as the explanation.

```python
from strata_forge.evals import LLMJudge
from strata_forge.llm import LLMClient

judge = LLMJudge(
    LLMClient("claude-opus-4-8", provider="anthropic"),
    criteria=(
        "Is the reply accurate, two sentences or fewer, and free of hedging? "
        "Score 0.0 to 1.0."
    ),
    pass_threshold=0.7,
)
```

`criteria` is keyword-only, as are `pass_threshold`, `name`, and the prompt overrides. The judge
is a normal `LLMClient` caller, so it participates in whatever cache, fallback chain, and budget
you configured on that client.

## 5. Declare the experiment and run it

An `Experiment` is data: frozen, hashable, serializable, carrying names rather than live
objects. The runner takes the experiment plus the concrete objects those names refer to.

```python
import asyncio

from strata_forge.evals import Experiment, SamplingParams, run_experiment

candidate = LLMClient("claude-haiku-4-5", provider="anthropic")

experiment = Experiment(
    name="support-replies-baseline",
    models=("claude-haiku-4-5",),
    prompts=("support",),
    dataset_name=dataset.name,
    grader_names=("contains_all", "llm_judge"),
    sampling=SamplingParams(temperature=0.0, max_tokens=200),
)

outcomes = asyncio.run(
    run_experiment(
        experiment,
        dataset=dataset,
        clients={"claude-haiku-4-5": candidate},
        graders=[ContainsAll(), judge],
        prompt_renderers={"support": render_support},
        concurrency=8,
    )
)
```

The runner validates up front and raises `ValueError` when a model has no client, a prompt has
no renderer, a `grader_names` entry has no matching grader, or two graders share a name. It
returns one `Outcome` per `(model, prompt, item)` in deterministic order, so two runs of the
same experiment line up row for row.

Two behaviours to know:

- **`concurrency`** bounds in-flight LLM calls with a semaphore. It is the only throttle; a
  large dataset against a rate-limited provider wants a small number here.
- **`continue_on_error=False`** (the default) propagates the first failure. Set it to `True`
  for long unattended runs: a failed call becomes a `Trial` with empty text,
  `provider="<failed>"`, and `metadata={"error": repr(exc)}`, and every grader on that row is
  recorded as failed. You keep the report; you lose the row.

## 6. Read the results

```python
from strata_forge.evals import (
    mean_score,
    pass_rate,
    pass_rate_by_grader,
    render_html,
    render_markdown,
)

print(render_markdown(outcomes, title="Support replies — baseline"))
print(pass_rate(outcomes, grader_name="contains_all"))
print(mean_score(outcomes, grader_name="llm_judge"))
print(pass_rate_by_grader(outcomes))

with open("report.html", "w", encoding="utf-8") as fh:
    fh.write(render_html(outcomes, title="Support replies — baseline"))
```

`render_markdown` emits a summary block, a per-trial table, and (unless you pass
`include_failures=False`) a Failures section built from grader explanations. `render_html`
returns a complete standalone document — no assets, no CDN.

To drill into a single row, `Outcome.by_grader(name)` returns that grader's result or `None`:

```python
for outcome in outcomes:
    verdict = outcome.by_grader("llm_judge")
    if verdict is not None and not verdict.passed:
        print(outcome.trial.item_id, verdict.score, verdict.explanation)
```

**One sharp edge.** `Trial.model` is `response.route.model` — the canonical registry name after
alias resolution, not the string you wrote in `Experiment.models`. Declare
`models=("opus",)` and the report groups under `claude-opus-4-7`. Use canonical names in the
experiment if you want the two to match.

## 7. Gate on it in CI

`evaluate_ci_gate` turns outcomes into a pass/fail verdict with per-grader thresholds and a cost
ceiling. By default it compares the **Wilson 95 % lower bound** rather than the point estimate,
which is what makes it usable on small evaluation sets: a 5-item run at 100 % does not clear an
85 % threshold, because five samples cannot distinguish 100 % from 60 %.

```python
from strata_forge.evals import CIGateThresholds, evaluate_ci_gate

result = evaluate_ci_gate(
    outcomes,
    thresholds=CIGateThresholds(
        min_pass_rate=0.80,
        max_cost_usd=0.50,
        per_grader_pass_rate={"llm_judge": 0.70},
    ),
)
if not result.passed:
    for issue in result.issues:
        print(issue)
    raise SystemExit(1)
```

Set `use_wilson_ci=False` to compare point estimates instead — appropriate once your set is
large enough that the confidence interval stops dominating. An empty `outcomes` tuple always
fails with `"no outcomes provided to evaluate"`, so a silently-empty run cannot pass the gate.

The repository ships a working gate at
[`scripts/run_eval_gate.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/scripts/run_eval_gate.py):
a deterministic dataset baked into the script, one model, one `ExactMatch` grader, committed
thresholds, non-zero exit on failure. It is the shape to copy.

---

## Runnable examples

- [`examples/21_eval_basic.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/21_eval_basic.py)
  — dataset, runner, `ExactMatch`, metrics.
- [`examples/22_eval_llm_judge.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/22_eval_llm_judge.py)
  — `LLMJudge` over open-ended summaries.
- [`examples/23_eval_ci_gate.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/23_eval_ci_gate.py)
  — thresholds and the Wilson bound.

## Where to go next

- [`docs/modules/evals.md`](../modules/evals.md) — every grader, metric, sweep, and report knob.
- [`docs/modules/datasets.md`](../modules/datasets.md) — versioning, diffing, the Hugging Face
  bridge, synthetic-data helpers.
- [ADR 0010](../architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md) — why the
  experiment is inert data and graders are a Protocol.
- [Control cost and reliability](control-cost-and-reliability.md) — put a `BudgetContext` around
  a large evaluation before it surprises you.
