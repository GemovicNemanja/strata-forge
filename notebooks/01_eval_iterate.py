"""Eval-and-iterate template.

Builds an :class:`InMemoryDatasetStore` with a handful of items,
runs an evaluation across one or more models with
:func:`forge.evals.runner.run_experiment`, and displays the
outcomes table for quick inspection.

Set the relevant provider env vars before launching with::

    uv run marimo edit notebooks/01_eval_iterate.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("# Eval and iterate\n\nTweak the dataset, pick models, run, inspect.")
    return


@app.cell
def _():
    # Tiny inline dataset — replace with a `DatasetStore.get` call once
    # you have a real one configured.
    from forge.datasets.schema import Dataset, DatasetItem

    items = (
        DatasetItem(
            id="cap-fr",
            input={"q": "What is the capital of France?"},
            expected_output="Paris",
        ),
        DatasetItem(
            id="cap-jp",
            input={"q": "What is the capital of Japan?"},
            expected_output="Tokyo",
        ),
        DatasetItem(
            id="cap-au",
            input={"q": "What is the capital of Australia?"},
            expected_output="Canberra",
        ),
    )
    dataset = Dataset(name="tiny-capitals", items=items, description="three capitals")
    return (dataset,)


@app.cell
def _(mo):
    models_input = mo.ui.multiselect(
        options=[
            "claude-haiku-4-5",
            "gpt-5.5-instant",
            "gemini-3.1-flash",
        ],
        value=["claude-haiku-4-5"],
        label="models",
    )
    return (models_input,)


@app.cell
def _(models_input):
    models_input
    return


@app.cell
async def _(dataset, models_input):
    from forge.evals.experiment import Experiment, SamplingParams
    from forge.evals.graders.exact import ExactMatch
    from forge.evals.runner import run_experiment
    from forge.llm.client import LLMClient

    chosen = list(models_input.value)
    clients = {m: LLMClient(model=m) for m in chosen}
    graders = (ExactMatch(),)
    exp = Experiment(
        name="tiny-capitals-quick",
        models=tuple(chosen),
        dataset_name=dataset.name,
        grader_names=(graders[0].name,),
        sampling=SamplingParams(temperature=0.0, max_tokens=50),
    )
    outcomes = await run_experiment(
        exp,
        dataset=dataset,
        clients=clients,
        graders=graders,
        concurrency=3,
        continue_on_error=True,
    )
    return (outcomes,)


@app.cell
def _(mo, outcomes):
    rows = []
    for outcome in outcomes:
        verdict = "✅" if all(r.passed for r in outcome.grader_results) else "❌"
        rows.append(
            {
                "item": outcome.trial.item_id,
                "model": outcome.trial.model,
                "response": outcome.trial.response_text[:80],
                "passed": verdict,
                "cost": f"${outcome.trial.cost_usd:.5f}",
            }
        )
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
