# Scripts

Operational helpers that are run on a schedule or on demand, not imported as library code.
Everything here is invokable with `uv run python scripts/<name>.py`.

## `run_eval_gate.py`

The canonical eval-regression gate. It builds a 20-item capital-cities dataset baked into the
script itself (deterministic across hosts, no `DatasetStore` involved), runs it through
`strata_forge.evals.run_experiment` against one model with the `ExactMatch` grader, and scores the
outcomes with `strata_forge.evals.evaluate_ci_gate`. The committed thresholds are the module-level
`THRESHOLDS` constant: a Wilson-95 lower bound on the pass rate of 0.80 and a total spend ceiling
of $0.10.

```
make eval-gate                              # default model: claude-haiku-4-5
make eval-gate MODEL=gpt-5.5                # any logical name in the model registry
uv run python scripts/run_eval_gate.py --model claude-opus-4-7
```

It prints trial counts, cost, per-grader pass rates and Wilson bounds, then exits 0 when the gate
passes and 1 when it does not — which is what makes it usable as a CI step. The nightly workflow
(`.github/workflows/nightly.yml`) runs it against `claude-haiku-4-5` and skips itself when no
`ANTHROPIC_API_KEY` is configured.

The run issues live LLM calls and therefore costs real money. Twenty short completions against a
small model is fractions of a cent; pointing `--model` at a flagship multiplies that.

## Adding a script

Open with a module docstring that says what the script does and how to invoke it; `run_eval_gate.py`
passes its own docstring to `argparse`, so that text doubles as `--help` output. If the script is
something people run routinely, add a `make` target for it so it shows up in `make help`.
