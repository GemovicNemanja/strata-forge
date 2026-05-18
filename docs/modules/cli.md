# `forge.cli` — the `forge` command-line interface

`forge.cli` is the Typer-based CLI for the library. Every Forge
module with a useful operator workflow surfaces a command here:
`doctor`, `chat`, `prompts`, `datasets`, `eval`, `experiments`,
`compute`, `train`, `serve`. Every command is a thin wrapper
over its module — Forge never duplicates business logic in the
CLI layer.

The CLI installs as the `forge` entry point. The same code is
importable as `forge.cli.app` for testing or for users embedding
the CLI in their own Typer apps.

Module rules: [`src/forge/cli/CLAUDE.md`](../../src/forge/cli/CLAUDE.md).
Source: [`src/forge/cli/`](../../src/forge/cli/).

---

## Contents

- [Quickstart](#quickstart)
- [doctor](#doctor)
- [chat](#chat)
- [prompts](#prompts)
- [datasets](#datasets)
- [eval](#eval)
- [experiments](#experiments)
- [compute](#compute)
- [train](#train)
- [serve](#serve)
- [Local state](#local-state)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```bash
# Verify your environment.
forge doctor

# Chat with a model.
forge chat --model claude-opus-4-7 --message "Summarize Forge in one line."

# Run a quick eval.
forge eval run --model claude-opus-4-7 --dataset eval-pack --grader exact_match

# Submit a compute job and watch it.
forge compute submit ./task.yaml --backend local
forge compute status <job-id>
forge compute logs <job-id> --tail 20

# Kick off SFT.
forge train sft --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset sft-pack --output-dir ./checkpoints/sft \
    --adapter lora --adapter-rank 32

# Build a vLLM serving task.
forge serve vllm --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

## doctor

Diagnose environment, configuration, and service reachability.
Always exits 0 — it's a report, not a gate.

```bash
forge doctor
```

## chat

Single-shot or interactive REPL against an `LLMClient`.

```bash
forge chat --model claude-opus-4-7 --message "what's 2+2?"
forge chat --model claude-opus-4-7  # opens a REPL
forge chat --model gpt-5.5 --provider azure \
    --system "answer only in haiku" --message "the moon"
```

Options: `--model`, `--provider`, `--message`, `--system`,
`--temperature`, `--max-tokens`.

## prompts

Inspect the configured `PromptStore`. Backend selection follows
`forge.config.settings` (Langfuse if configured, otherwise
in-memory).

```bash
forge prompts list
forge prompts show summarize
forge prompts render summarize --vars '{"passage": "..."}'
```

## datasets

Inspect the configured `DatasetStore`.

```bash
forge datasets list
forge datasets show eval-pack
forge datasets head eval-pack -n 5
```

## eval

Run quick evaluations against the configured dataset store.
Writes a markdown report to `~/.forge/experiments/<name>.md`.

```bash
forge eval run \
    --model claude-opus-4-7 \
    --dataset eval-pack \
    --grader exact_match \
    --grader regex:'^[A-Z]\d{3}$'

forge eval list
forge eval show <name>
```

Supported graders for the CLI: `exact_match`,
`regex:<pattern>`. For richer experiments (LLM-judge, pairwise,
sweeps), use `forge.evals.runner.run_experiment` directly.

## experiments

Mirror group over saved eval reports for users built around the
"experiments" verb.

```bash
forge experiments list
forge experiments show <name>
forge experiments delete <name>
```

## compute

Submit and monitor compute jobs against any backend. The
`submit` step persists the resulting Job to
`~/.forge/jobs/<id>.json` so subsequent commands rehydrate the
right backend automatically.

```bash
forge compute submit ./task.yaml --backend local
forge compute submit ./task.yaml --backend ssh \
    --ssh-host gpu-host --ssh-user ml-team
forge compute submit ./task.yaml --backend skypilot

forge compute status <job-id>
forge compute logs <job-id> --tail 50
forge compute cancel <job-id>
forge compute cleanup <job-id>
forge compute list
```

The SSH backend needs `--ssh-host` and `--ssh-user` (port
defaults to 22). SkyPilot reads its credentials from the
standard environment.

## train

Run SFT or DPO over the configured dataset store. Heavy ML
deps (`torch`, `transformers`, `trl`, `peft`) ride behind the
`[finetuning]` extra and lazy-import only when the runner
actually trains.

```bash
forge train sft \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset sft-pack \
    --output-dir ./checkpoints/sft \
    --epochs 3 --batch-size 2 --grad-accum 8 \
    --adapter lora --adapter-rank 32

forge train dpo \
    --model ./checkpoints/sft \
    --dataset preference-pairs \
    --output-dir ./checkpoints/dpo \
    --beta 0.1
```

For ORPO / KTO / GRPO or fine-grained hyperparameter control,
drop into Python and use
`forge.training.PreferenceRunner` directly.

## serve

Build (and optionally submit) self-hosted inference serving
tasks. The default `--submit none` prints the YAML so callers
can route it to SkyPilot or SSH; `--submit local` runs it on
the in-process `LocalBackend` and stores the resulting job for
`forge compute` to monitor.

```bash
forge serve vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 --tp 1 --max-model-len 8192

forge serve tgi --model mistralai/Mistral-7B-v0.1 \
    --port 8080 --num-shard 2

forge serve sglang --model Qwen/Qwen2-7B-Instruct \
    --port 30000 --tp-size 4

# Run locally and let `forge compute` take over for monitoring.
forge serve vllm --model my-model --submit local
```

## Local state

Two directories under `~/.forge` accumulate state across
invocations:

| Path | Owner | Contents |
|---|---|---|
| `~/.forge/jobs/<id>.json` | `forge compute submit` | Job handle + backend selection + backend kwargs. |
| `~/.forge/experiments/<name>.md` | `forge eval run` | Markdown reports. |

Both layouts are intentionally simple — JSON for jobs, markdown
for reports — so you can shell-script over them or wipe them
without touching the rest of your environment.

## Troubleshooting

- **`forge: command not found`:** install the package
  editably (``uv sync``) so the entry point lands on `PATH`,
  or run via `uv run forge ...`.
- **`Langfuse is configured but the [langfuse] extra is not
  installed`:** install with `pip install
  'ai-forge[langfuse]'`. Alternatively, unset
  `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` to fall back to
  the in-memory store.
- **`The [finetuning] extra is required for SFTRunner`:**
  install with `pip install 'ai-forge[finetuning]'`. Expect a
  lengthy `torch` wheel build on CPU-only macOS.
- **`unknown job id`:** the saved state file under
  `~/.forge/jobs` was removed (probably by `forge compute
  cleanup` on a previous invocation). Re-submit.
- **`forge serve ... --submit local` doesn't expose the
  endpoint:** the `LocalBackend` spawns the server in the
  background; use `forge compute status` and check the
  `base_url` printed at submit time once the server is ready.
