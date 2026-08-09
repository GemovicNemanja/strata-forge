# `strata_forge.cli` — the `strata-forge` command-line interface

`strata_forge.cli` is the Typer-based CLI for the library. Every module
with a useful operator workflow surfaces a command here:
`doctor`, `chat`, `prompts`, `datasets`, `eval`, `experiments`,
`compute`, `train`, `serve`. Every command is a thin wrapper
over its module — no business logic is duplicated in the
CLI layer.

The CLI installs as the `strata-forge` entry point. The same code is
importable as `strata_forge.cli.app` for testing or for users embedding
the CLI in their own Typer apps.

Module rules: [`src/strata_forge/cli/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/cli/CLAUDE.md).
Source: [`src/strata_forge/cli/`](https://github.com/GemovicNemanja/strata-forge/tree/main/src/strata_forge/cli/).

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
strata-forge doctor

# Chat with a model.
strata-forge chat --model claude-opus-4-7 --message "Summarize entropy in one line."

# Run a quick eval.
strata-forge eval run --model claude-opus-4-7 --dataset eval-pack --grader exact_match

# Submit a compute job and watch it.
strata-forge compute submit ./task.yaml --backend local
strata-forge compute status <job-id>
strata-forge compute logs <job-id> --tail 20

# Kick off SFT.
strata-forge train sft --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset sft-pack --output-dir ./checkpoints/sft \
    --adapter lora --adapter-rank 32

# Build a vLLM serving task.
strata-forge serve vllm --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

## doctor

Print the resolved settings, the versions of the packages the library
tracks, and TCP reachability probes for Langfuse, Redis, and Qdrant.
Always exits 0 — it's a report, not a gate, and it validates nothing.

```bash
strata-forge doctor
```

It does not check provider credentials, model-registry consistency, or
whether the LiteLLM Langfuse callback is installed. For the last one, call
`strata_forge.tracing.is_litellm_callback_installed()` yourself.

## chat

Single-shot or interactive REPL against an `LLMClient`.

```bash
strata-forge chat --model claude-opus-4-7 --message "what's 2+2?"
strata-forge chat --model claude-opus-4-7  # opens a REPL
strata-forge chat --model gpt-5.5 --provider azure \
    --system "answer only in haiku" --message "the moon"
```

Options: `--model`, `--provider`, `--message`, `--system`,
`--temperature`, `--max-tokens`.

## prompts

Inspect the configured `PromptStore`. Backend selection follows
`strata_forge.config.settings` (Langfuse if configured, otherwise
in-memory).

```bash
strata-forge prompts list
strata-forge prompts show summarize
strata-forge prompts render summarize --vars '{"passage": "..."}'
```

## datasets

Inspect the configured `DatasetStore`.

```bash
strata-forge datasets list
strata-forge datasets show eval-pack
strata-forge datasets head eval-pack -n 5
```

## eval

Run quick evaluations against the configured dataset store.
Writes a markdown report to `~/.forge/experiments/<name>.md`.

```bash
strata-forge eval run \
    --model claude-opus-4-7 \
    --dataset eval-pack \
    --grader exact_match \
    --grader regex:'^[A-Z]\d{3}$'

strata-forge eval list
strata-forge eval show <name>
```

Supported graders for the CLI: `exact_match`,
`regex:<pattern>`. For richer experiments (LLM-judge, pairwise,
sweeps), use `strata_forge.evals.runner.run_experiment` directly.

## experiments

Mirror group over saved eval reports for users built around the
"experiments" verb.

```bash
strata-forge experiments list
strata-forge experiments show <name>
strata-forge experiments delete <name>
```

## compute

Submit and monitor compute jobs against any backend. The
`submit` step persists the resulting Job to
`~/.forge/jobs/<id>.json` so subsequent commands rehydrate the
right backend automatically.

```bash
strata-forge compute submit ./task.yaml --backend local
strata-forge compute submit ./task.yaml --backend ssh \
    --ssh-host gpu-host --ssh-user ml-team
strata-forge compute submit ./task.yaml --backend skypilot

strata-forge compute status <job-id>
strata-forge compute logs <job-id> --tail 50
strata-forge compute cancel <job-id>
strata-forge compute cleanup <job-id>
strata-forge compute list
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
strata-forge train sft \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset sft-pack \
    --output-dir ./checkpoints/sft \
    --epochs 3 --batch-size 2 --grad-accum 8 \
    --adapter lora --adapter-rank 32

strata-forge train dpo \
    --model ./checkpoints/sft \
    --dataset preference-pairs \
    --output-dir ./checkpoints/dpo \
    --beta 0.1
```

Both subcommands also accept `--dataset-version` (pin a version),
`--dataset-file` (load a serialized `Dataset` JSON instead of reading the
store — how an orchestrator ships data to a remote worker), `--lr`,
`--precision`, `--max-seq-length`, `--seed`, and `--progress-jsonl`
(the NDJSON progress stream described in
[`strata_forge.training`](training.md#progress-streaming)).

`--dataset` is required even when `--dataset-file` supplies the data; pass
any placeholder name in that case, since the value is then ignored.

For ORPO / KTO / GRPO or fine-grained hyperparameter control,
drop into Python and use
`strata_forge.training.PreferenceRunner` directly.

## serve

Build (and optionally submit) self-hosted inference serving
tasks. The default `--submit none` prints the YAML so callers
can route it to SkyPilot or SSH; `--submit local` runs it on
the in-process `LocalBackend` and stores the resulting job for
`strata-forge compute` to monitor.

```bash
strata-forge serve vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 --tp 1 --max-model-len 8192

strata-forge serve tgi --model mistralai/Mistral-7B-v0.1 \
    --port 8080 --num-shard 2

strata-forge serve sglang --model Qwen/Qwen2-7B-Instruct \
    --port 30000 --tp-size 4

# Run locally and let `strata-forge compute` take over for monitoring.
strata-forge serve vllm --model my-model --submit local
```

> **Redirecting `--submit none` to a file needs a wide terminal.** Output
> goes through a Rich console, which hard-wraps at 80 columns when stdout
> is not a TTY — long enough `run:` lines break mid-command and the result
> is not valid YAML. Set `COLUMNS` before redirecting:
>
> ```bash
> COLUMNS=1000 strata-forge serve vllm --model my-model --port 8000 > task.yaml
> ```
>
> The same caveat applies to any command whose output you pipe:
> `compute logs`, `datasets head`, `prompts render`, `eval show`, and
> `experiments show`.

## Local state

Two directories under `~/.forge` accumulate state across
invocations:

| Path | Owner | Contents |
|---|---|---|
| `~/.forge/jobs/<id>.json` | `strata-forge compute submit` | Job handle + backend selection + backend kwargs. |
| `~/.forge/experiments/<name>.md` | `strata-forge eval run` | Markdown reports. |

Both layouts are intentionally simple — JSON for jobs, markdown
for reports — so you can shell-script over them or wipe them
without touching the rest of your environment.

## Troubleshooting

- **`strata-forge: command not found`:** install the package
  (`pip install strata-forge`, or `uv sync` in a checkout) so the entry
  point lands on `PATH`, or run via `uv run strata-forge ...`. The
  distribution installs exactly one console script, `strata-forge`;
  there is no bare `forge` binary.
- **A raw traceback out of `prompts` or `datasets` mentioning
  `langfuse`:** Langfuse credentials are set but the `[langfuse]` extra
  isn't installed, and the failure surfaces from the store rather than as
  a friendly CLI message. Install with
  `pip install 'strata-forge[langfuse]'`, or unset
  `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` to fall back to
  the in-memory store.
- **`The [finetuning] extra is required for SFTRunner`:**
  install with `pip install 'strata-forge[finetuning]'`. Expect a
  lengthy `torch` wheel build on CPU-only macOS.
- **`unknown job id`:** the saved state file under
  `~/.forge/jobs` was removed (probably by `strata-forge compute
  cleanup` on a previous invocation). Re-submit.
- **`strata-forge serve ... --submit local` doesn't expose the
  endpoint:** the `LocalBackend` spawns the server in the
  background; use `strata-forge compute status` and check the
  `base_url` printed at submit time once the server is ready.
- **Redirected output isn't parseable:** Rich wraps at 80 columns off a
  TTY. Prefix the command with `COLUMNS=1000` — see [serve](#serve).

---

## See also

- [`strata_forge.config`](config.md) — the settings every command reads
  to pick a store backend and reach a service.
- [`strata_forge.evals`](evals.md) — the full experiment surface behind
  `eval` and `experiments`, including the graders the CLI doesn't expose.
- [`strata_forge.compute`](compute.md) — the backends `compute` and
  `serve` drive.
- [`strata_forge.training`](training.md) — every knob `train` wraps, plus
  the ones it doesn't.
- [`strata_forge.prompts`](prompts.md) and
  [`strata_forge.datasets`](datasets.md) — the stores `prompts` and
  `datasets` inspect.
