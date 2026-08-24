# 0017 — Run telemetry rides the progress channel

**Status:** Accepted

## Context

`ProgressEvent` (ADR 0016's companion — the JSONL an orchestrator tails via `Backend.read_file`)
reports how far a run has got: `step`, `total_steps`, `epoch`, `loss`, `learning_rate`, plus a
free-form `metrics` bag and a prose `message`.

Two things a run watcher needs are missing from that, and both were discovered by trying to build
the watcher:

1. **Which milestone the run is in.** `kind="phase"` events carry prose — `"Loading the dataset"`,
   `"Starting the model server"`, `"Generating responses"` — and the wording changes whenever
   someone improves a caption. A consumer that wants to render an ordered stepper has no choice but
   to pattern-match English, which makes every caption edit a breaking change to a UI in another
   repository.

2. **Whether the machine is working.** A stalled dataloader, a model that quietly spilled to CPU, a
   card thermal-throttling at 88C, and a healthy run are indistinguishable from the step count. The
   numbers that tell them apart — utilization, memory, temperature — were nowhere in the system.

Alongside those, three inference counters the UI wants (rows/s, tokens/s, latency p50) turned out to
be derivable from data the runner already had in hand and was discarding.

## Decision

**Telemetry extends the existing progress channel rather than adding a second one.**

- `ProgressEvent` gains **`stage`**, a `RunStage` literal from a fixed ordered set:
  `provision → install_engine → load_model → run → push`, exported as `RUN_STAGES`. It is
  independent of `kind`: `kind` says what sort of record this is, `stage` says where in the run it
  sits. `message` keeps carrying the prose, and stays free to change.
- A runner reports only the last three stages. `provision` and `install_engine` describe the VM
  *before the runner's process exists*, so the orchestrator that submitted the job owns them and
  infers them from the job's own lifecycle. Dataset loading reports as `load_model` — not literally
  the model, but the same "getting ready" milestone from the watcher's side, and a stage the user
  never sees its own label for is a stage that should not exist.
- A `phase` event may now carry `stage` and `metrics`. Neither is a step count, and both stay true
  during exactly the long uncountable stretches where nothing else does.
- **GPU counters come from `nvidia-smi`, not NVML**, via `strata_forge.training.hardware`. They ride
  in `metrics` under `gpu_*` keys.
- **Inference throughput is computed from `LLMResponse`**, which already carries `latency_ms` and
  `usage` per row. Rates are cumulative and the latency is a real median.
- **Training throughput is derived from the wall clock between logged steps** (`_Pace`), and yields
  nothing for a key the trainer already reported.

## Consequences

- A consumer renders an ordered stepper from `RUN_STAGES` and `stage`, and never parses `message`.
  Caption wording stays editable without breaking a downstream UI.
- `metrics` remains the extension point: every new counter is a key, not a schema change, so an
  orchestrator on an older version ignores what it does not know rather than failing to parse.
- **No new dependency.** `nvidia-smi` ships with the driver on any box that has a GPU. This is
  deliberate and load-bearing: forge's dependencies install *fresh on the user's VM* at the start of
  every run, so each new pin is another package that can publish a breaking release between a green
  CI run and someone's four-hour fine-tune. NVML would be tidier in-process and is not worth that.
- **Telemetry never fails a run.** Every sampler failure path — no binary, no driver, a timeout, a
  changed CSV shape, a CPU-only box — returns `{}`. A missing gauge is cosmetic; an exception raised
  out of a metrics call mid-run is not.
- GPU sampling costs one subprocess per `GpuSampler.min_interval_s` (5s default). The sampler caches
  and repeats the last reading in between, because a slightly stale gauge is honest whereas a gap
  makes a chart look like the GPU stopped.
- Multi-GPU boxes report **mean** utilization, **summed** memory, and **max** temperature. Max, not
  mean, because one card cooking is the fact worth surfacing and an average hides it behind its
  healthy neighbours.
- `tokens_per_s` is reported for training only when the trainer was configured to count tokens
  (`TrainingArguments.include_num_input_tokens_seen`). Absent that, it is not knowable here and
  reporting an estimate would be worse than reporting nothing.

## Alternatives considered

**A second telemetry file/channel.** Rejected: it doubles what the orchestrator tails and what
`Backend.read_file` must confine, for data whose natural home is the event that already describes
the same instant.

**Structured stage objects instead of a literal.** Rejected: the set is small, closed, and ordered.
A literal keeps it comparable across a JSON boundary with no schema to negotiate.

**Widening `serving.py`'s `on_phase` hook to take a stage.** Rejected: `serving.py` reports prose
through a public one-argument hook and knows nothing about run stages, which is correct. The runner
binds the stage at the call site instead. For the same reason `ticking_phase` only passes `stage=`
when it has one — a bare `Callable[[str], None]` sink must keep working.
