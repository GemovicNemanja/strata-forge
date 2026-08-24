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
- **Sampling happens off the caller's thread.** `GpuSampler.sample()` returns the last reading and
  refreshes in a daemon thread. Two facts force this: the callers are coroutines driving
  generations and a liveness heartbeat, and `subprocess.run(timeout=...)` does not actually bound
  `nvidia-smi` — its POSIX timeout path kills the child then calls `wait()` with no timeout, which
  never returns for the uninterruptible `D` state a wedged driver produces. Blocking there would
  hang the run inside a metrics call *and* defeat cancellation, since the stall sits in a blocking
  C call that `task.cancel()` can never interrupt. The probe abandons an unkillable child, leaking
  a zombie — strictly better than stranding a GPU.
- A failed refresh keeps the last good reading rather than blanking it: a slightly stale gauge is
  honest, whereas a gap makes a chart look like the GPU stopped.
- **Non-finite values are dropped at `coerce_float`.** `loss=nan` and `grad_norm=inf` are routine in
  fp16 training. Pydantic serializes them to JSON `null`, which `metrics: dict[str, float]` refuses
  to parse back — so `ProgressEvent` could emit a document it could not itself read, and an
  ordinary gradient explosion would cost the orchestrator the whole event.
- **Metric keys are capped** in length and count. `trainer_callback` is public API, and a
  consumer whose `compute_metrics` keys a score on a dataset-derived label would otherwise put that
  text straight onto a channel that is tailed, relayed and rendered.
- **The inference p50 is over a bounded window** (`deque(maxlen=...)`). `statistics.median` copies
  and sorts what it is given, so an unbounded history would make the per-chunk cost quadratic in the
  row count — and `progress_chunk` can be 1 with no `row_limit`.
- Multi-GPU boxes report **mean** utilization, **summed** memory, and **max** temperature. Max, not
  mean, because one card cooking is the fact worth surfacing and an average hides it behind its
  healthy neighbours.
- `tokens_per_s` is reported for training only when the trainer was configured to count tokens
  (`TrainingArguments.include_num_input_tokens_seen`). Absent that, it is not knowable here and
  reporting an estimate would be worse than reporting nothing.

## The console is a third channel, read by offset

Structured progress says how far a run has got; it does not say what the machine is printing. A
watcher wants both — a stalled install, a CUDA OOM, a tokenizer warning are all console text and
none of them are events.

`Backend.console` reads that stream INCREMENTALLY, by byte offset, returning a `ConsoleChunk`. It
exists because `logs(tail=N)` cannot be de-duplicated by a caller polling on an interval:
consecutive windows overlap by an unknown amount, and the obvious remedy — remember the last line
and resume after it — fails on the output that most needs watching, since a progress bar rewriting
itself emits the same line repeatedly. An orchestrator storing those overlapping windows would
accumulate the same text once per poll for the life of the run.

Consequences:

- **Offsets advance to the stream's current size, not to what was read.** When more accumulated
  than one chunk may carry, the newest bytes are returned and the rest is reported as
  `dropped_bytes`. Re-offering the skipped middle would leave a busy job's reader permanently
  behind, dropping the same bytes forever.
- **A gap is stated, never smoothed over.** A jump-cut presented as a continuous transcript is a
  worse artifact than one that says what is missing.
- **The SSH read is one round trip, with the arithmetic done remotely.** Measuring the files here
  and slicing them there would race a job that is still writing, and the frame lengths would stop
  matching the payloads.
- **Payloads are base64.** The SSH channel yields decoded text, so a raw slice that cut a
  multi-byte character would arrive with a length no longer equal to the byte count the offsets
  depend on. Console output is not guaranteed to be text at all — ANSI, NULs and CRs all appear.
- **Each slice is anchored to its START, not to the end of the file.** `tail -c N` counts back
  from the current EOF, and the job is still writing: a file that grew between the measurement and
  the slice would hand back a window shifted off the one the header describes, losing bytes at the
  front and re-delivering bytes at the back — the two failures byte offsets exist to eliminate.
  Doing both in one remote shell is necessary but not sufficient; `wc` and `tail` are still two
  processes.
- **The reply is treated as input, not as instruction.** It is composed by a shell on a machine
  its owner controls, so the byte cap expressed in that shell is a cap the remote is free to
  ignore, and the size header is a claim rather than a measurement. The read is therefore bounded
  by the CALLER (`connection.run()` buffers a whole reply before returning it, so `console` uses a
  streaming read with an explicit limit), the header is range-checked before it can reach a
  `ConsoleChunk` — a negative size raises a `ValidationError` out of a backend method nobody is
  catching, and an absurd one is persisted as a cursor and then wraps silently in the remote
  shell's 64-bit arithmetic, turning the next read into a replay — and a decoded payload must be
  exactly the length the header promised.
- **An offset never advances past bytes that did not arrive intact.** A box with no `base64`
  binary, a reply cut short by the command timeout, a corrupt frame: each costs one poll and is
  retried, rather than being read as "nothing was printed" and skipped forever.
- **A shrinking stream is a rotation, not a negative number.** A file truncated under the reader
  (a restart opening it with `>`, logrotate) leaves the offset past the end; the whole file is
  then unread, and the bytes lost in between are reported rather than silently clamped to zero.
- **A sliding in-memory buffer has to report how far it slid.** `LocalBackend` keeps a 1 MiB
  window, so its buffer indices are not stream offsets — reading `len(buffer)` as one pins the
  cursor at the cap and the reader goes silent for the rest of the run, including the final line.

## Alternatives considered

**A second telemetry file/channel.** Rejected: it doubles what the orchestrator tails and what
`Backend.read_file` must confine, for data whose natural home is the event that already describes
the same instant.

**Structured stage objects instead of a literal.** Rejected: the set is small, closed, and ordered.
A literal keeps it comparable across a JSON boundary with no schema to negotiate.

**A line-anchored cursor instead of byte offsets.** Rejected: it is ambiguous exactly when it
matters. Repeated identical lines are the normal output of anything with a progress bar.

**Widening `serving.py`'s `on_phase` hook to take a stage.** Rejected: `serving.py` reports prose
through a public one-argument hook and knows nothing about run stages, which is correct. The runner
binds the stage at the call site instead. For the same reason `ticking_phase` only passes `stage=`
when it has one — a bare `Callable[[str], None]` sink must keep working.
