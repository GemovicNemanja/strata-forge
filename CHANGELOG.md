# Changelog

All notable changes to `strata-forge` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Changes accumulate under **Unreleased** as they merge
to `dev`; cutting a release renames that heading to the version and its date (see `CLAUDE.md`
§9b for what forces a release and which bump it takes).

## [Unreleased]

### Added

- Trusted plugins (`strata_forge.plugins`, ADR 0021). A plugin repository declares its entry
  points in `strata-plugin.json` (`strata.plugin/v1`): each one's type (`runner` or
  `step_scorer`), read-only dataset and model inputs, one output and closed typed configuration.
  An operator allow-list pins exact builds (distribution, version, a git source at a full commit
  with the digest of its built files, or a Hugging Face wheel at a full commit with its SHA-256),
  plus per-plugin limits; `preflight` checks a request against
  the declared interface as data, with no plugin code run. `pipelines/plugin_runner.py` runs an
  allowed entry point on the target: it refuses a build not in the orchestrator's
  `STRATA_ALLOWED_PLUGINS`, installs the pinned build into a dedicated per-build directory
  (`plugins/environment.py`: keyed by the pin, the engine and the interpreter, no credential in
  the install, and no plugin code outside the sandbox at any step: hash-locked dependency wheels
  and an hf wheel checked against its SHA-256 are only unpacked, a git source is built inside the
  sandbox from its locked build requirements, and the directory is read from its files, never
  run, with an executable `.pth` refused; every pin carries its installed files' digest, checked
  after the build and on every reuse), downloads the inputs itself (no pickle-format weights or
  other files that run code when loaded), refuses a token delivered in the environment, and runs
  the plugin on the engine's interpreter with `-I` in the strongest sandbox the machine has
  (`plugins/sandbox.py`: bubblewrap as `nobody`, else bubblewrap; no unsandboxed mode; the engine
  bound read-only, bind parents made 0755; recorded with the ptrace scope), with no credential,
  an offline and empty Hub cache, time and resource limits, an end to any process that stays
  stopped, and progress relayed through a validated pipe, while it samples the GPU itself. Its
  whole process group is killed and confirmed gone however it ends; its output is copied without
  following links (links, hard links and special files refused) and only that copy is checked,
  recorded with every file's hash in a `kind: "plugin"` manifest and pushed to a private
  repository in one commit, reported as `<repo>@<commit>`. Plugin-reported metrics are keyed
  `plugin.<name>`. The host re-verifies the build, appends the plugin's directory after the
  engine's on its path (never as a site directory) and imports only modules that are the pinned
  distribution's own files, a module already in `sys.modules` included.
- The step-scoring (PRM) contract: a `step_scorer` reads `steps` (`item_id`, `step_index`,
  `context` as a string or `{pid, text}` segments passed unchanged, `step`, and `step_type` when
  present), declares a `score_kind` (`probability` refused outside [0, 1] and never clipped;
  `logit` and `unconstrained` passed through; NaN and infinity always refused), returns one
  number or a declared `Unscored` reason per step, and forge writes `scores.parquet` with one row
  per input step and the declared pass-through columns, re-checked against the input before the
  push. The manifest records the counts by reason. `min`, `mean`, `last` and `product`
  reductions are opt-in only and go to `aggregates.diagnostic.parquet`, never the scores.
- Plugin runner hardening: a reused plugin directory must also match the digest of the whole
  directory recorded when it was built (dependencies and their data files included, not only the
  plugin's RECORD); the scores are parsed by `plugins/checker.py` inside the sandbox, never in
  the token-holding runner, which reads back only a bounded, validated record; a git source's
  build is stopped after 10 minutes without output as well as at its 30-minute limit; in a `uid`
  sandbox `plugins/supervisor.py` ends the sandbox when the runner's lifeline pipe closes, so a
  plugin dies with the runner rather than with `sudo`; the tools venv is created with `-I`.

### Changed

- The shared receipt (`build_receipt`) records `dataset.commit` and `template_sha256` where the
  manifest records them (scoring, evaluation, grading). A grading run writes its own receipt
  (`kind` `grade`, with the commit its push made), replacing the judge's, ends with `repo@commit`
  and removes the judge's failed-row record. `commit_oid` accepts only an exact 40-character oid,
  so a commit URL records no commit rather than one parsed out of it.

## [0.7.0] - 2026-11-18

### Added

- `seq_cls` monitors its eval split during training, for monitoring only: every
  `ceil(steps_per_epoch / 4)` steps and on the final checkpoint, macro-F1, accuracy and per-class
  F1 over three-way rows and accuracy and F1 for `entailed` over two-way rows (`monitor/...` keys,
  argmax, uncalibrated) beside `eval_loss`, as `eval` events labelled
  `monitoring (ID dev, final checkpoint is the model)` and in the manifest. The final checkpoint is
  the model: best-model selection, early stopping and `eval_steps` are refused, and the eval split
  must share no row with the training split (overlapping or undecidable split specs, such as
  `train[:80%]` with `train[70%:]`, are refused).
- `strata_forge.metrics.judge_agreement.judge_agreement` (`judge-gold-agreement-v1`): an LLM
  judge's scores against gold scores of the same outputs, per criterion (contingency table, exact
  agreement, Cohen's kappa, and for an integer scale the mean absolute error and the
  linearly and quadratically weighted kappas).
- `strata_forge.pipelines.grade_runner`: the LLM grader. A pinned judge model (the `vllm_chat`
  engine with the rubric's guided-JSON answer schema) grades each system's pinned outputs on a
  pinned gold split; per-case statuses and scores go to `items.parquet`, the judge's capped
  rationales to `rationales.parquet`, and cluster-bootstrap means, paired deltas, transitions and
  judge agreement to `metrics.json` (`strata.eval.grades/v1`). The inference runner's chat path
  takes prepared rows for it.
- `strata_forge.pipelines._grading`: the LLM grader's rubric (`Rubric`, `Criterion`: pass/fail
  or a bounded integer scale), the judge's guided-JSON answer schema (`rubric_schema`, rationale
  first), its digest and `parse_grade`, which reads a reply as unscored unless it is complete.
  A criterion id that would collide with the run's headline keys is refused: `cases`, `graded`,
  `coverage`, a `delta_` or `agreement_` prefix, a `_lower`, `_upper` or `_unresolved` suffix, or
  `__`.
- `strata_forge.metrics.calibration_fit.reliability_bins` returns the bins the top-label ECE
  weighs (`ReliabilityBin`: `lo`, `hi`, `n`, `mean_confidence`, `accuracy`), and the evaluation
  report carries them as `reliability_bins` in every raw, cross-fit and applied probability read.
- `strata_forge.metrics.cluster_bootstrap` adds `cluster_bootstrap` and
  `paired_cluster_bootstrap` (`cluster-percentile-bootstrap-v1`): seeded percentile intervals
  for accuracy, macro-F1 and invalid rate that resample whole caller-declared clusters with
  `numpy.random.Generator(PCG64(seed))` (default 10,000 resamples), score paired systems on the
  same resamples and difference them before the Hyndman-Fan type 7 quantiles.
- `strata_forge.pipelines.eval_runner` is an evaluation run kind: it joins pinned prediction
  parquet files to a pinned gold split on a stable item id, computes the candidate's and each
  baseline's metrics overall and per declared slice with cluster bootstrap intervals and paired
  deltas, marks slices under `min_slice_rows` diagnostic, and pushes `items.parquet`,
  `metrics.json`, `manifest.json` and a dataset card to a private repo in one commit (or keeps
  them on the machine). `EvalSpec` / `EvalSettings` / `PredictionSource` are its spec models.
  Every row has a status (`ok` / `tier_refused` / `excluded` / `overflow` / `missing`): a
  declared `tier_rule` refuses rows for every system, the primary read is over the rows no system
  overflowed with each system's full-tier read beside it, missing rows are always invalid,
  comparisons report the paired discordance rate `p_d`, rows without a cluster cluster on the
  SHA-256 of their normalised premises, and the manifest records the bootstrap contract.
  Resamples where macro-F1 is undefined are counted, never zeroed. An interval with more than
  `unstable_threshold` (default 0.01) of its resamples undefined is flagged `unstable`, and a
  delta's `excludes_zero` gate read is then `unresolved`. A tier's token-count method is
  recorded and must not mix, a declared premise count must match the platform's, and rows keyed
  on the premise-hash fallback are counted per slice.
- The clean-install job gains the evaluation machine's `scoring,storage,hf` leg.
- The clean-install smoke proves the run kinds compose end to end on a fresh install: an
  `hf_classify` scoring run's results parquet feeds an evaluation unchanged (its `item_id`,
  `pred`, `overflow` and raw `logit_<label>` columns, the last read by calibration), and on the
  `all` leg a tiny `seq_cls` verifier is trained with a calibration split, scored through
  `artifact: strata_verifier`, and evaluated with its `calibration.json`.
- `strata_forge.metrics.calibration_fit`: `fit_temperature` (NLL bisection on 1/T within
  [0.05, 20], the bound reported when taken), `cross_fit_temperature` (the seeded permutation of
  sorted cluster keys dealt into folds, the first ceil(C/2) in fold 0 for two, each fold scored
  with the temperature fitted on the others) and `expected_calibration_error` (top-label, 15
  equal-mass bins with ties by item id, or equal-width).
- `cluster_mean_bootstrap` / `paired_cluster_mean_bootstrap` bound the mean of a per-row score
  (a Brier term) under the cluster bootstrap contract; the paired form adds the SD of the per-row
  differences with its interval.
- The evaluation runner's optional `calibration` settings report three probability reads apart
  (raw, a temperature cross-fitted by cluster over the split or a named slice of it, and the
  system's own pinned `calibration.json` temperature) with sum-form Brier and its interval, NLL and
  ECE, plus a paired Brier delta and the SD of the per-item Brier differences per baseline.
- A latency benchmark runner, `strata_forge.pipelines.benchmark_runner`, measuring one pinned
  model (`model_revision` required) on every row of one pinned workload split
  (`dataset_commit_sha` required; a split past `item_cap` is refused) over a grid of batch sizes
  and concurrent request streams. The headline path is `hf_classify` with
  `artifact: strata_verifier`: `total_ms` runs from the raw cells through the verifier's own
  rendering, the forward pass and heads, the 3-way softmax, `p_inconsistent` and
  `support_scores` to ONE `torch.cuda.synchronize()`; `forward_ms` is CUDA events around the
  forward pass in the same request, a diagnostic. The plain AutoModel path is recorded as a
  third-party reference model. `vllm_chat` times each chat request end to end, with no client
  cache, one attempt and the server's prefix cache off unless declared. Three recorded seeds
  (`sample_seed`, `order_seed + pass`, `warmup_seed + pass`); warmup before every pass; lengths
  and L/P buckets measured with each model's own tokenizer and template; exclusions counted by
  reason. A GPU another process holds refuses the run at the start and stops it as incomplete
  (`gpu_busy`) before a pass; clocks, power and temperature are recorded around every pass. Only
  completed passes are summarized, and the headline is the median of the per-pass p95 with min
  and max, readable only for a complete run on a GPU checked idle. `timings.parquet`,
  `summary.json` (`BenchmarkSummary`, finite numbers only) and `manifest.json` (`kind`
  `benchmark`) are pushed in one commit to a private output repo of the run's own, also when the
  model server dies mid-run (the run then fails as incomplete). The `end` event carries
  `gpu_count` and `gpu_seconds` over the GPUs `CUDA_VISIBLE_DEVICES` exposes.
- Premise-prefix KV reuse for a decoder verifier's latency benchmark (`prefix_reuse`,
  `prefix_cache_max_bytes`; batch size 1): per timed pass, the first request of each premise set
  fills its prefix inside its clock and a later one in the same pass reuses it; the entries are
  dropped after each pass's warmup (the process stays warm), each pass's hits, fills and
  evictions are recorded with the byte limit, an in-pass eviction is flagged, and each
  configuration adds a no-cache diagnostic pass. `prefix_cache_hit`
  and `pass_kind` in `timings.parquet`; `prefix_cache`, `no_cache_diagnostic` and `prefix_reuse`
  in the summary. `score_with_prefix` / `timed_prefix_request` / `PrefixCache` on the scoring
  engine, exact against the full forward pass. A decoder with sliding-window layers is refused
  prefix reuse at load under a transformers without `DynamicCache.activate_past_recording`
  (`sliding_rollback_supported`), and the engine facts record the transformers version.
- `strata_forge.metrics.latency`: `percentile` (type 7, equal to `numpy.percentile` without
  NumPy), `latency_summary` (per-pass p50/p95/p99, their median across passes with min and max,
  and the pooled percentiles) and the opt-in, diagnostic-only `latency_bootstrap` interval.
- `Classifier.timed_forward`, `Classifier.forward_timer` (`ForwardTimer`: CUDA events, no added
  synchronization), `Classifier.synchronize`, `Classifier.attention` and
  `timed_verifier_request` on the scoring engine.
- The clean-install matrix gains the benchmark machine's extras, `serving,storage,hf,scoring`,
  and the smoke script a `check_benchmark` on the tiny classifier.
- The batch-inference runner names rows by a declared column: `RunSpec.id_column` makes each
  results row's `custom_id` that column's value (present, at most 512 characters and unique, else
  the run is refused naming the row position, never the value); unset, rows keep their position
  `row-<i>`. `RunSpec.row_ids` (1-100,000 distinct ids) runs only those rows, selected from the
  whole split whatever `row_limit` says, and refuses a run whose ids are not all found. Both are
  chat-only. The manifest records `id_column` and `row_ids` as `{count, sha256}`.
- Results rows gain `latency_ms` (the client's per-request measure, `None` without a response) and
  `error_kind` (`timeout`, `rate_limited`, `model_error`, `invalid_output` or `other`, from the
  exception's type; `None` on success).
- A chat run writes `receipt.json` (`strata.run.receipt/v1`) into its working directory, where the
  orchestrator reads it before the workdir is removed: the manifest's SHA-256, the model and
  dataset pins requested and served, the template digest, the engine version, `id_column`,
  `row_ids`, row counts and the output repo with the commit the push made. `_push_results` returns
  that commit id beside the repo (`commit_oid`).
- A chat run records its failed rows in `failed_rows.json` beside its receipt (`failed_rows_record`:
  the distinct failed ids when rows are named by `id_column`, bounded by count and size, with
  `complete`), the commit it read its split at (`dataset.commit`, resolved before the load when the
  run is unpinned), and checks one id rule shared with the control plane (`valid_row_id`: 1-512
  printable characters, not blank). `row_ids` are selected from the rows the run reads, so
  `row_limit` bounds a retry's window, and its id uniqueness check, as it bounded its source run.
- A pushing chat, scoring or fine-tuning run ends with `repo@commit` (`_common.pushed_at`), as an
  evaluation run already did, so the orchestrator records the commit the output landed in
  (`output_commit_sha`) rather than a repo whose head can move; a push the Hub named no commit for
  ends with the repo alone.
- The chat receipt names a prompt-build run's builds: `prompt` and `system_prompt`, each
  `{build_id, build_hash}` from the manifest's prompt blocks (null for a template run).
- Every pushing run kind writes a receipt, with a `kind` (`chat`, `scoring`, `training`, `eval`):
  scoring (`hf_classify`, `hf_label_logits`, verifier artifacts), fine-tuning (incl. `seq_cls`) and
  evaluation runs write `_common.build_receipt`'s shape (manifest digest, the model, dataset and
  engine their manifest records, row counts where they have rows, and `output` `{repo_id,
  commit_sha}` of the commit the push made). The commit is what links one run's output to the run
  that reads it next. `finetune_runner._push_artifact` and `eval_runner._push` return
  `(repo_id, commit_sha)`; a push failure still leaves the receipt before the run fails.
- `hf_classify` scores a verifier trained by `seq_cls` as it was trained
  (`hyperparams.artifact: strata_verifier`): rendering, labels, pooling and
  auxiliary heads come only from its `strata_verifier.json`, through the one
  render module the trainer uses (`training/verifier_render.py`). Results add
  `p_inconsistent` and per-premise `support_scores` when those heads exist;
  `apply_calibration: auto` applies the artifact's `calibration.json` to `p_*`
  with the logits kept raw. Training checks its own artifact through this path
  before the push.
- `Hyperparams.n` (1-16) draws `n` samples per chat row, one request each; a seeded run seeds sample
  `k` with `seed + k`, the seed vLLM gives the `k`-th of `n` parallel samples. A run that declares
  `n` keys its rows by `(custom_id, sample_index)` and records each sample's `seed`; the manifest
  records `sampling` and the distinct prompt count. `n > 1` below temperature 1e-5 (vLLM's greedy threshold) is refused, and a chunk holds
  `progress_chunk` requests, so cancel checks and progress keep their spacing at any `n`.
- `Hyperparams.chat_template_kwargs` also applies to chat runs, fixed for the run and sent as vLLM's
  `chat_template_kwargs` (`enable_thinking` passes only this way, as a boolean). A kwarg the
  snapshot's chat template never reads, a model without a template and a gpt-oss model (rendered
  with harmony, not the template) are refused before the model server starts; the manifest records
  the template's and the kwargs' sha256.
- `Hyperparams.guided_json`: one JSON Schema per chat run, sent with every request as vLLM's
  `structured_outputs.json` constraint and checked against every reply. A reply that fails
  validation gets one repair turn (the invalid reply plus a fixed message naming the error, under
  the same constraint). Rows gain `json_valid`, `json_retried` and `json_error`; the manifest
  records the schema's and the repair prompt's sha256 and counts valid, invalid, retried and
  repaired rows. A schema over 16 KiB, invalid for its draft, or referencing anything outside
  itself is refused at spec load, and references are never fetched.
- `Hyperparams.engine` also takes `hf_label_logits`: a causal LM's next-token logits for DECLARED
  single-token labels, read after the model's chat template, with no model server. Each row's
  `template` is one user message; `Hyperparams.chat_template_kwargs` passes the template's own
  variables (per-row `{name}` placeholders allowed in string values; the template's own
  parameters, the conversation and special tokens refused; a kwarg the template never reads
  refused at load), and `Hyperparams.assistant_prefill` is appended after the assistant header.
  `Hyperparams.label_tokens` lists `{label, token}` in order; a token that is not one distinct
  token, alone and as the continuation of the first prompt, is refused. Rows carry the raw
  `logit_<label>`, the softmax over the labels, `pred` and `mass_outside` (the full-vocabulary
  probability on every other token). It shares `hf_classify`'s frame: drop-and-count overflow,
  `id_column` / `slice_column`, a required `dtype` (TF32 off in float32), the batch plan, the dataset
  commit read and a `scoring` manifest, which also records the
  chat template's, the canonical kwargs' and the prefill's sha256, the prefill's token ids and each
  label's resolved token id.
- The `seq_cls` fine-tuning method (`SeqClsConfig`, `SeqClsRunner`) and the `nli_pair` dataset
  format: a premise/hypothesis classifier on `transformers.Trainer` with full weights, bf16
  autocast over float32 master weights, length-grouped batches and an optional activation
  checkpointing flag (`use_reentrant=False`, `preserve_rng_state=True`, off by default).
  Gradient accumulation is refused by name. The objective (`classifier_loss`) is grouped soft
  cross-entropy plus an optional sum-form Brier term, where a two-way row's negative is the
  aggregate of the other labels, plus null-masked auxiliary flag and per-premise support heads.
  Before step 0 a run checks every row's label and length (a row over `max_length` refuses the
  run; nothing is truncated), loads under a strict guard that allows only the declared head keys
  of the architecture to be missing, initialises a decoder's classifier from its own LM head
  (prediction head, verbalizer rows of the output projection and its bias) and verifies that step
  0 reproduces the base model's zero-shot label logits, and measures one optimizer step at the
  longest real batch against the device. `head_init="existing"` continues a classifier a previous
  run saved. Intermediate checkpoints never enter the artifact, which carries the stock
  checkpoint, `strata_aux_heads.safetensors` and `strata_verifier.json`. Premises are lists of
  `{pid, text}` with pids `p1..pN`; a two-way row's positive label and its two spellings are
  declared, with no default. Logit comparisons run in float32 with TF32 off and require the same
  predicted label on every row. The length receipt carries the per-row length distribution, and
  the memory preflight also measures the in-run eval pass. After training the saved artifact is
  reloaded as a scorer would and must match the trained logits (`ArtifactCheckError` otherwise),
  with zero-held parameters asserted exactly zero at save and reload; a declared
  `calibration_split` gets a temperature fitted on it and written to `calibration.json`.
- `FinetuneSpec.model_revision`: a full commit id pins the base model for every load of a
  fine-tuning run; `seq_cls` requires it and `dataset_commit_sha`.
- Every fine-tuning run writes a `strata.run.manifest/v1` manifest of kind `finetune`, with each
  check's receipt for `seq_cls` and the measured `gpu_seconds`, pushed with the artifact in one
  commit to a private model repo (refused when public, before any GPU time and again at push).
  The terminal `end` event carries `gpu_seconds`.
- `finetune_runner.check_spec`, the spec checks that need neither the machine nor the network,
  for a control plane to run at submit.
- `build_trainer(..., revision=...)` on `SFTRunner` and `PreferenceRunner`, and `revision` on
  `model_load_kwargs` / `tokenizer_load_kwargs`.

### Changed

- `pipelines._common.ensure_private_output` is the one private-output-repo check, shared by the
  inference and evaluation runners.
- `jsonschema` (`>=4.23,<5`, the range litellm already requires) is a direct dependency: guided
  JSON validates schemas and replies with it.
- A fine-tuning push no longer uploads `checkpoint-*` directories.
- Only the `unpaired_preference` format coerces `label` to a boolean.

### Fixed

- `hf_classify` refuses a list-valued column whose elements are not strings or numbers (a
  `list<struct>` premises column, a nested list, a boolean, a null), naming the column and asking for
  a `list<string>` column, instead of scoring each element's Python text.

## [0.6.0] - 2026-11-18

### Added

- The batch-inference runner serves reasoning models with their reasoning kept apart from the
  answer. `Hyperparams.reasoning_parser` (`openai_gptoss` | `qwen3`) becomes vLLM's
  `--reasoning-parser`, results rows gain a `reasoning` column (string or `None`) and `output`
  holds the final text only. Before the dataset or the weights are downloaded the runner reads
  only the model's `config.json`: a gpt-oss checkpoint (`model_type` `gpt_oss`) with no parser is
  served with `openai_gptoss`, any other parser on it is refused, `openai_gptoss` is refused on
  another model and `qwen3` on anything but `qwen3` / `qwen3_moe`. An unpinned run downloads the
  weights at the commit its config was read at. The manifest records the parser served with as
  `serving.reasoning_parser`.
- `Hyperparams.reasoning_effort` (`low` | `medium` | `high`) is sent in `extra_body` with every
  request, and is refused with the `qwen3` parser or on a model that is not gpt-oss.
- `Hyperparams.gpu_memory_utilization` and `Hyperparams.max_num_batched_tokens` set vLLM's
  `--gpu-memory-utilization` and `--max-num-batched-tokens`, the two knobs vLLM's gpt-oss recipe
  uses to fit gpt-oss-120b on one 80 GB GPU. `gpu_memory_utilization` is a strict float (a
  boolean or a string is refused).
- `Hyperparams.harmony_date` (`YYYY-MM-DD`, a real calendar date, gpt-oss only) pins
  the "Current date" of gpt-oss's harmony system message through vLLM's `VLLM_SYSTEM_START_DATE`.
  Unset, a gpt-oss run pins the serving machine's local date once at start-up, so every row states
  the same date. The manifest records the date used as `serving.harmony_date`.
- The inference manifest records the model server's command-line flags as `serving.flags`.
- `LLMResponse.reasoning` carries the plain-text reasoning a chat-completions server returned
  (`reasoning` / `reasoning_content`, also read from LiteLLM's `provider_specific_fields`),
  separate from `text`.
- The inference runner takes `Hyperparams.engine`: `vllm_chat` (the default, unchanged) or
  `hf_classify`, exact sequence-classification scoring with no model server. Each row renders
  into a pair (`template` = the premises, a list column joined by `list_separator`;
  `pair_template` = the hypothesis); labels come from `config.id2label` by class id through a
  REQUIRED `label_alias` naming every class exactly (`null` excludes a class; many-to-one is
  refused), and `dtype` is REQUIRED (`float32`, with TF32 switched off, or `bfloat16`). Pairs over
  the model's token
  limit are dropped before the forward pass, never truncated, counted, and written with
  `overflow` true and no score. Results carry raw `logit_<label>` columns beside an uncalibrated
  softmax and the argmax `pred`, a stable `item_id` from `id_column`, and a `slice` from
  `slice_column`. The `scoring` manifest (`kind` `scoring`) records the model's `id2label` and the
  alias used, the length limit (less the padding offset on RoBERTa-style models), dtype and TF32
  state, the batch plan, the overflow policy, counts per slice, and the dataset commit actually
  read (an unpinned dataset is resolved once and loaded at that commit). A scoring run pushes only
  to an output repo of its own.
- `strata_forge.training.loading.check_loading_info` refuses a model load that left weights
  uninitialised, mismatched a shape, reported an error or ignored a head tensor, and returns the
  other ignored checkpoint keys for the record.

### Changed

- A model whose response LiteLLM already split into `<think>` reasoning and content (Qwen3 with
  no parser) now has that reasoning in the `reasoning` column instead of losing it.
- The `[serving]` extra declares `torch` and `transformers` directly (the same ranges as
  `[finetuning]`), since the scoring engine imports them itself.

## [0.5.0] - 2026-11-18

### Added

- `strata_forge.metrics.agreement.agreement_bootstrap` returns a seeded item-level percentile
  bootstrap interval (`item-percentile-bootstrap-v1`) for the two-rater agreement coefficient
  that the full-sample reference prevalence selects, as `research_annotation_gate` does: Gwet's
  AC1 at an inclusive 80% reference category, otherwise Cohen's kappa. Every resample keeps that
  coefficient rather than reselecting. Items are resampled with replacement by
  `numpy.random.Generator(PCG64(seed))` (one `(resamples, n)` draw, default 10,000 resamples),
  confusion counts are vectorised across resamples, each resample's value equals the exact point
  arithmetic, and the 95% bounds use Hyndman-Fan type 7 quantiles. Resamples whose kappa is
  undefined (chance agreement one) are counted in `undefined_resamples`. The frozen
  `AgreementBootstrap` records the contract, seed, NumPy version and bounds. NumPy comes from the
  `[scoring]` extra and is imported only when the function runs.
- `strata_forge.metrics.agreement.human_model_agreement` adds the `nominal-human-vs-model-v1`
  statistic for one human (`HumanAnnotations`, `None` = skip) against one model
  (`ModelRatings`, `None` = null label, with `model_prompt_sha256` and `model_run_ref`), beside the
  unchanged two-human statistic. Skips and nulls are excluded from both series and counted. More
  than 10% skips fails the slice (`skip_fraction_exceeded`); that signed failure outranks every
  refusal and lists each applicable one beside it in an ordered `reasons` tuple, such as
  `reference_unavailable` or `model_null_fraction_exceeded`. Without it, more than 10% nulls or
  a missing reference label refuses the slice. Both fractions are parameters in `[0, 1)`, and the
  threshold is bounded to `[-1, 1]`. The reference-prevalence selector runs once over all slice
  items, and kappa is reported beside a selected AC1. The exact point value gates at an
  inclusive 0.70; a value, kappa and the bootstrap interval are reported only when a coefficient
  is defined and the slice is not refused, and the interval never gates or refuses. A report-only
  user-versus-solver kappa is reported beside it. The frozen `HumanModelAgreement` holds exact
  rational values and no item IDs or labels.

## [0.4.0] - 2026-11-17

### Added

- The batch-inference runner pins a model to a full Hub commit id (`RunSpec.model_revision`;
  a branch, tag or abbreviated id is refused), sends `Hyperparams.seed` and `Hyperparams.top_k`
  with every request, adds `finish_reason` and per-row token counts to the results, and writes a
  content-free `manifest.json` (`strata.run.manifest/v1`, `kind` `inference`) beside them: the
  served model commit, the engine and installed library versions, the served context length and
  dtype, the template digest, the settings and the counts. Results and manifest are pushed in
  one commit, only to a private output repo: a public one is refused before the run starts and
  again at push time. This changes the runner spec, so the next release is a minor bump.
- `HFHubClient.is_private` reports a repo's visibility, since `create_repo(private=True,
  exist_ok=True)` leaves an existing repo's visibility unchanged. `HFHubClient.upload_files`
  uploads exactly the given files in one commit.
- `strata_forge.pipelines._common` holds the helpers every runner's manifest shares: the commit-id
  check, the snapshot-commit reader, installed-version capture and `strata.run.manifest/v1`.
- `strata_forge.metrics.comparison` provides two-sided Wilson proportion intervals, exact
  conditional McNemar tests with all-pair accuracy deltas and finite log/underflow provenance,
  and explicit named-family Holm correction that retains caller order. Frozen results and
  evaluation aliases require no numerical extra and choose no rejection threshold. Large
  binomial tails avoid cancellation from subtracting log-gamma values. Bootstrap execution
  remains outside these helpers' contract.

- `strata_forge.metrics` provides explicit-label classification reports, accuracy, macro-F1,
  confusion matrices with invalid-output counts, multiclass sum-form Brier score, and NLL.
  Undefined values, denominators, probability coverage, clipping, and raw/restricted probability
  provenance are preserved in serializable frozen models. `evals.classification` re-exports
  the same functions. Pure metric functions are synchronous under ADR 0020.
- Temperature application is stable and loads NumPy lazily through `[scoring]`. A draft
  `TemperatureCalibration` component records its label order, fit provenance, and optional
  before/after scores; it does not fit temperatures or write a training calibration file.

## [0.3.0] - 2026-10-06

### Added

- Every runner spec (`RunSpec`, `FinetuneSpec`) carries `engine_version: str | None`, the engine
  the orchestrator validated the spec against. `load_config` refuses any other installed engine
  with `engine version mismatch`, read off the parsed JSON before the model validates any other
  field (so a stale machine is blamed even when the spec also carries a field it does not know).
  The version half must equal both the executing `strata_forge.__version__` and the version the
  installed distribution's metadata records; a `"<version>+<commit>"` value is accepted only when
  the installed distribution's PEP 610 `direct_url.json` records that exact full commit id (an
  index install or an editable checkout, which record none, cannot satisfy a commit claim, and a
  `+` followed by anything else is refused as malformed). A spec class that does not declare the
  field is a `TypeError`, so a runner cannot opt out. `strata_forge.pipelines.SPEC_VERSION`
  exports the value an orchestrator writes.
- `None` (no claim) is accepted as the transition for an orchestrator from before the handshake,
  and recorded: the run's first `phase` event and a stderr `warning:` line say the installed
  engine was not checked. Setting `FORGE_REQUIRE_ENGINE_VERSION=1` in the runner's environment
  refuses a missing claim like any other mismatch, so an orchestrator that stamps every spec can
  close the transition on its machines at once. The default flips to refusal in the first minor
  release after every orchestrator that launches this engine stamps `engine_version`.
- The version compare is string equality, never a PEP 440 normalisation (`0.3`, `v0.3.0` and
  `0.3.0.post0` are refused for `0.3.0`), and the mismatch message caps the echoed claim.
- The model registry carries the current Anthropic and OpenAI lineups: `claude-fable-5-1`,
  `claude-opus-5-5`, `claude-sonnet-5-5` (anthropic, bedrock and vertex routes) and
  `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna` (openai route), each copied from the vendor's own
  model, pricing and guide pages. A native `anthropic` / `openai` route refused every one of these
  ids with `unknown_model`.
  - The three Claude 5.x entries have `structured_output: false`: they reject the forced tool
    call that structured output on an Anthropic route relies on, so `complete_structured` now
    refuses them pre-flight with `capability_missing`. Tool calling works.
  - All three GPT-6 entries have `tool_calling: true` on their Responses API route (below).
- OpenAI's Responses API ([ADR 0018](docs/architecture/adr/0018-openai-routes-speak-the-responses-api.md)).
  A registry route carries `wire_api: chat_completions | responses` (`responses` only on `openai`
  and `azure`), copied onto `ModelRoute.wire_api`. Every OpenAI model's `openai` route, and the
  `azure` routes of `gpt-5.5`, `gpt-5.5-thinking` and `gpt-5.5-instant`, speak it through
  `litellm.aresponses`; `openai_compat` (OpenRouter, a self-hosted vLLM, Ollama), Anthropic,
  Vertex and Bedrock stay on Chat Completions. The call surface is unchanged.
  - Requests are stateless (`store: false`, `include: ["reasoning.encrypted_content"]`, never
    `previous_response_id`), and `provider_extras` that set `store`, `previous_response_id`,
    `conversation` or `background` raise `ValidationError`. A turn's output items (encrypted
    reasoning, text with its `phase`, function-call references) come back on
    `LLMResponse.provider_items` and the final `ResponseChunk`, ride on
    `AssistantMessage.provider_items` through both tool loops and `PendingToolCalls.messages`,
    and are replayed verbatim to the provider that produced them (a replayed message with the
    `status` its input shape requires). The `complete()` cache never stores them.
  - In `provider_extras`, an `include` adds to the encrypted-reasoning include, a `text` object
    merges into the computed one (a structured-output format survives unless it sets its own),
    and `reasoning_effort` / `verbosity` / `response_format` become `reasoning.effort` /
    `text.verbosity` / `text.format`.
  - A registered model that LiteLLM's model map lacks is registered with LiteLLM from the Forge
    registry (streaming flag, limits, prices) before a Responses call. LiteLLM fakes the stream
    of a model it cannot look up, so the GPT-6 and `gpt-5.5` streams otherwise depended on
    LiteLLM fetching its remote map at import.
  - `response.incomplete` reports `length` / `content_filter`, `response.failed` and `error`
    events raise a mapped `ProviderError` (`map_responses_error`, reading an `error` event's code
    and message at its top level or in its nested `error` object), and a stream that ends before
    its terminal event raises `ProviderServerError` instead of reporting a clean stop.
  - `AzureConfig.responses_api_version` (default `v1`, env `AZURE_OPENAI_RESPONSES_API_VERSION`)
    selects Azure's `/openai/v1/responses` endpoint; Chat Completions keeps `api_version`. An
    OpenAI organization (`OPENAI_ORG_ID`) is sent as the `OpenAI-Organization` header, which
    LiteLLM's Responses path does not set from `organization`.
  - `Tool` / `ToolDeclaration.to_openai_responses_schema()`, `to_openai_responses_tool_schema`
    and `ImageContent.to_openai_responses_format()` render the Responses wire shapes.
- `Capabilities.sampling_params`: whether a model accepts `temperature` / `top_p` at its default
  reasoning effort. The client refuses them pre-flight with `ValidationError` where the provider
  would reject them: on a Responses route unless the caller sets the reasoning effort to `none`
  (or the model defaults to it), and on any route of a model with `sampling_params: false` (the
  Claude 5.x entries).
- `scripts/smoke_responses.py`: a live two-leg tool-call smoke test of every registered OpenAI
  Responses route (reads `OPENAI_API_KEY` from the environment and never prints it).
- `strata_forge.core.redact`: `Redactor`, the one redaction implementation for every string a
  credential-holding process surfaces and every console a relay forwards. It removes each given
  value as written, line by line, in all four Unicode normal forms, base64-encoded at any
  alignment (standard and URL-safe), percent-encoded and JSON- or `repr`-escaped, plus
  `DEFAULT_PATTERNS` (Hugging Face, OpenAI/Anthropic `sk-`, GitHub, AWS access key ids, JWTs,
  URL userinfo, bearer credentials), plus PEM private-key blocks with or without their footer,
  which stay on with `patterns=()`. Each maximal redacted run becomes one fixed `***`, and runs
  separated only by short whitespace count as one, so the output encodes neither a secret's
  length nor its line count. A value shorter than 8 characters without its surrounding
  whitespace, or made only of `*`, is refused with `ValidationError`; a pydantic `SecretStr` is
  unwrapped. A redactor and its streams refuse to be pickled or copied.
- `Redactor.stream()` returns a `RedactingStream` for text read incrementally: `feed` holds back
  the last `max_len - 1` characters, `flush` releases them, and the concatenated output equals
  `redact` of the concatenated input whatever the piece boundaries, so a secret split across two
  console reads is still caught. `gap()` handles a read that dropped bytes, and a fresh stream
  resumed mid-transcript: the held tail and the first `max_len - 1` characters after the hole
  are masked. `docs/modules/compute.md` states the contract a console relay follows.
- `strata_forge.pipelines._common.run_redactor` builds a run's redactor from its write token.
- A nightly clean-install job (`.github/workflows/clean-install.yml`). It installs each extras
  string a run's machine installs (`[finetuning,storage]`, `[serving,storage,hf]`) plus `[all]`
  into empty Python 3.14 venvs from the public index, with no lockfile and no constraints, and
  `scripts/smoke_clean_install.py` then builds, offline and without a GPU, every enabled
  fine-tuning method x dataset format x adapter through the runner's own spec path and
  `build_trainer` (the TRL config, the peft config, the bitsandbytes config, and the runner's
  trainer keywords bound against the installed trainer's signature), the batch-inference spec,
  requests and results parquet, and the vLLM command line through the installed vLLM's own
  entrypoint parser, with each `VLLM_` variable the runner sets checked as one vLLM still reads.
  A breaking upstream release fails there the day it ships instead of on a user's machine. A
  QLoRA case first checks that `[finetuning]` names `bitsandbytes` and the install has it (QLoRA
  cannot load a model without it), and fails before it builds anything otherwise. It also runs on
  any branch push other than `main` that changes `pyproject.toml`, and a manual run takes an
  `override` of plain version pins (no URL, path or option) as a drill that must turn it red. The
  schedule, which runs from `main`, only dispatches the job on `dev` and reports that run's
  result, so the unlocked install never runs in the default branch's cache scope.
- `strata_forge.training.require_bitsandbytes()` and `MissingBitsAndBytesError` (a `ForgeError`
  and an `ImportError`): the up-front check that a QLoRA run can quantise its model. It looks
  `bitsandbytes` up without importing it, and refuses one whose metadata records a version below
  the `[finetuning]` floor, naming that version. `QLoRAConfig.to_bnb_config()` calls it first.
- `Task.secrets: dict[str, SecretStr]` carries the credentials a job needs beside the task
  rather than in it: excluded from `model_dump`, `to_yaml`, `repr` and YAML loading, keys shaped
  like environment variable names, at most 8, never also in `env`. The values are masked before
  any validator runs, so no rendering of a validation error (`str`, `errors()`, `json()`) carries
  one; a pickled `Task` still does, so a task with secrets is never pickled. Every backend
  re-validates the task before delivering anything (`Task.revalidated()`), since
  `model_copy(update=...)` skips validation. Every backend delivers the secrets as a 0600
  `.secrets.json` in a 0700 directory, exports its absolute path in `FORGE_SECRETS_FILE`
  (`strata_forge.compute.SECRETS_FILE_ENV`) to the task's `run` step only (never to `setup`),
  and removes the file on exit through a trap in an outer shell that the task's own `EXIT` traps
  cannot displace (ADR 0019). `SkyPilotBackend.submit` refuses a task with secrets.
- `SubmitCleanupError` (a `RuntimeError`): `SSHBackend.submit` raises it, from the original
  failure, when a submit fails after creating its workdir and the workdir then cannot be removed.
  Its `job` is a handle `cleanup()` accepts, so an orchestrator can remove that workdir (and any
  secrets file in it) once the host is reachable. A cancelled submit is still re-raised as the
  cancellation.
- `strata_forge.pipelines.HF_TOKEN_SECRET` (`"HF_TOKEN"`): the `Task.secrets` key the runners
  read the Hugging Face token from, and the only key they accept.
- `strata_forge.pipelines._common.load_secrets()` and `RunSecrets`: a runner reads its secrets
  file first, deletes it before any network call or subprocess, and gets the values as
  `SecretStr`. A path that is not an absolute `.secrets.json`, a symlink, a file another user
  could read, a FIFO (refused without waiting for a writer), an unknown key (counted, never
  named) or malformed JSON is a named error that never carries the file's contents.
- `CleanupError` (a `ForgeError` and an `OSError`, exported from `strata_forge.compute`):
  `Backend.cleanup` raises it when the job's backend-side state survived the removal. Its message
  names the job and an exit status or exception type, never the remote's output or the state's
  contents. Being an `OSError`, it lands in a caller's existing `except OSError` around
  `cleanup` rather than escaping it.
- `RunSecrets.hub_credential()`: the `token=` argument for a runner's Hub reads, the delivered
  token or `False`, never `None`.
- `strata_forge.training.loading` (`HubToken`, `model_load_kwargs`, `tokenizer_load_kwargs`): the
  terms every model and tokenizer load in the training runners passes (`trust_remote_code=False`,
  `use_safetensors=True` on models, the caller's token).
- `SFTRunner` and `PreferenceRunner` `build_trainer` / `train` take a keyword-only `token`, the
  Hub credential for the loads they make. It is never stored on a config, so it cannot reach TRL's
  arguments or the `training_args.bin` TRL saves beside a checkpoint.
- `HFHubClient(token=False)` sends no token and skips the settings and library fallbacks.
- `build_vllm_task(served_model_name=...)` (`--served-model-name`), for serving a local snapshot
  under its Hub id.
- `inference_runner.SNAPSHOT_PATTERNS` / `SNAPSHOT_IGNORED`: the files the inference runner
  downloads for the model server, and the ones it refuses.

### Changed

- The `gpt-5.5` family's `openai` routes speak the Responses API: `max_tokens` is sent as
  `max_output_tokens`, which bounds reasoning as well as visible output, `response_format` as
  `text.format`, and a `provider_extras` `reasoning_effort` as `reasoning.effort`. This makes tool
  calls work on `gpt-5.5` at its default effort (Chat Completions rejected them) and makes
  `gpt-5.5-pro` callable at all (it is not served on Chat Completions).
- `Usage.input_tokens` is reported net of cached reads and cache writes on every route. Providers
  include both in their input count while `compute_cost` prices each bucket separately, so cached
  tokens were billed twice; Chat Completions usage now also reports `cache_write_tokens` from
  LiteLLM's `cache_creation_input_tokens`. A `BudgetContext` token ceiling counts every bucket
  (uncached input, cache reads and writes, output), so the netting does not loosen it.
- The `unknown_model` message from `resolve()` states only the failure (`Unknown model: '<id>' is
  not available on ...`), because it can reach an application's end users; the remediation (the
  registry file, the `openai_compat` escape hatch) is logged as an `unknown_model` warning. The
  `reason` is unchanged.
- A chain entry pinned only to `openai_compat` is not checked against the registry's capability
  flags even when its id equals a registered name: the operator's endpoint serves its own model.
- Registry corrections from the vendor pages: `gpt-5.5` and `gpt-5.5-pro` have a 1,050,000-token
  context window and 128,000 max output tokens; `claude-haiku-4-5` has 64,000 max output tokens.
  The registry header states the sourcing rule and marks unchecked values with `# TBD verify`.
- The `litellm` floor is `>=1.83,<2`, the release the Responses path is tested against (releases
  before 1.66 have no `litellm.aresponses`).
- `sanitize` and every runner message (phase captions, error events, the stderr failure reason)
  go through `Redactor`, so they also catch the token's encoded forms and the wider set of
  credential shapes. A write token shorter than 8 characters fails the run before any work.
- The `[finetuning]` extra installs `bitsandbytes>=0.49,<1` (and so does `[all]`). There is no
  platform marker because from 0.49 bitsandbytes has a wheel for every platform `torch>=2.9` ships
  a Python 3.14 wheel for; 0.49 is the first release with both that macOS wheel and a CUDA 13.0
  kernel, which the `torch` pin resolves to on Linux. The macOS wheel needs macOS 14 and
  bitsandbytes publishes no sdist, so an Apple Silicon Mac on macOS 12 or 13 can no longer install
  `[finetuning]`. A dependency pin change, so the release that carries it is a minor one
  (`CLAUDE.md` §9b).
- `strata-forge train sft|dpo` builds the adapter config before it resolves the dataset, so the
  QLoRA check runs before the dataset store is read.
- `SSHBackend.submit` creates the job's workdir mode 0700 with no `-p` (an existing directory or
  symlink fails the submit), requires the remote root to be owned by the login user and narrows
  it to 0700, and writes the wrapper and then the secrets file through the SSH channel's stdin
  with `umask 077`, from inside the workdir after checking it is the login user's own, refusing
  anything (a FIFO included) already at the name, so no file content is ever on a remote command
  line. The secrets file is written last, immediately before the launch. A submit that fails
  after creating the workdir removes it within a 10 s bound. Every submit also removes secrets
  files older than 10 minutes from job directories that were never launched (no pid file), so
  `remote_root` must be a directory dedicated to Forge.
- `SSHBackend.cancel` removes the job's secrets file after its final SIGKILL, for the jobs whose
  trap never ran (SIGKILL, or a job without a process group of its own).
- `SSHBackend` treats a remote command that ended without an exit status (asyncssh's `None`) as a
  failure rather than a success.
- The inference runner starts its model server with `LocalBackend(env_inherit=False)` and an
  allow-listed environment (`model_server_environ()`), so the server never sees the spec, the
  secrets file's path or a token. That allow-list also stops variables the server used to inherit:
  an ambient `HF_TOKEN`, `HF_ENDPOINT` (a Hub mirror), `HF_HUB_*` settings such as
  `HF_HUB_ENABLE_HF_TRANSFER`, and any `CUDA_*`/`NVIDIA_*`/`NCCL_*`/`VLLM_*` variable whose name
  contains TOKEN, KEY, SECRET, PASSWORD or CREDENTIAL. A machine that relied on one of those
  reaching vLLM (a token for a gated model, a mirror) loses it. `LocalBackend` no longer passes an
  inherited `FORGE_SECRETS_FILE` to a child.
- `runner_main`'s callable takes `(writer, RunSecrets)` instead of `(writer, str | None)`.
- `Backend.cleanup` returning now means the job's state is gone, on every backend, so an
  orchestrator can retry a cleanup until it succeeds; a caller that relied on `cleanup` never
  raising now sees `CleanupError` for a removal that failed. State that is already gone is still
  a success, and a transport failure still propagates as itself. `SSHBackend.cleanup` enters
  the remote root and removes and checks the workdir from there, since `test` reads a lookup it
  may not make as "absent"; a root it cannot enter counts as clean only when it is provably
  absent. It checks the exit status itself (a nonzero status, or none at all, raises), and a
  failed submit's workdir discard uses the same command. `LocalBackend.cleanup` keeps the job,
  and raises, until `lstat` finds nothing at its secrets directory's path.
  `SkyPilotBackend.cleanup` resolves the request id `down` returns before returning, raises when
  the teardown fails, and treats `ClusterDoesNotExist` as success.
- `strata-forge compute cleanup` exits 1 and keeps the saved job when the backend reports the
  job's state survived, so running it again retries.
- `SSHBackend` refuses a job whose `remote_workdir` is not exactly one directory directly in the
  remote root, which is all `submit` creates: the root itself (`<root>/`, `<root>/.`,
  `<root>//`) or a nested path (`<root>/a/b`) gets the same `ValueError` as a workdir outside the
  root.
- Every Hub read a runner makes (the dataset split, the training base model and tokenizer, the
  merge step, the inference model download) passes the delivered token explicitly, and a run that
  received none sends no credential. A machine whose own `HF_TOKEN` or `huggingface-cli login`
  used to unlock a gated model or dataset no longer does; the token must be delivered with the run.
  A gated or private repo already in the machine's Hub cache still loads from that cache.
- The inference runner downloads the model itself before starting the model server (safetensors
  weights, JSON configs, tokenizer and chat-template files; never a pickle checkpoint or an array
  format that can carry one, a `*.py` file, compiled module or shell script, or `original/`),
  refuses a snapshot with no safetensors weights at its root by name, and serves the local
  snapshot with `--served-model-name <model_id>`, `--load-format safetensors` and
  `HF_HUB_OFFLINE=1`. A model published only as pickle weights, or with its safetensors only in a
  subfolder, no longer serves.
- A fine-tuning run spec's `hyperparams.extra_trainer_args` may no longer set
  `chat_template_path`, `trust_remote_code`, `model_init_kwargs`, `ref_model_init_kwargs`,
  `push_to_hub`, any `hub_*` or `trackio_*` key, `resume_from_checkpoint`, `deepspeed`,
  `fsdp_config`, `accelerator_config`, `logging_dir` or `report_to`; the runner refuses the spec
  and names every such key. A Python caller of `strata_forge.training` keeps the whole passthrough.
- Every model load in the training runners sets `trust_remote_code=False` and
  `use_safetensors=True`, and every tokenizer load `trust_remote_code=False`. A base model
  published only as pickle weights no longer fine-tunes or merges.
- `PreferenceRunner` loads the DPO / KTO reference model itself when it loads the policy without
  an adapter (and `precompute_ref_log_probs` is off), with the policy's own load arguments. TRL
  used to re-download it by name with no token, and in `float32` by default; it now matches the
  policy's precision. Like the policy, it loads with no device placement of its own, so a
  full-weight DPO / KTO run holds both models in host memory until the trainer moves them (TRL
  placed its own reference load with `device_map="auto"`).
- The `jinja2` floor is `3.1.6` (from `3.1`): the fine-tuning runner renders a repo's chat
  template in-process, and earlier 3.1.x releases have sandbox escapes. A machine with an older
  `jinja2` upgrades it on the next install.

### Deprecated

- The `HF_WRITE_TOKEN` environment variable as the runners' token channel. With no
  `FORGE_SECRETS_FILE` set, a runner still reads it, removes it from its own environment and
  records a `phase` event (and a stderr `warning:`) saying the delivery is deprecated; with a
  secrets file set it is never read. It is removed in 0.5.0.

### Removed

- The `azure` route of `gpt-5.5-pro`. Azure does not list the model for its Responses API, and the
  model takes no tools on Chat Completions, so the route failed every tool call; a registry test
  now refuses a Chat Completions route on any tool-capable OpenAI model.

### Fixed

- A QLoRA fine-tune on a freshly provisioned machine failed: `[finetuning]` did not install
  `bitsandbytes`, and since `BitsAndBytesConfig` constructs without it, the run only failed inside
  `from_pretrained`, after the dataset and the model weights had been downloaded. The extra now
  installs it, and a QLoRA run on a machine without it is refused before anything is downloaded:
  `finetune_runner` fails the spec check with `adapter 'qlora' needs bitsandbytes`, and so does
  `strata-forge train ... --adapter qlora`.
- CLI error messages printed bracketed text as rich markup, so an extra's name
  (`strata-forge[finetuning]`) vanished from the line and exception text containing `[/...]` could
  raise inside the error path. They are printed literally.

### Security

- The batch-inference runner's per-row `error` column, written into the results and pushed to
  the Hub, is redacted. It used to carry each failed row's exception text unscrubbed.
- `release.yml`'s build job and every `nightly.yml` job no longer restore the Actions cache. Every
  ref restores the default branch's cache scope, so an entry written there would otherwise shape
  the distributions the publish job uploads, or run beside the nightly's provider keys and write
  token.

- The `openai_compat` provider documents that `OpenAICompatConfig.base_url` is caller-trusted:
  the provider validates nothing about the URL and keeps no private-address block, because
  loopback is its normal target. Its transport follows redirects. A hop to another host or port
  drops the `Authorization` header, so the configured `api_key` stays on the configured host, but
  nothing else is dropped: a 307/308 re-sends the request body, an `extra_headers` credential
  follows every hop, and the final reply (or an error quoting it) reaches the caller. A caller
  that takes `base_url` from an untrusted party therefore compares the whole URL for exact
  equality with one it allows, never the host or a prefix, and passes `api_key` explicitly
  (`docs/modules/llm.md`, "OpenAI-compatible endpoints"). Unit tests run the real LiteLLM, OpenAI
  client and `httpx` stack, streaming and not, against loopback servers and pin both halves, so a
  dependency upgrade that changes either fails there.
- A Hugging Face token handed to a run is no longer written into `wrapper.sh`, placed on a remote
  command line, exported into the job's environment (where the setup step's package installs
  inherited it) or inherited by the model server.
- A cleanup that leaves a job's workdir, and any secrets file in it, behind is reported instead of
  silent, and a job record that names the SSH remote root can no longer make `cleanup` remove
  every job's workdir, a running job's unread secrets file included. A remote root, or a local
  secrets directory's parent, that cannot be searched no longer reads as empty.
- Gated and private models and datasets load with the run's own token, passed explicitly; the
  model server never holds it and runs offline; no load runs a repo's code or unpickles a
  checkpoint, and the model server is told to read safetensors only, so a pickle checkpoint left
  in the shared Hub cache folder is not loaded either.
- A fine-tuning run spec can no longer reach TRL knobs that read or push with the machine's own
  credential, run a repo's code, read a file on the machine or report to a third party (see
  Changed).

## [0.2.0] - 2026-08-27

The first release this changelog records. Earlier history is in the git log.

[Unreleased]: https://github.com/GemovicNemanja/strata-forge/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/GemovicNemanja/strata-forge/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/GemovicNemanja/strata-forge/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/GemovicNemanja/strata-forge/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/GemovicNemanja/strata-forge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/GemovicNemanja/strata-forge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GemovicNemanja/strata-forge/releases/tag/v0.2.0
