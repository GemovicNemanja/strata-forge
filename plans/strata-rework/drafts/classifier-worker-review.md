# TRAIN-001: launchable primary classifier worker

Status: proposal for coordinator review, 2026-11-04. This packet changes no repository source,
registers no route/tool, executes no training, and grants no research/data/compute authority.
Implement only after exact file leases and the private worker contract are approved.

## Recommendation

Add `seq_cls` to the existing fine-tuning lifecycle, with a separate fixed installed classifier
worker. Reuse `ClassifierRunner`, safe initialization, ordinary custody primitives and existing
SSH supervision. Make safe decoder export→installed `hf_classify`→evaluation consumption part
of this same delivery. Do not feed the classifier through TRL's causal `build_trainer` path.

The first useful worker supports full-model primary classification, generic CE or the explicit
existing CE+0.1 Brier research preset, final-only selection, safe export and exact pinned input
provenance. It has no LoRA/QLoRA, optimizer resume, calibration, auxiliary heads, weighting,
smoothing or grouped/masked NLI binary objective. Those remain explicit subsequent increments,
not fields whose mere presence claims implementation. It does not change scientific gates.

The work has three reviewable units with one TRAIN-001 acceptance owner:

1. **Forge worker plus scoring interop.** Strict inert spec, pinned safe loading, bounded row
   projection, native SDK training, safe complete export and fixed installed decoder reload in
   `hf_classify`. Synthetic CPU proof includes the actual evaluation computation. No Hub call.
2. **Strata dispatch and read authority.** Existing `kind=finetune`, `method=seq_cls`, independent
   training-purpose run credential, exact prepared reads, request-bound submit replay, cleanup,
   local result evidence and disabled-by-default capability. All authority negative tests pass
   before SSH activation. The public server continues to refuse `backend=local`.
3. **Guarded model publication and actual outcome projection.** A private model destination,
   exact expected-parent/create-only custody fence, verified safetensor artifact inventory and
   actual accepted model commit receipt. This is required for an ordinary launched training
   run to feed the current Hub-ref-only EvalSpec without manual out-of-band artifact transfer.
   Initial `output=null` training is useful, but must not be advertised as a complete hosted
   G2→G3 path until this unit is accepted. The unit has an explicit proposed source lease below.

Unit1's decoder export/eval interop is mandatory even if publication activation follows later.
Neither a checkpoint receipt nor a mocked Hub transport proves real publication. A research G2
run additionally requires Claude's accepted objective/input/model/budget/security gates.

## Verified source baseline

| Source | Immutable pin and important seams |
| --- | --- |
| Classifier SDK | `strata-forge` `9006b41867f65784c17a59d8f4dbef03aede0266`, `/private/tmp/strata-classifier-sdk-20261104` |
| Server integration | `5946ebee35e1db545af33dcf1af047ad62ef60f2`, `/private/tmp/strata-server-integrated-20261104`; committed source equivalent to published `1e4eb05c63025409dc1a3e6fa7a7cc0593f0e201` |
| SDK contract | `docs/architecture/adr/0022-primary-classifier-sdk.md`; `docs/modules/training.md` primary-classifier section |
| Existing research plans | `plans/strata-rework/README.md` TRAIN-001 is a broader target, not a statement of current SDK functionality |

Source findings, with line numbers at those pins:

- Forge `training/classifier.py:42,89,374,472` supplies strict rows/config, resolved-object
  training and safe local reload. `training/_classifier_heads.py:23,150,266` owns exact head
  transformations. Initialization receipts explicitly describe caller-resolved objects.
- `training/methods.py:43,56` has no `seq_cls`; `MethodSpec` assumes causal configs/runners.
  `dataset_format.py:40` has neither generic classification nor `nli_pair`.
- `pipelines/finetune_runner.py:120,319,435` assumes TRL, permits an absent dataset pin, has no
  model pin, and uses the causal runner factory. Its ordinary push is not the new classifier
  publication authority. Do not inherit these limitations into the classifier spec.
- `pipelines/_scoring.py:448` always uses Transformers AutoModel for `hf_classify`; it does
  not consume the Forge decoder wrapper. SDK `load_classifier` does, without inference head
  exemptions. The change belongs in this installed loader seam, not in EvalSpec.
- Server `schemas.py:412` currently refuses a fine-tune `model_revision`; `:458` validates
  against the causal registry. `run_tasks.py:412` chooses only `finetune_runner`.
- Server `run_launch.py:104` prepares custody reads only for Eval. The current Eval credential
  in `evaluation_policy.py:150` explicitly requires `Run.kind == eval`. It is not a training
  capability. `run_poller.py:391` trusts a legacy non-Eval terminal message as a repo name;
  classifier output must instead use its accepted receipt or computed retained path.
- SDK does no periodic save/resume: `save_strategy=no`, `eval_strategy=no`, final safe export.
  Coroutine cancellation requests a stop; SSH/LocalBackend process-group cancellation is the
  hard termination boundary. Existing SSH implementation has TERM/grace/KILL group behavior.

## Installed dependency evidence and packaging

Read-only package metadata from the existing SDK `.venv` showed Python3.14.4, Forge0.3.0,
Torch2.13.0, Transformers5.8.1, TRL1.4.0, PEFT0.19.1, Datasets4.8.5,
Accelerate1.13.0, huggingface_hub1.14.0 and NumPy2.4.6. No environment was installed or mutated
and no ML function was run during this audit. These observations are not a lockfile or new
worker acceptance. The SDK owner reports existing tiny CPU tests in this environment.

The least disruptive first worker uses the current `[finetuning,storage]` installation profile.
It is larger than the primary classifier needs, but supplies the tested Torch/Transformers/
Accelerate/Parquet/Hub stack. It need not import TRL/PEFT/bitsandbytes when executing classifiers.
Do not add vLLM/serving to training. A narrower extra is a later packaging change with actual
clean-install proof, not a reason to delay the worker or to claim existing ranges are locked.

Core-only installation must still import training, method descriptors and the inert spec
without Torch/Transformers/NumPy imports. Reuse lazy imports; make missing dependencies a fixed
early installation refusal. Task setup installs an immutable future Forge source SHA and the
worker verifies its engine handshake. Version0.3.0 alone is insufficient. Record the concrete
installed dependency versions in each attempt's metadata.

Current [Transformers5.8.1 Trainer documentation](https://huggingface.co/docs/transformers/v5.8.1/en/main_classes/trainer)
distinguishes training/checkpoint state from model saving and supports installed callbacks.
Use the SDK's tested APIs rather than assuming a different Trainer version's keyword names.
The [Transformers5.5.3 loading contract](https://huggingface.co/docs/transformers/v5.5.3/en/main_classes/model)
exposes pinned revision, explicit token, loading diagnostics and safetensor controls. The worker
must set these terms, reject structural diagnostics and never rely on their permissive defaults.

## Proposed source leases, not current assignments

All implementation starts in isolated topics off coordinated current source, preserving SDK
9006 and existing evaluation-policy retry work. No source lease is claimed by this packet.

| Owner requested | Files and bounded change |
| --- | --- |
| Forge TRAIN worker | New `pipelines/classifier_spec.py`, `classifier_runner.py`, `_classifier_checkpoint.py`; matching unit tests, module docs/rules and one no-network CPU worker→evaluation test |
| Forge registry/rows | `training/methods.py`, `dataset_format.py`, `__init__.py` and tests: typed classifier descriptor/row formats; preserve every causal descriptor and constructor |
| Forge SDK telemetry | `training/classifier.py`, `progress.py` and tests, only a trusted Python progress-sink seam for actual optimizer start/steps/end; no objective/head changes |
| Forge installed scoring | `_scoring.py` plus focused scoring tests: recognized pinned Forge artifact branch, strict `load_classifier`, same class order/probability/padding rules; no EvalSpec/mode/retry change |
| Forge training authority | New `_classifier_authority.py`; additive training-secret constant/parser in `pipelines/__init__.py`, `_common.py` and tests. Do not reuse or broaden Eval credentials. Existing Eval fixtures remain unchanged |
| Server TRAIN adapter | New `classifier_training.py`, `classifier_launch.py`, `classifier_policy.py`, `classifier_policy_schemas.py`, private `routers/classifier_policy.py`; synthetic tests/docs |
| Server dispatch | Narrow coordinated `schemas.py`, `run_tasks.py`, `run_launch.py`, `routers/runs.py`, `config.py`, `main.py`: classifier-only spec/prepare/replay/capability/registration, default off; existing paths byte/behavior compatible |
| Server results/publication | Classifier-only branches in `run_poller.py` and history retention; model-domain custody adapter/receipt tests. Generic cleanup should need no code change; test it |
| Browser/agent follow-up | Fine-tune form/spec/types/grounding and existing `start_finetune` extension require a separate exact file lease and user tool-wire approval. Do not sneak classifier metadata into the pending Eval wire |

For registry typing, use a discriminated classifier descriptor with `task_type=SEQ_CLS` and a
classifier-specific factory contract. Existing causal `MethodSpec.build_runner(peft_config=...)`
must keep its behavior. Do not widen a return union then call `.build_trainer` on ClassifierRunner.
Register/enable `seq_cls` atomically with its handler; a server must refuse when the installed
engine has no validated worker, authority or reload capability.

No schema migration is expected: new spec/credential/receipt records are versioned Run JSON,
and existing Run.project_id can hold the owner's server-resolved personal project. Preserve
User/Principal/vault/legacy Run IDs. No cross-project grant is compute/vault delegation; first
classifier launches remain current-user-owned, with an internal personal-project association.

## Acceptance and gates

See [classifier-worker-contract.md](classifier-worker-contract.md) for exact proposed data/authority/result behavior and
[classifier-worker-example.json](classifier-worker-example.json) for a fictional draft spec.

- Core-only strict config/import proof and actual clean `[finetuning,storage]` constructor test.
- Existing SDK head/loss/safe-reload tests stay green; actual tiny CPU optimizer step changes
  learned weights, exports no pickle/optimizer state, and retained metadata binds actual inputs.
- Tiny BERT plus decoder family fixtures include at least ModernBertDecoder with actual output
  bias and pad=EOS, ordered three-class labels, and no paired decoder input. Complete saved
  decoder passes installed scorer→actual `run_evaluation`; missing/extra/mismatched weights,
  bad receipt/config, reordered labels and unavailable policy refuse before scoring.
- No row/model read before complete training-purpose policy approval. A revoked/expired,
  wrong-run/spec/kind/purpose credential, positive/unknown custody, new producer or changed
  frozen ancestor denies. A credential/fixture trusted for Eval never authorizes training.
- Public HTTP submission→actual typed Task/spec→fixed worker function is covered with a fake
  SSH transport and real synthetic CPU work; no server-local execution escape is enabled.
- Actual process-supervision fixture kills a stuck native training process group, preserves
  retained attempt/artifact evidence and releases guarded reads only after confirmed cleanup.
- Request-bound same-key replay returns the existing run; different body conflicts. Classifier
  rerun creates a new record/attempt; resume is refused. Existing causal and Eval retry behavior
  stays unchanged, including the mandatory Eval new-record rule.
- For unit3, fake Hub/ASGI protocol proof includes private create-only races, expected-parent
  conflict, vanished/changed target, accepted receipt surviving later cancel/failure, and no
  success/link from a requested repo or terminal message. No real Hub write is part of this work.
- Real MODEL jobs, special model conversion, gate/test reads, owner-budget spends and research
  claims remain Claude's responsibility after applicable approvals. No E0 bypass is introduced.

Research readiness is reported separately for G1 platform scoring and G2 training. This packet
does not imply the G1 baseline has run, that a scientific model has trained, that final-only
training meets every planned long-running experiment, or that the broader TRAIN-001 outputs
are implemented.
