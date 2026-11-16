# Task-profile inventory

`strata_forge.pipelines.capabilities` exposes a bounded, versioned source
inventory for existing inference, fine-tuning, classification evaluation,
population and primary-classifier profiles. Import names directly from that
leaf; it adds no package export, dispatcher, CLI, worker registration or remote
API. It performs no target, credential, dependency or hardware inspection.

```python
from strata_forge.pipelines.capabilities import task_capabilities, task_catalog_sha256

catalog = task_capabilities()
identity = task_catalog_sha256(catalog)
for profile in catalog.profiles:
    print(profile.profile, profile.surface, profile.status, profile.installed)
```

## Declared boundaries

`supported` means that the named source surface exists. It is scoped by
`surface`, not a claim that an installation or target can execute it:

| Profile | Surface | Meaning |
| --- | --- | --- |
| `batch_inference_v1` | `inert_worker` | Existing inference module entrypoint and legacy RunSpec, including its actual engine choices. |
| `causal_finetune_v1` | `inert_worker` | Existing fine-tuning entrypoint, with selector declarations composed from training v1. |
| `classification_evaluation_v1` | `inert_worker` | Existing evaluation entrypoint; its actual read/publication authority remains mandatory. |
| `evaluation_population_v2` | `trusted_python` | Explicit Python SDK callable; its population schema does not activate the ordinary worker entrypoint. |
| `scoring_population_v2` | `trusted_python` | Explicit scoring SDK callable; trusted read policy and full population conditions remain required. |
| `primary_classifier_v1` | `trusted_python` | Explicit training SDK callable with installed training authority/source objects; `seq_cls` is not an enabled causal method. |
| `primary_classifier_hf_worker_v1` | `inert_worker` | Fixed installed HF primary-classifier entrypoint over the same `ClassifierTrainSpec`; current training-purpose authority and the Server's configured-profile gate remain required. |
| `generation_v2` | `schema_only` | `experimental` data contract with `no_worker_entrypoint`; validating it cannot launch generation. |

The inventory is deliberately finite. A profile absent from it has no support
claim. It does not enumerate every Forge primitive, trusted compute Task or
operator extension. Benchmarking, calibration, auxiliary training and other SDK
surfaces retain their own contracts and acceptance.

`entrypoint` and `spec_type` are descriptive source locations, never an input
for dynamic import or execution. Every profile reports `admitted=not_assessed`.
The ordinary service must still resolve current authority, immutable inputs,
schema semantics, exact worker version, resources, limits and output policy.
Neither a schema hash nor a copied catalog is an execution grant.

## One training declaration

The nested `training` value is the unchanged
[`training.capabilities` v1 projection](training.md#declaration-capability-metadata)
over the existing METHODS/FORMATS registries. Its `declared` and `refused`
decisions retain their names and refusal precedence. In particular, inert GRPO
still refuses without a reward callable; from-scratch, resume, other task heads
and explicit distributed selectors remain outside that profile. `adapter=none`
means pretrained full-weight fine-tuning rather than random initialization.

The task inventory does not maintain a second method, role or hyperparameter
table. Existing training selectors are assessed through
`assess_training_capability`; complete runtime specifications still use their
own actual schemas and semantic validators.

The fixed HF worker and the trusted primary SDK share the exact classifier spec
schema while retaining distinct entrypoints and surfaces. Its descriptive
profile/result marker creates no dispatcher or causal-training method. The
Server enables that fixed profile only through its operator-configured worker
interpreter and actual admission/policy path; a catalog row cannot enable it or
establish that a selected remote target is ready.

## Schemas, dependencies and identity

Each `spec_schema_sha256` hashes the actual source class's generated JSON Schema,
using compact sorted-key UTF-8 JSON, without a trailing newline. The catalog's
`schema_sha256` hashes its own complete generated schema. The whole catalog hash
also binds engine version, profile ordering, descriptive entrypoints, selectors
and the complete nested training declaration.

Generated JSON Schema does not capture every Python semantic validator or prove
which distribution was installed. `engine_version` is the package's declared
version, not an observed source commit. Consumers requiring execution evidence
also need their exact installed source/distribution and engine handshake.

`primitive_extras` names relevant existing package extras. It is not a minimal
or complete installation command: a profile can have conditional storage,
generation, scoring or publication needs. `dependency_profile`, `installed`,
`environment` and `hardware` are always `unknown` until a separate qualified
installation/target observation establishes them. The inventory neither probes
packages nor fills unknowns from the machine running the control plane.

The HF classifier row names `finetuning` and `storage`: the former declares the
training stack and Arrow/datasets, and the latter declares the Hub client. These
broad extras are not the separately qualified CPU environment or a minimal
installation recipe; they do not imply adapter/causal-method support or actual
dependency compatibility. The source inventory keeps all readiness values
unknown even when one specific operator environment has been qualified.

## Values and compatibility

Returned models and their nested tuples are frozen and detached. Literal choices
are read from the actual spec annotations; the inventory does not duplicate those
engine/mode enumerations. Profiles and choice sets have deterministic ordering.
Canonicalization rebuilds a plain typed copy, including nested training values,
so malformed `model_copy` updates cannot bypass validation. It checks typed
coherence rather than authenticity: constructing or hashing a catalog does not
prove that its descriptive source exists or confer runtime authority.

The frozen JSON fixture and source-entrypoint guards detect changes to this
projection. Compatible metadata additions still change its complete digest;
consumers must explicitly accept schema/profile versions and unknown profiles
must not be treated as executable. Existing worker specs, serialized payloads,
training v1 bytes and their digests are unchanged. App/server/MCP consumption,
installed-consumer conformance and any exact remote-wire approval remain separate.
