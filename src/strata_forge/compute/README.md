# strata_forge.compute

Remote compute orchestration. The module ships typed `Task` /
`ResourceSpec` / `Job` / `JobStatus` shapes, a YAML task loader for
the SkyPilot-task-YAML subset, a `Backend` Protocol that every
implementation satisfies, and an in-process `LocalBackend` for
tests and ad-hoc local runs.

The SSH and SkyPilot backends sit behind the lazy `[compute]` extra,
alongside a batch inference runner that consumes any `Backend`, and
vLLM / TGI / SGLang serving adapters that present as
`strata_forge.llm` `openai_compat` providers. The training runners
(SFT, DPO/ORPO/KTO, PEFT/LoRA/QLoRA) live under
`strata_forge.training`.

Credentials a job needs travel beside the task, never in it:
`Task.secrets` holds them as `SecretStr` values that are never
serialised, rendered or logged, and every backend delivers them as a
private file the job finds through `FORGE_SECRETS_FILE`, never as an
environment variable or on a command line. A backend with no private
channel (SkyPilot) refuses a task that carries secrets.

See [ADR 0013](../../../docs/architecture/adr/0013-compute-task-and-backend-shapes.md)
for the task-as-data + Protocol design rationale,
[ADR 0019](../../../docs/architecture/adr/0019-secrets-travel-beside-the-task.md)
for secret delivery, and `docs/roadmap.md` for the current status.
