# forge.compute

Remote compute orchestration. The module ships typed `Task` /
`ResourceSpec` / `Job` / `JobStatus` shapes, a YAML task loader for
the SkyPilot-task-YAML subset, a `Backend` Protocol that every
implementation satisfies, and an in-process `LocalBackend` for
tests and ad-hoc local runs.

Phase 5.1 ships the foundation; Phase 5.2 adds the SSH and
SkyPilot backends (lazy `[compute]` extra) plus a batch inference
runner that consumes any `Backend`. Phase 5.3 adds the training
runners (SFT, DPO/ORPO/KTO, PEFT/LoRA/QLoRA) under
`forge.training`. Phase 5.4 closes out with vLLM / TGI / SGLang
serving adapters that present as `forge.llm` `openai_compat`
providers, plus examples and module docs.

See [ADR 0013](../../../docs/architecture/adr/0013-compute-task-and-backend-shapes.md)
for the task-as-data + Protocol design rationale, and
`docs/roadmap.md` for the current status.
