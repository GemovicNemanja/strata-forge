# strata_forge.compute

Remote compute orchestration. A unit of work is inert data — `Task` and `ResourceSpec`, which
round-trip through the SkyPilot-task-YAML subset — and every backend satisfies one `Backend`
Protocol (`submit`, `status`, `logs`, `read_file`, `cancel`, `cleanup`) returning `Job` and
`JobStatus`. See
[ADR 0013](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0013-compute-task-and-backend-shapes.md)
for the task-as-data plus Protocol design.

Three backends ship: `LocalBackend` (subprocess, for tests and ad-hoc runs), `SSHBackend` and
`SkyPilotBackend`. On top of them sit `BatchInferenceRunner` for concurrency-bounded fan-out, and
`build_vllm_task` / `build_tgi_task` / `build_sglang_task` plus `serving_endpoint` for standing up
a self-hosted inference server that `strata_forge.llm` talks to over its `openai_compat` route.

`SSHBackend` and `SkyPilotBackend` need the `[compute]` extra; `LocalBackend` does not.

Reference:
[docs/modules/compute.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/compute.md).
