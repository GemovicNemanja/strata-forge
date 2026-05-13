# forge.compute

Remote compute orchestration. Programmatic SkyPilot submission via `sky.api.sdk` reaches Nebius / AWS / GCP / Azure / RunPod / Lambda / Fluidstack / Kubernetes; a raw `asyncssh` SSH backend handles direct submit/exec/file-transfer/log-tail. SkyPilot task YAML templates for SFT, DPO, vLLM/TGI/SGLang serve, and eval. Async concurrency-controlled batch inference layered on top of `forge.llm`.

> Implementation pending. See `docs/roadmap.md` for current status.
