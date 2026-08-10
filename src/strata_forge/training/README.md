# strata_forge.training

Fine-tuning primitives that render typed, frozen Pydantic configs into TRL trainers.
`SFTConfig` + `SFTRunner` cover supervised fine-tuning; `DPOConfig`, `ORPOConfig`, `KTOConfig` and
`GRPOConfig` + `PreferenceRunner` cover preference tuning; `LoRAConfig` and `QLoRAConfig` attach
PEFT adapters. `apply_chat_template` and `pack_sequences` prepare data, and both runners can stream
`ProgressEvent` records to a JSONL file that an orchestrator tails while training runs.

Every heavy import (torch, transformers, trl, peft) is deferred until `train()` is called, so
importing this module costs nothing. `train()` is deliberately synchronous — it is a long-running
blocking process, not I/O to await.

Needs the `[finetuning]` extra, and in practice a GPU host to train on.

Reference:
[docs/modules/training.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/training.md).
