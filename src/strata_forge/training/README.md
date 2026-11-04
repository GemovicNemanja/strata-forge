# strata_forge.training

Fine-tuning primitives: SFT via TRL `SFTTrainer`; preference tuning (DPO, ORPO, KTO, GRPO); PEFT
(LoRA, QLoRA); chat-template formatting; sequence packing; and progress telemetry.

`loading` supplies explicit Hub credentials, safetensors-only model loads, and no remote code.
The private `_load_guard` checks bounded checkpoint-completeness reports for trained-artifact
consumers. HF scoring consumes that guard; training initialization and merge integration have
their own contracts. An exact head-key decision never authorizes fresh inference weights.
