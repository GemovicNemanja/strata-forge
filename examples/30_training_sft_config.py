"""Build a Forge :class:`SFTConfig` + LoRA adapter and print the rendered TRL kwargs.

The example does NOT actually train (TRL training requires a GPU
and many GB of model weights). Instead, it shows the full shape
that Forge sends to TRL — the rendered kwargs dict for
``trl.SFTConfig`` and the peft adapter config — so users can see
exactly how their Pydantic config maps onto TRL's surface.

Run the same code on a real GPU host with the ``[finetuning]``
extra installed to launch the training loop.

Usage::

    uv run python examples/30_training_sft_config.py
"""

from __future__ import annotations

import json

from strata_forge.training import LoRAConfig, QLoRAConfig, SFTConfig, SFTRunner


def _main() -> None:
    # Standard LoRA, bf16, packed 4k sequences.
    sft = SFTConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="./checkpoints/llama-3.1-8b-sft",
        max_seq_length=4096,
        packing=True,
        num_epochs=3,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        precision="bf16",
        save_steps=200,
    )

    lora = LoRAConfig(r=32, alpha=64, target_modules=("q_proj", "v_proj"))

    runner = SFTRunner(sft, peft_config=lora)
    print("--- runner: SFT + plain LoRA")
    print(f"model_id        = {runner.config.model_id}")
    print(f"output_dir      = {runner.config.output_dir}")
    print(f"adapter rank    = {lora.r} (alpha={lora.alpha})")
    print(f"target modules  = {list(lora.target_modules or [])}")

    trl_kwargs = sft.to_trl_kwargs()
    print("\n--- rendered trl.SFTConfig kwargs:")
    print(json.dumps(trl_kwargs, indent=2, default=str))

    # QLoRA variant — 4-bit base + nf4 + double quant.
    print("\n--- QLoRA variant (bnb config):")
    qlora = QLoRAConfig()
    print(f"load_in_4bit    = {qlora.load_in_4bit}")
    print(f"quant type      = {qlora.bnb_4bit_quant_type}")
    print(f"double quant    = {qlora.bnb_4bit_use_double_quant}")
    print(f"compute dtype   = {qlora.bnb_4bit_compute_dtype}")
    print(f"adapter rank    = {qlora.lora.r}")


if __name__ == "__main__":
    _main()
