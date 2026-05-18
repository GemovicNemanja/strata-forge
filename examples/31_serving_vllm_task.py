"""Print the YAML task that Forge ships to a backend to launch vLLM / TGI / SGLang.

The example builds a :class:`Task` for each of the three serving
adapters and prints the YAML you'd submit to a SkyPilot or SSH
backend. No GPU is required.

To actually launch one of these you'd do something like::

    from forge.compute import LocalBackend
    from forge.compute.serving import build_vllm_task, serving_endpoint
    from forge.llm import LLMClient
    from forge.llm.providers.config import OpenAICompatConfig

    task = build_vllm_task("meta-llama/Llama-3.1-8B-Instruct", port=8000)
    backend = LocalBackend()
    async with serving_endpoint(
        backend, task, base_url="http://localhost:8000/v1"
    ) as endpoint:
        client = LLMClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            provider="openai_compat",
            provider_config=OpenAICompatConfig(base_url=endpoint.base_url),
        )
        response = await client.complete(messages=[...])

Usage::

    uv run python examples/31_serving_vllm_task.py
"""

from __future__ import annotations

from forge.compute import ResourceSpec
from forge.compute.serving import build_sglang_task, build_tgi_task, build_vllm_task


def _main() -> None:
    vllm = build_vllm_task(
        "meta-llama/Llama-3.1-8B-Instruct",
        port=8000,
        tensor_parallel_size=1,
        max_model_len=8192,
        dtype="bfloat16",
    )
    tgi = build_tgi_task(
        "mistralai/Mistral-7B-v0.1",
        port=8080,
        num_shard=2,
        max_input_tokens=4096,
        max_total_tokens=8192,
    )
    sglang = build_sglang_task(
        "Qwen/Qwen2-7B-Instruct",
        port=30000,
        tp_size=4,
        resources=ResourceSpec(accelerators="H100:4"),
    )

    for label, task in [("vLLM", vllm), ("TGI", tgi), ("SGLang", sglang)]:
        print(f"--- {label} task ({task.name})")
        print(task.to_yaml())
        print()


if __name__ == "__main__":
    _main()
