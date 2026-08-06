"""Self-hosted inference serving adapters.

Build :class:`strata_forge.compute.Task` instances that launch
OpenAI-compatible inference servers (vLLM / TGI / SGLang) so
Forge can talk to them through the existing
``openai_compat`` provider. The functions in this module **build
Tasks** — they don't submit. Callers pass the returned task to
whichever :class:`Backend` they want and then point a
:class:`strata_forge.llm.LLMClient` at the resulting endpoint::

    from strata_forge.compute import LocalBackend
    from strata_forge.compute.serving import build_vllm_task, serving_endpoint
    from strata_forge.llm.providers.config import OpenAICompatConfig
    from strata_forge.llm import LLMClient

    task = build_vllm_task("meta-llama/Llama-3.1-8B-Instruct", port=8000)
    backend = LocalBackend()
    async with serving_endpoint(backend, task, base_url="http://localhost:8000/v1") as endpoint:
        client = LLMClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            provider="openai_compat",
            provider_config=OpenAICompatConfig(base_url=endpoint.base_url),
        )
        response = await client.complete(messages=[...])

The probe-based readiness wait is intentionally simple — Forge
doesn't run a healthcheck binary, it just polls the ``/v1/models``
endpoint until it responds.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strata_forge.compute.task import ResourceSpec, Task

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from strata_forge.compute.backends.base import Backend
    from strata_forge.compute.job import Job

__all__ = [
    "ServingEndpoint",
    "build_sglang_task",
    "build_tgi_task",
    "build_vllm_task",
    "serving_endpoint",
    "wait_for_endpoint",
]


def _quote_args(args: Sequence[str]) -> str:
    """Render args as shell-safe space-joined string."""
    import shlex

    return " ".join(shlex.quote(a) for a in args)


def build_vllm_task(
    model: str,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",  # noqa: S104 — vLLM defaults to 0.0.0.0 for in-cluster reach
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    dtype: str | None = None,
    extra_args: Sequence[str] = (),
    resources: ResourceSpec | None = None,
    name: str = "vllm-serve",
    setup: str = "pip install 'vllm>=0.7'",
) -> Task:
    """Build a Forge :class:`Task` that launches a vLLM server.

    Args:
        model: HuggingFace model id or local path. Becomes
            ``--model`` for vLLM and the logical model id callers
            pass to :class:`LLMClient`.
        port: HTTP port to listen on.
        host: Bind address.
        tensor_parallel_size: vLLM's ``--tensor-parallel-size``.
        max_model_len: Optional context cap (``--max-model-len``).
        dtype: Optional dtype override (``--dtype``).
        extra_args: Extra CLI args forwarded verbatim to
            ``vllm serve``.
        resources: Optional :class:`ResourceSpec`. Default targets
            a single GPU.
        name: Task name.
        setup: Setup command run before the server starts. Pass
            an empty string when the host already has vLLM
            installed.
    """
    cli_args: list[str] = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if max_model_len is not None:
        cli_args.extend(["--max-model-len", str(max_model_len)])
    if dtype is not None:
        cli_args.extend(["--dtype", dtype])
    cli_args.extend(extra_args)

    return Task(
        name=name,
        run=_quote_args(cli_args),
        setup=setup,
        resources=resources or ResourceSpec(accelerators="A100:1"),
    )


def build_tgi_task(
    model: str,
    *,
    port: int = 8080,
    num_shard: int = 1,
    max_input_tokens: int | None = None,
    max_total_tokens: int | None = None,
    extra_args: Sequence[str] = (),
    resources: ResourceSpec | None = None,
    name: str = "tgi-serve",
    image: str = "ghcr.io/huggingface/text-generation-inference:latest",
) -> Task:
    """Build a Forge :class:`Task` that launches a TGI container.

    Args:
        model: HuggingFace model id.
        port: HTTP port to expose (mapped to the container's 80).
        num_shard: Tensor-parallel shard count.
        max_input_tokens: Optional input-length cap.
        max_total_tokens: Optional total-length cap.
        extra_args: Extra CLI args appended to the container
            command.
        resources: Optional :class:`ResourceSpec`.
        name: Task name.
        image: Docker image. Defaults to the official TGI image.
    """
    docker_args: list[str] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-p",
        f"{port}:80",
        "--shm-size",
        "1g",
        image,
        "--model-id",
        model,
        "--num-shard",
        str(num_shard),
    ]
    if max_input_tokens is not None:
        docker_args.extend(["--max-input-tokens", str(max_input_tokens)])
    if max_total_tokens is not None:
        docker_args.extend(["--max-total-tokens", str(max_total_tokens)])
    docker_args.extend(extra_args)

    return Task(
        name=name,
        run=_quote_args(docker_args),
        resources=resources or ResourceSpec(accelerators="A100:1"),
    )


def build_sglang_task(
    model: str,
    *,
    port: int = 30000,
    host: str = "0.0.0.0",  # noqa: S104 — same rationale as vLLM
    tp_size: int = 1,
    extra_args: Sequence[str] = (),
    resources: ResourceSpec | None = None,
    name: str = "sglang-serve",
    setup: str = "pip install 'sglang[all]'",
) -> Task:
    """Build a Forge :class:`Task` that launches an SGLang server.

    Args:
        model: HuggingFace model id or local path. Becomes
            ``--model-path`` for SGLang.
        port: HTTP port.
        host: Bind address.
        tp_size: Tensor-parallel size.
        extra_args: Extra CLI args forwarded to SGLang.
        resources: Optional :class:`ResourceSpec`.
        name: Task name.
        setup: Setup command run before the server starts. Pass
            an empty string when the host already has SGLang
            installed.
    """
    cli_args: list[str] = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
    ]
    cli_args.extend(extra_args)

    return Task(
        name=name,
        run=_quote_args(cli_args),
        setup=setup,
        resources=resources or ResourceSpec(accelerators="A100:1"),
    )


@dataclass(frozen=True, slots=True)
class ServingEndpoint:
    """A live serving endpoint produced by :func:`serving_endpoint`.

    Attributes:
        job: The :class:`Job` running the server.
        base_url: The OpenAI-compatible URL to point an
            :class:`LLMClient` at (e.g.
            ``http://localhost:8000/v1``).
    """

    job: Job
    base_url: str


async def wait_for_endpoint(
    base_url: str,
    *,
    timeout_s: float = 600.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Poll the ``/models`` endpoint until it returns HTTP 200.

    Args:
        base_url: The OpenAI-compatible base URL (with the ``/v1``
            suffix). The probe hits ``{base_url}/models``.
        timeout_s: How long to wait before giving up. Default
            10 minutes.
        poll_interval_s: Seconds between probes.

    Raises:
        TimeoutError: If the endpoint isn't healthy in time.
        ImportError: When ``httpx`` is unavailable.
    """
    import httpx  # already a core dep via litellm

    deadline = asyncio.get_event_loop().time() + timeout_s
    probe_url = f"{base_url.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                response = await client.get(probe_url)
                if response.status_code == 200:
                    return
            except httpx.RequestError, httpx.HTTPStatusError:
                pass
            if asyncio.get_event_loop().time() > deadline:
                err = f"serving endpoint {probe_url!r} not ready after {timeout_s:.0f}s"
                raise TimeoutError(err)
            await asyncio.sleep(poll_interval_s)


@contextlib.asynccontextmanager
async def serving_endpoint(
    backend: Backend,
    task: Task,
    *,
    base_url: str,
    wait_timeout_s: float = 600.0,
    cleanup: bool = True,
) -> AsyncGenerator[ServingEndpoint]:
    """Launch ``task`` on ``backend``, wait for ``base_url`` to respond.

    The context manager yields a :class:`ServingEndpoint` while
    the server is running. On exit, the job is cancelled (and,
    when ``cleanup=True``, the backend's ``cleanup`` runs).

    Args:
        backend: Any :class:`Backend` implementation.
        task: A :class:`Task` typically built by
            :func:`build_vllm_task` / :func:`build_tgi_task` /
            :func:`build_sglang_task`.
        base_url: The OpenAI-compatible URL the launched server
            will respond on. Forge does NOT guess this from the
            task — the caller knows the port they configured.
        wait_timeout_s: How long to wait for readiness.
        cleanup: When ``True``, call ``backend.cleanup`` on exit.

    Yields:
        A :class:`ServingEndpoint` carrying the job handle and
        the verified base URL.
    """
    job = await backend.submit(task)
    try:
        await wait_for_endpoint(base_url, timeout_s=wait_timeout_s)
        yield ServingEndpoint(job=job, base_url=base_url)
    finally:
        with contextlib.suppress(Exception):
            await backend.cancel(job)
        if cleanup:
            with contextlib.suppress(Exception):
                await backend.cleanup(job)
