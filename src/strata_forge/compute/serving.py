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

That wait is also the longest thing a caller does, so both
:func:`serving_endpoint` and :func:`wait_for_endpoint` accept an
``on_phase`` sink that receives short phrases ("Starting the
model server", "Loading the model onto the GPU (90s)", "model server
ready"). The sink takes a plain ``str``: :mod:`strata_forge.compute`
must not import :mod:`strata_forge.training`, so it is the caller —
typically a :mod:`strata_forge.pipelines` runner — that turns a
phrase into a ``ProgressEvent``.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strata_forge.compute.task import ResourceSpec, Task

# How much of a dead server's output to carry in the error. Enough for a traceback or a
# "command not found", not so much that it swamps whatever surfaces the message.
_FAILURE_LOG_CHARS = 2000


class ServingProcessError(RuntimeError):
    """The serving process exited before its endpoint became reachable.

    Distinct from :class:`TimeoutError`: the server did not merely take too long, it is gone —
    so the caller should report the process's own output rather than a duration.
    """


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from strata_forge.compute.backends.base import Backend
    from strata_forge.compute.job import Job

__all__ = [
    "ServingEndpoint",
    "ServingProcessError",
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


def _report(on_phase: Callable[[str], None] | None, message: str) -> None:
    """Hand one progress phrase to the caller's sink, if any.

    Exceptions are swallowed: ``on_phase`` is caller-supplied, and a broken sink (a progress
    file closed early, a full disk) must not take down a live serving job. Messages are
    fixed phrases that never interpolate ``base_url`` — a caller may legitimately pass a URL
    carrying credentials, and this text is meant to be surfaced to a human.
    """
    if on_phase is None:
        return
    with contextlib.suppress(Exception):
        on_phase(message)


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
    setup: str | None = None,
    python_executable: str | None = None,
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
        setup: Setup command run before the server starts.
            Defaults to installing vLLM with ``python_executable``'s
            own pip. Pass an empty string when the host already
            has vLLM installed.
        python_executable: Interpreter to serve with. When ``None``
            the task invokes the ``vllm`` console script and relies
            on ``PATH``, which is right for a task that will run on
            some other machine (an emitted spec, a cloud backend).
            Pass :data:`sys.executable` when the task runs on THIS
            host: a backend may execute it through a login shell,
            which re-sources the profile and drops the caller's
            virtualenv from ``PATH`` — the bare command then fails
            with "command not found". Naming the interpreter removes
            that dependency, and keeps the install and the server in
            the same environment.
    """
    # Deliberately NOT defaulting to sys.executable: this function also builds specs destined for
    # another machine, where an absolute path into the local virtualenv does not exist. The caller
    # knows where the task will run; this does not.
    if python_executable is None:
        cli_args: list[str] = ["vllm", "serve", model]
        install = "pip install 'vllm>=0.7'" if setup is None else setup
    else:
        quoted = _quote_args([python_executable])
        cli_args = [python_executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model]
        # Same reasoning for the install: `pip` is not on a login shell's PATH either, and
        # installing with the wrong pip puts vLLM where the served interpreter cannot import it.
        install = f"{quoted} -m pip install 'vllm>=0.7'" if setup is None else setup

    cli_args.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]
    )
    if max_model_len is not None:
        cli_args.extend(["--max-model-len", str(max_model_len)])
    if dtype is not None:
        cli_args.extend(["--dtype", dtype])
    cli_args.extend(extra_args)

    return Task(
        name=name,
        run=_quote_args(cli_args),
        setup=install,
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
    is_alive: Callable[[], Awaitable[bool]] | None = None,
    on_phase: Callable[[str], None] | None = None,
    phase_interval_s: float = 30.0,
) -> None:
    """Poll the ``/models`` endpoint until it returns HTTP 200.

    Args:
        base_url: The OpenAI-compatible base URL (with the ``/v1``
            suffix). The probe hits ``{base_url}/models``.
        timeout_s: How long to wait before giving up. Default
            10 minutes.
        poll_interval_s: Seconds between probes.
        is_alive: Optional liveness probe for the serving process.
            When it reports the process is gone, the wait fails
            immediately with :class:`ServingProcessError` instead
            of running out the timeout.
        on_phase: Optional sink for short human-readable progress
            phrases. It is called from the poll loop, so it must
            not block; exceptions it raises are swallowed.
        phase_interval_s: Minimum seconds between two heartbeat
            phrases. Deliberately much coarser than
            ``poll_interval_s``: an orchestrator persisting these
            phrases has a finite budget per run, and one row per
            probe would exhaust it while a large model loads.

    Raises:
        TimeoutError: If the endpoint isn't healthy in time.
        ServingProcessError: If ``is_alive`` reports the serving
            process is gone before the endpoint answered.
        ImportError: When ``httpx`` is unavailable.
    """
    import httpx  # already a core dep via litellm

    loop = asyncio.get_event_loop()
    started = loop.time()
    deadline = started + timeout_s
    probe_url = f"{base_url.rstrip('/')}/models"
    next_phase_at = started  # the first heartbeat goes out before the first probe
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            now = loop.time()
            if now >= next_phase_at:
                _report(on_phase, f"Loading the model onto the GPU ({int(now - started)}s)")
                next_phase_at = now + phase_interval_s
            try:
                response = await client.get(probe_url)
                if response.status_code == 200:
                    return
            except httpx.RequestError, httpx.HTTPStatusError:
                pass
            # A server that has already exited is never going to answer. Without this the caller
            # waits out the entire timeout, turning a fast and legible failure — a bad command, a
            # missing weight, an unusable GPU — into a slow and opaque one.
            if is_alive is not None and not await is_alive():
                err = f"serving process exited before {probe_url!r} became ready"
                raise ServingProcessError(err)
            if loop.time() > deadline:
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
    on_phase: Callable[[str], None] | None = None,
    phase_interval_s: float = 30.0,
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
        on_phase: Optional sink for short human-readable progress
            phrases, forwarded to :func:`wait_for_endpoint`.
            Bringing a server up is the longest uninterruptible
            step most callers have; without a sink the whole of it
            is one silent ``await``. The sink must not block, and
            any exception it raises is swallowed.
        phase_interval_s: Minimum seconds between two readiness
            heartbeats. Forwarded to :func:`wait_for_endpoint`.

    Yields:
        A :class:`ServingEndpoint` carrying the job handle and
        the verified base URL.
    """
    # Submitting is not the same as serving: a backend may return the moment the process is
    # spawned, so report the two separately rather than letting one phrase cover both.
    _report(on_phase, "Starting the model server")
    job = await backend.submit(task)

    async def _alive() -> bool:
        # Anything but a live job means the server is gone. A status the backend cannot report is
        # treated as alive, so a flaky probe cannot abort a server that is merely slow to start.
        try:
            status = await backend.status(job)
        except Exception:  # a status probe must never decide the run's fate
            return True
        return status.state in {"pending", "running"}

    try:
        try:
            await wait_for_endpoint(
                base_url,
                timeout_s=wait_timeout_s,
                is_alive=_alive,
                on_phase=on_phase,
                phase_interval_s=phase_interval_s,
            )
        except ServingProcessError as exc:
            # The process's own output is the diagnosis — a bad command or an unloadable model
            # says so here. Without it the caller only learns that nothing answered.
            tail = ""
            with contextlib.suppress(Exception):
                tail = (await backend.logs(job))[-_FAILURE_LOG_CHARS:]
            raise ServingProcessError(f"{exc}\n\n{tail}".rstrip()) from exc
        _report(on_phase, "Model server ready")
        yield ServingEndpoint(job=job, base_url=base_url)
    finally:
        # Teardown can hang too (a cancel that waits on an unresponsive process), so it is a
        # reportable phase rather than another silent stretch.
        _report(on_phase, "Stopping the model server")
        with contextlib.suppress(Exception):
            await backend.cancel(job)
        if cleanup:
            with contextlib.suppress(Exception):
                await backend.cleanup(job)
