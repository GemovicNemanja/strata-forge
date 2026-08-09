"""Unit tests for `strata_forge.compute.serving`."""

from __future__ import annotations

import re
import shlex
import sys
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from strata_forge.compute import (
    Job,
    JobStatus,
    ResourceSpec,
    ServingEndpoint,
    Task,
    build_sglang_task,
    build_tgi_task,
    build_vllm_task,
    serving_endpoint,
    wait_for_endpoint,
)
from strata_forge.compute.serving import ServingProcessError


class TestBuildVLLMTask:
    def test_defaults(self) -> None:
        task = build_vllm_task("meta-llama/Llama-3.1-8B-Instruct")
        assert task.name == "vllm-serve"
        # Default stays PATH-based: this spec may be destined for another machine.
        assert "vllm serve" in task.run
        assert "meta-llama/Llama-3.1-8B-Instruct" in task.run
        assert "--port 8000" in task.run
        assert "--tensor-parallel-size 1" in task.run
        assert task.resources is not None
        assert task.resources.accelerators == "A100:1"
        assert "pip install" in (task.setup or "")

    def test_named_interpreter_bypasses_path_entirely(self) -> None:
        """A backend may run the task in a login shell whose PATH lacks the caller's virtualenv.

        A bare `vllm` console script then fails with "command not found" and, because nothing
        watches the serving job, the caller waits out its entire readiness timeout for a server
        that never existed. Naming the interpreter is what prevents that.
        """
        task = build_vllm_task("m", python_executable=sys.executable)
        assert task.run.startswith(shlex.quote(sys.executable))
        assert "vllm.entrypoints.openai.api_server" in task.run
        # Nothing may fall back to a console script being resolvable.
        assert not re.search(r"(^|\s)vllm\s+serve\b", task.run)
        # The model must survive the switch from positional to flag form.
        assert "--model m" in task.run
        # The install has to land in the SAME environment the server is started from.
        assert (task.setup or "").startswith(shlex.quote(sys.executable))
        assert "-m pip install" in (task.setup or "")

    def test_a_named_interpreter_keeps_every_other_argument(self) -> None:
        task = build_vllm_task(
            "m",
            python_executable="/opt/venv/bin/python",
            port=9001,
            tensor_parallel_size=8,
            max_model_len=4096,
            dtype="bfloat16",
            extra_args=["--gpu-memory-utilization", "0.9"],
        )
        assert task.run.startswith("/opt/venv/bin/python")
        assert "--port 9001" in task.run
        assert "--tensor-parallel-size 8" in task.run
        assert "--max-model-len 4096" in task.run
        assert "--dtype bfloat16" in task.run
        assert "0.9" in task.run

    def test_an_emitted_spec_carries_no_local_path(self) -> None:
        """The default builds specs for OTHER machines, where this venv's path does not exist."""
        task = build_vllm_task("m")
        assert sys.executable not in task.run
        assert sys.executable not in (task.setup or "")

    def test_custom_args(self) -> None:
        task = build_vllm_task(
            "my-model",
            port=8080,
            tensor_parallel_size=4,
            max_model_len=32_000,
            dtype="bfloat16",
            extra_args=["--gpu-memory-utilization", "0.92"],
        )
        assert "--port 8080" in task.run
        assert "--tensor-parallel-size 4" in task.run
        assert "--max-model-len 32000" in task.run
        assert "--dtype bfloat16" in task.run
        assert "0.92" in task.run

    def test_setup_can_be_disabled(self) -> None:
        task = build_vllm_task("m", setup="")
        assert task.setup == ""

    def test_resources_override(self) -> None:
        task = build_vllm_task("m", resources=ResourceSpec(accelerators="H100:8", cpus=32))
        assert task.resources is not None
        assert task.resources.accelerators == "H100:8"
        assert task.resources.cpus == 32


class TestBuildTGITask:
    def test_defaults(self) -> None:
        task = build_tgi_task("mistralai/Mistral-7B-v0.1")
        assert task.name == "tgi-serve"
        assert "docker run" in task.run
        assert "8080:80" in task.run
        assert "mistralai/Mistral-7B-v0.1" in task.run

    def test_caps_forwarded(self) -> None:
        task = build_tgi_task(
            "m",
            max_input_tokens=4096,
            max_total_tokens=8192,
            num_shard=2,
        )
        assert "--max-input-tokens 4096" in task.run
        assert "--max-total-tokens 8192" in task.run
        assert "--num-shard 2" in task.run

    def test_custom_image(self) -> None:
        task = build_tgi_task("m", image="ghcr.io/custom/tgi:1.0")
        assert "ghcr.io/custom/tgi:1.0" in task.run


class TestBuildSGLangTask:
    def test_defaults(self) -> None:
        task = build_sglang_task("Qwen/Qwen2-7B")
        assert task.name == "sglang-serve"
        assert "sglang.launch_server" in task.run
        assert "--model-path Qwen/Qwen2-7B" in task.run
        assert "--port 30000" in task.run
        assert "--tp-size 1" in task.run

    def test_extra_args(self) -> None:
        task = build_sglang_task("m", port=40000, tp_size=8, extra_args=["--disable-radix-cache"])
        assert "--port 40000" in task.run
        assert "--tp-size 8" in task.run
        assert "--disable-radix-cache" in task.run


# ---------------------------------------------------------------------------
# wait_for_endpoint
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        pass

    async def get(self, _url: str) -> Any:
        self.calls += 1
        if not self.responses:
            err = "no more responses queued"
            raise httpx.RequestError(err)
        next_response = self.responses.pop(0)
        if isinstance(next_response, BaseException):
            raise next_response
        return next_response


def _factory(client: _FakeAsyncClient) -> Any:
    def _make(**_kwargs: Any) -> _FakeAsyncClient:
        return client

    return _make


class TestWaitForEndpoint:
    async def test_immediate_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _OK:
            status_code = 200

        fake = _FakeAsyncClient([_OK()])
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        await wait_for_endpoint("http://localhost:8000/v1", timeout_s=5)
        assert fake.calls == 1

    async def test_retries_until_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _OK:
            status_code = 200

        class _NotYet:
            status_code = 503

        fake = _FakeAsyncClient(
            [
                httpx.RequestError("connection refused"),
                _NotYet(),
                _OK(),
            ]
        )
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        await wait_for_endpoint("http://localhost:8000/v1", timeout_s=10, poll_interval_s=0.01)
        assert fake.calls == 3

    async def test_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _NotYet:
            status_code = 503

        fake = _FakeAsyncClient([_NotYet()] * 50)
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        with pytest.raises(TimeoutError, match="not ready"):
            await wait_for_endpoint(
                "http://localhost:8000/v1", timeout_s=0.05, poll_interval_s=0.01
            )


class TestWaitForEndpointPhases:
    """The readiness wait is the longest thing a caller awaits; it must not be silent."""

    async def test_heartbeat_on_every_poll_when_uncapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _OK:
            status_code = 200

        class _NotYet:
            status_code = 503

        fake = _FakeAsyncClient([_NotYet(), _NotYet(), _OK()])
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        seen: list[str] = []
        await wait_for_endpoint(
            "http://localhost:8000/v1",
            timeout_s=10,
            poll_interval_s=0,
            on_phase=seen.append,
            phase_interval_s=0,  # every iteration is eligible
        )
        assert fake.calls == 3
        assert len(seen) == 3
        assert all(m.startswith("waiting for the model server (") for m in seen)

    async def test_heartbeat_is_throttled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The property that protects an orchestrator's per-run event budget: many probes,
        # one phrase. Without it a 30-minute wait at a 2s poll would emit ~900 rows.
        class _OK:
            status_code = 200

        class _NotYet:
            status_code = 503

        fake = _FakeAsyncClient([_NotYet()] * 20 + [_OK()])
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        seen: list[str] = []
        await wait_for_endpoint(
            "http://localhost:8000/v1",
            timeout_s=10,
            poll_interval_s=0,
            on_phase=seen.append,
            phase_interval_s=600,  # far longer than this test can run
        )
        assert fake.calls == 21
        assert len(seen) == 1

    async def test_never_echoes_the_probe_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # base_url is caller-supplied and may carry credentials; phrases are surfaced to a
        # human, so they must never interpolate it.
        class _OK:
            status_code = 200

        fake = _FakeAsyncClient([_OK()])
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        seen: list[str] = []
        await wait_for_endpoint(
            "http://user:hunter2@localhost:8000/v1",
            timeout_s=5,
            on_phase=seen.append,
            phase_interval_s=0,
        )
        assert seen
        assert not any("hunter2" in m or "localhost" in m for m in seen)

    async def test_broken_sink_cannot_break_the_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _OK:
            status_code = 200

        def _explode(_message: str) -> None:
            err = "sink is closed"
            raise RuntimeError(err)

        fake = _FakeAsyncClient([_OK()])
        monkeypatch.setattr(httpx, "AsyncClient", _factory(fake), raising=False)
        await wait_for_endpoint("http://localhost:8000/v1", timeout_s=5, on_phase=_explode)


# ---------------------------------------------------------------------------
# serving_endpoint
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self) -> None:
        self.submitted: list[Task] = []
        self.cancelled: list[Job] = []
        self.cleaned: list[Job] = []
        self.submit_should_raise: BaseException | None = None

    @property
    def name(self) -> str:
        return "fake"

    async def submit(self, task: Task) -> Job:
        if self.submit_should_raise is not None:
            raise self.submit_should_raise
        self.submitted.append(task)
        return Job(id="job-1", backend=self.name, task_name=task.name)

    async def status(self, job: Job) -> JobStatus:
        del job
        return JobStatus(state="running")

    async def logs(self, job: Job, *, tail: int | None = None) -> str:
        del job, tail
        return ""

    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str:
        del job, path, tail
        return ""

    async def cancel(self, job: Job) -> None:
        self.cancelled.append(job)

    async def cleanup(self, job: Job) -> None:
        self.cleaned.append(job)


@pytest.fixture
def backend() -> _FakeBackend:
    return _FakeBackend()


class TestServingEndpoint:
    async def test_yields_then_cleans_up(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(return_value=None),
        )
        task = build_vllm_task("m")
        async with serving_endpoint(backend, task, base_url="http://localhost:8000/v1") as endpoint:
            assert isinstance(endpoint, ServingEndpoint)
            assert endpoint.base_url == "http://localhost:8000/v1"
            assert endpoint.job.task_name == "vllm-serve"
        assert backend.submitted == [task]
        assert len(backend.cancelled) == 1
        assert len(backend.cleaned) == 1

    async def test_cleanup_disabled(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(return_value=None),
        )
        task = build_vllm_task("m")
        async with serving_endpoint(
            backend, task, base_url="http://localhost:8000/v1", cleanup=False
        ):
            pass
        assert len(backend.cancelled) == 1
        assert backend.cleaned == []

    async def test_cleans_up_when_body_raises(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(return_value=None),
        )
        task = build_vllm_task("m")
        err = RuntimeError("body crashed")
        with pytest.raises(RuntimeError, match="body crashed"):
            async with serving_endpoint(backend, task, base_url="http://localhost:8000/v1"):
                raise err
        # Even on body exception, the job got cancelled + cleaned.
        assert len(backend.cancelled) == 1
        assert len(backend.cleaned) == 1

    async def test_timeout_propagates(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(side_effect=TimeoutError("not ready")),
        )
        task = build_vllm_task("m")
        with pytest.raises(TimeoutError, match="not ready"):
            async with serving_endpoint(backend, task, base_url="http://localhost:8000/v1"):
                pass
        # Cleanup still ran.
        assert len(backend.cancelled) == 1

    async def test_reports_launch_ready_and_teardown(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(return_value=None),
        )
        seen: list[str] = []
        async with serving_endpoint(
            backend, build_vllm_task("m"), base_url="http://localhost:8000/v1", on_phase=seen.append
        ):
            # Submitting and serving are separate facts: "launched" must be visible before
            # readiness, or a server that never binds looks identical to one still starting.
            assert seen == ["launching the serving task", "model server ready"]
        assert seen[-1] == "stopping the serving task"

    async def test_reports_teardown_even_when_readiness_times_out(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(side_effect=TimeoutError("not ready")),
        )
        seen: list[str] = []
        with pytest.raises(TimeoutError, match="not ready"):
            async with serving_endpoint(
                backend,
                build_vllm_task("m"),
                base_url="http://localhost:8000/v1",
                on_phase=seen.append,
            ):
                pass
        assert seen == ["launching the serving task", "stopping the serving task"]

    async def test_forwards_the_sink_to_the_readiness_wait(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def _spy(_base_url: str, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("strata_forge.compute.serving.wait_for_endpoint", _spy)
        seen: list[str] = []
        async with serving_endpoint(
            backend,
            build_vllm_task("m"),
            base_url="http://localhost:8000/v1",
            on_phase=seen.append,
            phase_interval_s=7.5,
        ):
            pass
        # The blackout is INSIDE the wait, so the sink has to reach it — reporting only around
        # the context manager would leave the whole readiness window silent.
        assert captured["on_phase"] is not None
        assert captured["phase_interval_s"] == 7.5

    async def test_broken_sink_cannot_break_the_context_manager(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(return_value=None),
        )

        def _explode(_message: str) -> None:
            err = "sink is closed"
            raise RuntimeError(err)

        async with serving_endpoint(
            backend, build_vllm_task("m"), base_url="http://localhost:8000/v1", on_phase=_explode
        ):
            pass
        assert len(backend.cleaned) == 1

    async def test_swallows_cancel_and_cleanup_errors(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "strata_forge.compute.serving.wait_for_endpoint",
            AsyncMock(return_value=None),
        )

        async def _raise(*_args: Any, **_kwargs: Any) -> None:
            err = "boom"
            raise RuntimeError(err)

        backend.cancel = _raise  # type: ignore[method-assign]
        backend.cleanup = _raise  # type: ignore[method-assign]
        task = build_vllm_task("m")
        # Body completes cleanly despite cancel/cleanup raising.
        async with serving_endpoint(backend, task, base_url="http://localhost:8000/v1"):
            pass


class TestServingProcessDiesEarly:
    """A server that has already exited is never going to answer.

    Waiting out the readiness timeout turns a fast, legible failure — a bad command, an unloadable
    model — into a slow, opaque one, and discards the process's own account of what went wrong.
    """

    async def test_a_dead_job_fails_immediately_and_carries_its_output(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _dead(job: Job) -> JobStatus:
            del job
            return JobStatus(state="failed")

        async def _logs(job: Job, *, tail: int | None = None) -> str:
            del job, tail
            return "bash: line 1: vllm: command not found"

        backend.status = _dead  # type: ignore[method-assign]
        backend.logs = _logs  # type: ignore[method-assign]
        monkeypatch.setattr(
            "httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("refused"))
        )

        task = build_vllm_task("m")
        # A generous timeout: the point is that it does NOT wait for it.
        with pytest.raises(ServingProcessError) as caught:
            async with serving_endpoint(
                backend, task, base_url="http://localhost:8000/v1", wait_timeout_s=3600
            ):
                pass

        assert "exited before" in str(caught.value)
        # The diagnosis itself, not merely the fact that nothing answered.
        assert "command not found" in str(caught.value)
        assert len(backend.cancelled) == 1

    async def test_an_unreadable_status_does_not_abort_a_slow_start(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flaky status probe must not kill a server that is merely slow to come up."""

        async def _explode(job: Job) -> JobStatus:
            del job
            err = "transient"
            raise RuntimeError(err)

        backend.status = _explode  # type: ignore[method-assign]
        monkeypatch.setattr(
            "httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("refused"))
        )

        task = build_vllm_task("m")
        # Falls through to the ordinary timeout rather than reporting the process dead.
        with pytest.raises(TimeoutError):
            async with serving_endpoint(
                backend, task, base_url="http://localhost:8000/v1", wait_timeout_s=0.05
            ):
                pass
