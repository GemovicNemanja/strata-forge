"""Unit tests for `forge.compute.serving`."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from forge.compute import (
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


class TestBuildVLLMTask:
    def test_defaults(self) -> None:
        task = build_vllm_task("meta-llama/Llama-3.1-8B-Instruct")
        assert task.name == "vllm-serve"
        assert "vllm serve" in task.run
        assert "meta-llama/Llama-3.1-8B-Instruct" in task.run
        assert "--port 8000" in task.run
        assert "--tensor-parallel-size 1" in task.run
        assert task.resources is not None
        assert task.resources.accelerators == "A100:1"
        assert "pip install" in (task.setup or "")

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
            "forge.compute.serving.wait_for_endpoint",
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
            "forge.compute.serving.wait_for_endpoint",
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
            "forge.compute.serving.wait_for_endpoint",
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
            "forge.compute.serving.wait_for_endpoint",
            AsyncMock(side_effect=TimeoutError("not ready")),
        )
        task = build_vllm_task("m")
        with pytest.raises(TimeoutError, match="not ready"):
            async with serving_endpoint(backend, task, base_url="http://localhost:8000/v1"):
                pass
        # Cleanup still ran.
        assert len(backend.cancelled) == 1

    async def test_swallows_cancel_and_cleanup_errors(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "forge.compute.serving.wait_for_endpoint",
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
