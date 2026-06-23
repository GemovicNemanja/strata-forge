"""Unit tests for `forge.compute.backends.skypilot.SkyPilotBackend`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.compute import Backend, SkyPilotBackend, Task

# ---------------------------------------------------------------------------
# Fake sky.api.sdk
# ---------------------------------------------------------------------------


@dataclass
class _FakeSkyJobRecord:
    job_id: int
    status: str


class _FakeSky:
    """Fake SkyPilot SDK with the methods our backend touches."""

    def __init__(self) -> None:
        self.launches: list[dict[str, Any]] = []
        self.cancellations: list[dict[str, Any]] = []
        self.downed: list[str] = []
        self.queues: dict[str, list[_FakeSkyJobRecord]] = {}
        self._next_job_id = 100

    def launch(self, **kwargs: Any) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        self.launches.append(kwargs)
        cluster = kwargs["cluster_name"]
        self.queues.setdefault(cluster, []).append(
            _FakeSkyJobRecord(job_id=job_id, status="PENDING")
        )
        return job_id

    def queue(self, *, cluster_name: str) -> list[_FakeSkyJobRecord]:
        return list(self.queues.get(cluster_name, []))

    def tail_logs(self, *, cluster_name: str, job_id: int, follow: bool) -> str:
        del cluster_name, follow
        return f"logs for job {job_id}: line one\nline two\nline three\n"

    def cancel(self, *, cluster_name: str, job_ids: list[int]) -> None:
        for job_id in job_ids:
            self.cancellations.append({"cluster": cluster_name, "job_id": job_id})

    def down(self, *, cluster_name: str) -> None:
        self.downed.append(cluster_name)


@pytest.fixture
def fake_sky() -> _FakeSky:
    return _FakeSky()


@pytest.fixture
def backend(fake_sky: _FakeSky) -> SkyPilotBackend:
    return SkyPilotBackend(client=fake_sky)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_name(self, fake_sky: _FakeSky) -> None:
        backend = SkyPilotBackend(client=fake_sky)
        assert backend.name == "skypilot"

    def test_custom_name(self, fake_sky: _FakeSky) -> None:
        assert SkyPilotBackend(client=fake_sky, name="my-sky").name == "my-sky"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SkyPilotBackend(client=_FakeSky(), name="")

    def test_extra_missing_raises_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        monkeypatch.setitem(sys.modules, "sky.api.sdk", None)
        backend = SkyPilotBackend()
        with pytest.raises(ImportError, match=r"\[compute\] extra"):
            asyncio.run(backend.submit(Task(name="t", run="echo")))

    def test_satisfies_backend_protocol(self, fake_sky: _FakeSky) -> None:
        backend = SkyPilotBackend(client=fake_sky)
        assert isinstance(backend, Backend)


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    async def test_launches_with_canonical_kwargs(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        task = Task(name="trainer", run="python train.py", setup="pip install torch")
        job = await backend.submit(task)
        assert job.backend == "skypilot"
        assert job.task_name == "trainer"
        # cluster_name is prefixed.
        assert job.metadata["cluster_name"].startswith("forge-trainer-")
        assert len(fake_sky.launches) == 1
        launch_kwargs = fake_sky.launches[0]
        assert launch_kwargs["run"] == "python train.py"
        assert launch_kwargs["setup"] == "pip install torch"
        assert launch_kwargs["cluster_name"] == job.metadata["cluster_name"]

    async def test_resources_forwarded(self, backend: SkyPilotBackend, fake_sky: _FakeSky) -> None:
        from forge.compute.task import ResourceSpec

        await backend.submit(
            Task(
                name="t",
                run="hi",
                resources=ResourceSpec(accelerators="A100:1", cpus=8),
            )
        )
        launch_kwargs = fake_sky.launches[0]
        assert launch_kwargs["resources"]["accelerators"] == "A100:1"
        assert launch_kwargs["resources"]["cpus"] == 8

    async def test_env_forwarded_as_envs_plural(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        await backend.submit(Task(name="t", run="hi", env={"FOO": "1", "BAR": "2"}))
        launch_kwargs = fake_sky.launches[0]
        assert launch_kwargs["envs"] == {"FOO": "1", "BAR": "2"}

    async def test_tuple_return_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Older SkyPilot returns a (job_id, cluster_name) tuple."""

        class _OlderSky:
            def launch(self, **kwargs: Any) -> tuple[int, str]:
                return (777, "old-cluster-abc")

        backend = SkyPilotBackend(client=_OlderSky())
        job = await backend.submit(Task(name="t", run="hi"))
        assert job.id == "777"
        assert job.metadata["cluster_name"] == "old-cluster-abc"


# ---------------------------------------------------------------------------
# status — state mapping
# ---------------------------------------------------------------------------


class TestStatus:
    async def test_pending_maps_to_pending(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        # Status starts as PENDING in the fake.
        status = await backend.status(job)
        assert status.state == "pending"

    async def test_running_maps_to_running(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        fake_sky.queues[job.metadata["cluster_name"]][0].status = "RUNNING"
        status = await backend.status(job)
        assert status.state == "running"

    async def test_succeeded_maps_to_succeeded(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        fake_sky.queues[job.metadata["cluster_name"]][0].status = "SUCCEEDED"
        status = await backend.status(job)
        assert status.state == "succeeded"

    async def test_failed_setup_maps_to_failed(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        fake_sky.queues[job.metadata["cluster_name"]][0].status = "FAILED_SETUP"
        status = await backend.status(job)
        assert status.state == "failed"

    async def test_cancelled(self, backend: SkyPilotBackend, fake_sky: _FakeSky) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        fake_sky.queues[job.metadata["cluster_name"]][0].status = "CANCELLED"
        status = await backend.status(job)
        assert status.state == "cancelled"

    async def test_unknown_state_falls_back_to_running(
        self, backend: SkyPilotBackend, fake_sky: _FakeSky
    ) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        fake_sky.queues[job.metadata["cluster_name"]][0].status = "WHO_KNOWS"
        status = await backend.status(job)
        # Unknown maps to running so callers don't crash on new SkyPilot states.
        assert status.state == "running"

    async def test_missing_job_in_queue(self, backend: SkyPilotBackend, fake_sky: _FakeSky) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        # Remove from queue.
        fake_sky.queues[job.metadata["cluster_name"]] = []
        status = await backend.status(job)
        assert status.state == "failed"
        assert "not found" in status.message


# ---------------------------------------------------------------------------
# logs / cancel / cleanup
# ---------------------------------------------------------------------------


class TestLogs:
    async def test_full_logs(self, backend: SkyPilotBackend, fake_sky: _FakeSky) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        logs = await backend.logs(job)
        assert "line one" in logs

    async def test_tail(self, backend: SkyPilotBackend) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        tail = await backend.logs(job, tail=1)
        # Only the last line of the fake's three-line response.
        assert "line three" in tail
        assert "line one" not in tail

    async def test_invalid_tail(self, backend: SkyPilotBackend) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        with pytest.raises(ValueError, match="tail"):
            await backend.logs(job, tail=0)


class TestCancel:
    async def test_cancels_job(self, backend: SkyPilotBackend, fake_sky: _FakeSky) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        await backend.cancel(job)
        assert len(fake_sky.cancellations) == 1
        assert fake_sky.cancellations[0]["job_id"] == int(job.id)


class TestCleanup:
    async def test_downs_cluster(self, backend: SkyPilotBackend, fake_sky: _FakeSky) -> None:
        job = await backend.submit(Task(name="t", run="hi"))
        await backend.cleanup(job)
        assert fake_sky.downed == [job.metadata["cluster_name"]]


class TestJobOwnership:
    async def test_wrong_backend_rejected(self, backend: SkyPilotBackend) -> None:
        from forge.compute.job import Job

        bogus = Job(
            id="1",
            backend="not-sky",
            task_name="t",
            metadata={"cluster_name": "x"},
        )
        with pytest.raises(ValueError, match="backend"):
            await backend.status(bogus)

    async def test_missing_cluster_metadata(self, backend: SkyPilotBackend) -> None:
        from forge.compute.job import Job

        bogus = Job(id="1", backend="skypilot", task_name="t")
        with pytest.raises(ValueError, match="cluster_name"):
            await backend.status(bogus)


# ---------------------------------------------------------------------------
# State map covers documented inputs
# ---------------------------------------------------------------------------


class TestStateMap:
    @pytest.mark.parametrize(
        ("sky_state", "expected"),
        [
            ("INIT", "pending"),
            ("SETTING_UP", "pending"),
            ("PENDING", "pending"),
            ("RUNNING", "running"),
            ("SUCCEEDED", "succeeded"),
            ("FAILED", "failed"),
            ("FAILED_SETUP", "failed"),
            ("FAILED_PRECHECKS", "failed"),
            ("FAILED_NO_RESOURCE", "failed"),
            ("FAILED_CONTROLLER", "failed"),
            ("CANCELLED", "cancelled"),
            ("CANCELLING", "cancelled"),
        ],
    )
    def test_canonical_states(self, sky_state: str, expected: str) -> None:
        assert SkyPilotBackend._map_state(sky_state) == expected  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Settings-based client construction
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_lazy_imports_sky_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_module = types.ModuleType("sky.api.sdk")
        fake_module.launch = MagicMock(return_value=42)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sky.api.sdk", fake_module)

        backend = SkyPilotBackend()
        client = backend._get_client()  # type: ignore[attr-defined]
        assert client is fake_module
        # Cached on subsequent calls.
        assert backend._get_client() is fake_module  # type: ignore[attr-defined]


class TestReadFile:
    def _job(self) -> Any:
        from forge.compute.job import Job

        return Job(id="1", backend="skypilot", task_name="t", metadata={"cluster_name": "forge-x"})

    async def test_deferred_to_skypilot_path(self, backend: SkyPilotBackend) -> None:
        with pytest.raises(NotImplementedError, match="SkyPilot"):
            await backend.read_file(self._job(), "progress.jsonl")

    async def test_validates_path_before_deferral(self, backend: SkyPilotBackend) -> None:
        # The workdir-confinement contract is enforced uniformly, even for the stub.
        with pytest.raises(ValueError, match="within the job workdir"):
            await backend.read_file(self._job(), "../../etc/passwd")
