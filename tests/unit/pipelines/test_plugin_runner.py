"""The plugin runner: every check before plugin code runs, the credential kept from the plugin,
and the output checked before it is pushed privately."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from pydantic import SecretStr

from strata_forge.pipelines import plugin_runner
from strata_forge.pipelines._common import RunError, RunSecrets
from strata_forge.plugins.host import SCORES_FILENAME, HostJob, run
from strata_forge.plugins.manifest import PluginManifestError
from strata_forge.plugins.pins import ALLOWED_PLUGINS_ENV, PluginPin
from tests.unit.plugins.conftest import MANIFEST, SHA

if TYPE_CHECKING:
    from pathlib import Path


def _arrow() -> tuple[Any, Any]:
    """pyarrow ships with the [hf] extra; the tests that write parquet skip without it."""
    return pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")


TOKEN = "hf_" + "Z" * 34


def _spec(pin: PluginPin, entry: str = "prm", **changes: Any) -> plugin_runner.PluginRunSpec:
    decl = next(e for e in MANIFEST["entry_points"] if e["name"] == entry)
    data: dict[str, Any] = {
        "run_id": "run1",
        "plugin": pin.model_dump(),
        "entry": decl,
        "config": {"mode": "fast"} if entry == "prm" else {"label": "L"},
        "inputs": [
            {"role": "steps", "repo_id": "acme/steps", "revision": SHA, "split": "train"}
            if entry == "prm"
            else {"role": "items", "repo_id": "acme/items", "revision": SHA, "split": "train"}
        ],
        "output_repo_id": "acme/out",
    }
    data.update(changes)
    return plugin_runner.PluginRunSpec.model_validate(data)


def _allow(monkeypatch: pytest.MonkeyPatch, *pins: PluginPin) -> None:
    monkeypatch.setenv(ALLOWED_PLUGINS_ENV, json.dumps([p.model_dump() for p in pins]))


def _steps_parquet(path: Path, n: int = 5) -> None:
    pa, pq = _arrow()
    pq.write_table(
        pa.table(
            {
                "item_id": [f"s{i // 2}" for i in range(n)],
                "step_index": [i % 2 for i in range(n)],
                "context": ["premises"] * n,
                "step": ["x" * (i + 1) for i in range(n)],
            }
        ),
        str(path),
    )


@pytest.mark.usefixtures("toy_site")
def test_admission_refuses_before_any_plugin_code_runs(
    monkeypatch: pytest.MonkeyPatch, pin: PluginPin
) -> None:
    monkeypatch.delenv(ALLOWED_PLUGINS_ENV, raising=False)
    with pytest.raises(RunError, match="not set"):
        plugin_runner.admit(_spec(pin))
    other = pin.model_copy(update={"version": "9.9.9"})
    _allow(monkeypatch, other)
    with pytest.raises(RunError, match="not an allowed plugin"):
        plugin_runner.admit(_spec(pin))
    _allow(monkeypatch, pin)
    with pytest.raises(RunError, match="interface"):
        plugin_runner.admit(_spec(pin, config={"extra": 1}))
    with pytest.raises(RunError, match="interface"):
        plugin_runner.admit(_spec(pin, inputs=[]))
    with pytest.raises(RunError, match="invalid output repo"):
        plugin_runner.admit(_spec(pin, output_repo_id="../x"))
    assert plugin_runner.admit(_spec(pin)) == {"mode": "fast", "scale": 1.0}


def test_an_uninstalled_build_is_refused(monkeypatch: pytest.MonkeyPatch, pin: PluginPin) -> None:
    _allow(monkeypatch, pin)
    with pytest.raises(RunError, match="is not installed"):
        plugin_runner.admit(_spec(pin))


def test_the_plugin_environment_holds_no_credential(tmp_path: Path) -> None:
    env = plugin_runner.plugin_environ(
        gpu=False,
        hf_home=tmp_path,
        environ={
            "PATH": "/bin",
            "HOME": "/home/u",
            "HF_TOKEN": "hf_x",
            "STRATA_RUN_CONFIG": "{}",
            "STRATA_SECRETS_FILE": "/run/s",
            "PYTHONPATH": "/evil",
            "CUDA_VISIBLE_DEVICES": "0",
            "AWS_SECRET_ACCESS_KEY": "x",
        },
    )
    assert env["PATH"] == "/bin"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HOME"] == str(tmp_path)
    for name in (
        "HF_TOKEN",
        "STRATA_RUN_CONFIG",
        "STRATA_SECRETS_FILE",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "AWS_SECRET_ACCESS_KEY",
    ):
        assert name not in env
    gpu = plugin_runner.plugin_environ(
        gpu=True, hf_home=tmp_path, environ={"CUDA_VISIBLE_DEVICES": "0", "CUDA_TOKEN": "x"}
    )
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0"
    assert "CUDA_TOKEN" not in gpu


@pytest.mark.usefixtures("toy_site")
def test_the_host_scores_every_step(tmp_path: Path, pin: PluginPin) -> None:
    steps = tmp_path / "steps.parquet"
    _steps_parquet(steps)
    out = tmp_path / "out"
    run(
        HostJob(
            pin=pin,
            entry_point="prm",
            type="step_scorer",
            config={"scale": 2.0},
            datasets={"steps": steps},
            models={},
            output_dir=out,
            batch_size=2,
        )
    )
    _, pq = _arrow()
    table = pq.read_table(str(out / SCORES_FILENAME)).to_pydict()
    assert table["item_id"] == ["s0", "s0", "s1", "s1", "s2"]
    assert table["step_index"] == [0, 1, 0, 1, 0]
    assert table["score"] == [0.2, 0.4, 0.6, 0.8, 1.0]


@pytest.mark.usefixtures("toy_site")
def test_a_scorer_returning_the_wrong_count_fails(tmp_path: Path, pin: PluginPin) -> None:
    steps = tmp_path / "steps.parquet"
    _steps_parquet(steps)
    job = HostJob(
        pin=pin,
        entry_point="bad",
        type="step_scorer",
        config={},
        datasets={"steps": steps},
        models={},
        output_dir=tmp_path / "out",
    )
    with pytest.raises(PluginManifestError, match="1 scores for 5 steps"):
        run(job)


def _admit_with(config: dict[str, Any]) -> Any:
    def admit(spec: plugin_runner.PluginRunSpec) -> dict[str, Any]:
        return config

    return admit


class _FakeHub:
    pushed: ClassVar[list[tuple[str, str]]] = []
    private: ClassVar[bool] = True

    def __init__(self, *, token: Any) -> None:
        self.token = token

    async def create_repo(self, repo_id: str, **_: Any) -> None:
        return None

    async def is_private(self, repo_id: str, **_: Any) -> bool:
        return self.private

    async def upload_folder(self, folder: Any, repo_id: str, *, repo_type: str, **_: Any) -> str:
        _FakeHub.pushed.append((repo_id, repo_type))
        return "ok"


def _wire(monkeypatch: pytest.MonkeyPatch, toy_site: Path, tmp_path: Path) -> None:
    async def fake_inputs(spec: Any, secrets: Any, workdir: Path) -> Any:
        role = spec.inputs[0].role
        dest = workdir / "inputs" / f"{role}.parquet"
        dest.parent.mkdir(parents=True)
        if role == "steps":
            _steps_parquet(dest)
        else:
            pa, pq = _arrow()
            pq.write_table(pa.table({"text": ["a", "b"]}), str(dest))
        return {role: dest}, {}, {role: 5 if role == "steps" else 2}

    real_environ = plugin_runner.plugin_environ

    def environ_with_toy(*, gpu: bool, hf_home: Path, environ: Any = None) -> dict[str, str]:
        env = real_environ(gpu=gpu, hf_home=hf_home, environ=environ)
        # The test's toy distribution lives on a temporary path the real allow-list strips.
        env["PYTHONPATH"] = str(toy_site)
        return env

    monkeypatch.setattr(plugin_runner, "_download_inputs", fake_inputs)
    monkeypatch.setattr(plugin_runner, "plugin_environ", environ_with_toy)
    monkeypatch.setattr(plugin_runner, "HFHubClient", _FakeHub)

    def results(run_id: str | None, *, name: str) -> Path:
        return tmp_path / "results"

    monkeypatch.setattr(plugin_runner, "results_dir", results)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    _FakeHub.pushed = []
    _FakeHub.private = True


def test_a_step_scoring_run_end_to_end(
    monkeypatch: pytest.MonkeyPatch, toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, toy_site, tmp_path)
    secrets = RunSecrets(hf_token=SecretStr(TOKEN))
    out = asyncio.run(plugin_runner._execute(_spec(pin), secrets, None))  # pyright: ignore[reportPrivateUsage]
    assert out == "acme/out"
    assert _FakeHub.pushed == [("acme/out", "dataset")]
    manifest = json.loads((tmp_path / "results" / "manifest.json").read_text())
    assert manifest["kind"] == "plugin"
    assert manifest["plugin"]["source"]["commit"] == SHA
    assert manifest["output"] == {"path": SCORES_FILENAME, "rows": 5}


def test_the_plugin_process_never_sees_the_credential(
    monkeypatch: pytest.MonkeyPatch, toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, toy_site, tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", "{}")
    probe = dict(next(e for e in MANIFEST["entry_points"] if e["name"] == "job"), name="probe")
    spec = _spec(pin, entry="job").model_copy(
        update={"entry": plugin_runner.EntryPointDecl.model_validate(probe)}
    )
    # The probe is not in the toy manifest's allowed set of names; admission is exercised above.
    monkeypatch.setattr(plugin_runner, "admit", _admit_with({"label": "L"}))
    asyncio.run(plugin_runner._execute(spec, RunSecrets(hf_token=SecretStr(TOKEN)), None))  # pyright: ignore[reportPrivateUsage]
    assert (tmp_path / "results" / "env.txt").read_text() == ""


def test_a_public_output_repository_is_never_written(
    monkeypatch: pytest.MonkeyPatch, toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, toy_site, tmp_path)
    _FakeHub.private = False
    with pytest.raises(RunError, match="public"):
        asyncio.run(plugin_runner._execute(_spec(pin), RunSecrets(hf_token=SecretStr(TOKEN)), None))  # pyright: ignore[reportPrivateUsage]
    assert _FakeHub.pushed == []


def test_a_failing_plugin_reports_its_exit_scrubbed(
    monkeypatch: pytest.MonkeyPatch, toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, toy_site, tmp_path)
    spec = _spec(pin, entry="prm").model_copy(
        update={
            "entry": plugin_runner.EntryPointDecl.model_validate(
                dict(next(e for e in MANIFEST["entry_points"] if e["name"] == "prm"), name="bad")
            )
        }
    )
    monkeypatch.setattr(plugin_runner, "admit", _admit_with({}))
    with pytest.raises(RunError, match="exited with code 1"):
        asyncio.run(plugin_runner._execute(spec, RunSecrets(hf_token=SecretStr(TOKEN)), None))  # pyright: ignore[reportPrivateUsage]
    assert _FakeHub.pushed == []
