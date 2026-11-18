"""The plugin runner: every check before plugin code runs, the credential kept from the plugin,
the plugin gone before its output is copied, checked and pushed privately."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import SecretStr

from strata_forge.pipelines import plugin_runner
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV, RunError, RunSecrets
from strata_forge.plugins.environment import PluginEnvironment
from strata_forge.plugins.host import SCORES_FILENAME, HostJob, run
from strata_forge.plugins.manifest import EntryPointDecl, PluginManifestError
from strata_forge.plugins.pins import ALLOWED_PLUGINS_ENV, AllowedPlugin, PluginPin, RunnerAllowance
from strata_forge.plugins.sandbox import Sandbox
from tests.unit.plugins.conftest import MANIFEST, SHA, build_toy

TOKEN = "hf_" + "Z" * 34


def _decl(name: str, base: str = "prm", **changes: Any) -> dict[str, Any]:
    decl = copy.deepcopy(next(e for e in MANIFEST["entry_points"] if e["name"] == base))
    decl["name"] = name
    decl.update(changes)
    return decl


#: The entry points the orchestrator hands the runner: the manifest's, plus toy ones the tests use.
ENTRIES = [
    *MANIFEST["entry_points"],
    _decl("bad"),
    _decl("escape"),
    _decl("probe", base="job"),
]


def _spec(pin: PluginPin, entry: str = "prm", **changes: Any) -> plugin_runner.PluginRunSpec:
    scorer = entry in {"prm", "bad", "escape"}
    data: dict[str, Any] = {
        "run_id": "run1",
        "plugin": pin.model_dump(),
        "entry_point": entry,
        "config": {"mode": "fast"} if scorer else {"label": "L"},
        "inputs": [
            {"role": "steps", "repo_id": "acme/steps", "revision": SHA, "split": "train"}
            if scorer
            else {"role": "items", "repo_id": "acme/items", "revision": SHA, "split": "train"}
        ],
        "output_repo_id": "acme/out",
    }
    data.update(changes)
    return plugin_runner.PluginRunSpec.model_validate(data)


def _allow(
    monkeypatch: pytest.MonkeyPatch, pin: PluginPin, *, entry_points: tuple[str, ...] | None = None
) -> None:
    allowance = RunnerAllowance(
        plugin=AllowedPlugin(id="toy", label="Toy", pin=pin, entry_points=entry_points),
        entry_points=tuple(EntryPointDecl.model_validate(e) for e in ENTRIES),
    )
    monkeypatch.setenv(ALLOWED_PLUGINS_ENV, json.dumps([allowance.model_dump(mode="json")]))


def _steps_parquet(path: Path, n: int = 5, *, step_type: bool = False) -> None:
    columns: dict[str, Any] = {
        "item_id": [f"s{i // 2}" for i in range(n)],
        "step_index": [i % 2 for i in range(n)],
        "context": ["premises"] * n,
        "step": ["x" * (i + 1) for i in range(n)],
    }
    if step_type:
        columns["step_type"] = ["claim" if i != 1 else "skip" for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), str(path))


def test_admission_refuses_before_any_plugin_code_runs(
    monkeypatch: pytest.MonkeyPatch, pin: PluginPin
) -> None:
    monkeypatch.delenv(ALLOWED_PLUGINS_ENV, raising=False)
    with pytest.raises(RunError, match="not set"):
        plugin_runner.admit(_spec(pin))
    _allow(monkeypatch, pin.model_copy(update={"version": "9.9.9"}))
    with pytest.raises(RunError, match="not an allowed plugin"):
        plugin_runner.admit(_spec(pin))
    _allow(monkeypatch, pin)
    with pytest.raises(RunError, match="interface"):
        plugin_runner.admit(_spec(pin, config={"extra": 1}))
    with pytest.raises(RunError, match="interface"):
        plugin_runner.admit(_spec(pin, inputs=[]))
    with pytest.raises(RunError, match="invalid output repo"):
        plugin_runner.admit(_spec(pin, output_repo_id="../x"))
    with pytest.raises(RunError, match="does not allow the entry point 'nope'"):
        plugin_runner.admit(_spec(pin, entry="nope"))
    admitted = plugin_runner.admit(_spec(pin))
    assert admitted.config == {"mode": "fast", "scale": 1.0}
    assert admitted.entry.score_kind == "probability"


def test_the_operator_s_entry_point_limit_is_enforced_on_the_target(
    monkeypatch: pytest.MonkeyPatch, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin, entry_points=("job",))
    with pytest.raises(RunError, match="does not allow the entry point 'prm'"):
        plugin_runner.admit(_spec(pin))
    assert plugin_runner.admit(_spec(pin, entry="job")).entry.type == "runner"


def test_the_spec_cannot_carry_an_interface_of_its_own(pin: PluginPin) -> None:
    data = {**_spec(pin).model_dump(mode="json"), "entry": MANIFEST["entry_points"][0]}
    with pytest.raises(ValueError, match="Extra inputs"):
        plugin_runner.PluginRunSpec.model_validate(data)


def test_diagnostic_aggregations_are_opt_in_and_scorer_only(
    monkeypatch: pytest.MonkeyPatch, pin: PluginPin
) -> None:
    assert _spec(pin).diagnostic_aggregations == ()
    with pytest.raises(ValueError, match="repeats"):
        _spec(pin, diagnostic_aggregations=["min", "min"])
    with pytest.raises(ValueError, match="validation error"):
        _spec(pin, diagnostic_aggregations=["median"])
    _allow(monkeypatch, pin)
    with pytest.raises(RunError, match="step_scorer only"):
        plugin_runner.admit(_spec(pin, entry="job", diagnostic_aggregations=["min"]))


def test_the_plugin_environment_holds_no_credential(tmp_path: Path) -> None:
    env = plugin_runner.plugin_environ(
        gpu=False,
        home=tmp_path / "home",
        hf_home=tmp_path / "hf",
        environ={
            "PATH": "/bin",
            "HOME": "/home/u",
            "HF_TOKEN": "hf_x",
            "STRATA_RUN_CONFIG": "{}",
            "FORGE_SECRETS_FILE": "/run/s",
            "PYTHONPATH": "/evil",
            "CUDA_VISIBLE_DEVICES": "0",
            "AWS_SECRET_ACCESS_KEY": "x",
        },
    )
    assert env["PATH"] == "/bin"
    assert env["HOME"] == str(tmp_path / "home")  # never the runner's
    assert env["HF_HUB_OFFLINE"] == "1"
    for name in (
        "HF_TOKEN",
        "STRATA_RUN_CONFIG",
        "FORGE_SECRETS_FILE",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "AWS_SECRET_ACCESS_KEY",
    ):
        assert name not in env


def _host_job(
    pin: PluginPin, site: Path, entry: str, steps: Path, out: Path, **changes: Any
) -> HostJob:
    decl = _decl(entry)
    decl.update(changes.pop("decl", {}))
    return HostJob(
        pin=pin,
        plugin_paths=(site,),
        entry=EntryPointDecl.model_validate(decl),
        config=changes.pop("config", {}),
        datasets={"steps": steps},
        models={},
        output_dir=out,
        **changes,
    )


def test_the_host_scores_every_step_and_carries_the_input_columns(
    toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    steps = tmp_path / "steps.parquet"
    _steps_parquet(steps, step_type=True)
    out = tmp_path / "out"
    run(_host_job(pin, toy_site, "prm", steps, out, config={"scale": 2.0}, batch_size=2))
    table = pq.read_table(str(out / SCORES_FILENAME)).to_pydict()
    assert list(table) == ["item_id", "step_index", "step_type", "score", "reason"]
    assert table["step_type"] == ["claim", "skip", "claim", "claim", "claim"]
    assert table["score"] == [0.2, None, 0.6, 0.8, 1.0]
    assert table["reason"] == [None, "not_scored_type", None, None, None]


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("bad", "1 results for 5 steps"),
        ("over", "probability"),
        ("nan", "non-finite"),
        ("reason", "undeclared reason"),
        ("escape", "not one of toy-prm's files"),
    ],
)
def test_a_scorer_outside_the_contract_fails(
    toy_site: Path, tmp_path: Path, pin: PluginPin, entry: str, match: str
) -> None:
    steps = tmp_path / "steps.parquet"
    _steps_parquet(steps)
    with pytest.raises(PluginManifestError, match=match):
        run(_host_job(pin, toy_site, entry, steps, tmp_path / "out"))
    assert not (tmp_path / "out" / SCORES_FILENAME).exists()


def test_the_host_refuses_a_build_that_changed_after_install(
    toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    steps = tmp_path / "steps.parquet"
    _steps_parquet(steps)
    (toy_site / "toy_prm" / "__init__.py").write_text("raise SystemExit('planted')\n")
    with pytest.raises(PluginManifestError, match="RECORD hash"):
        run(_host_job(pin, toy_site, "prm", steps, tmp_path / "out"))


def test_the_host_does_not_trust_a_module_already_in_sys_modules(
    toy_site: Path, tmp_path: Path, pin: PluginPin, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = tmp_path / "steps.parquet"
    _steps_parquet(steps)
    planted = types.ModuleType("toy_prm")
    planted.__file__ = str(tmp_path / "elsewhere.py")
    monkeypatch.setitem(sys.modules, "toy_prm", planted)
    with pytest.raises(PluginManifestError, match="not one of toy-prm's files"):
        run(_host_job(pin, toy_site, "prm", steps, tmp_path / "out"))


def test_a_structured_context_and_declared_columns_pass_through_unchanged(
    toy_site: Path, tmp_path: Path, pin: PluginPin
) -> None:
    steps = tmp_path / "steps.parquet"
    context = [
        [{"pid": "p1", "text": "a"}, {"pid": "p2", "text": "b"}],
        [{"pid": "p1", "text": "c"}],
    ]
    pq.write_table(
        pa.table(
            {
                "item_id": ["i", "i"],
                "step_index": pa.array([0, 1], type=pa.int32()),
                "context": context,
                "step": ["s", "t"],
                "source": pa.array([7, 8], type=pa.int16()),
            }
        ),
        str(steps),
    )
    columns = ["item_id", "step_index", "context", "step", "source"]
    decl = {
        "inputs": [{"role": "steps", "required_columns": columns}],
        "passthrough_columns": ["source"],
        "score_kind": "unconstrained",
    }
    out = tmp_path / "out"
    run(_host_job(pin, toy_site, "contexts", steps, out, decl=decl))
    table = pq.read_table(str(out / SCORES_FILENAME))
    assert table.column_names == ["item_id", "step_index", "source", "score", "reason"]
    assert table.column("score").to_pylist() == [2.0, 1.0]  # the segments, never joined
    assert table.schema.field("source").type == pa.int16()


# ------------------------------------------------------------------- the whole run, end to end


@pytest.fixture(scope="session")
def plugin_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A plugin directory as the runner builds it (``pip --target``): the toy build alone. The
    host runs on the engine's own interpreter with ``-I`` and appends it to its path."""
    root = tmp_path_factory.mktemp("plugin-env")
    build_toy(root / "site")
    return root


class _FakeHub:
    pushed: ClassVar[list[tuple[str, str, list[str]]]] = []
    last: ClassVar[dict[str, bytes]] = {}
    private: ClassVar[bool] = True

    def __init__(self, *, token: Any) -> None:
        self.token = token

    async def create_repo(self, repo_id: str, **_: Any) -> None:
        return None

    async def is_private(self, repo_id: str, **_: Any) -> bool:
        return self.private

    commit: ClassVar[str] = "c" * 40

    async def upload_files(
        self, files: dict[str, Path], repo_id: str, *, repo_type: str, **_: Any
    ) -> str:
        _FakeHub.pushed.append((repo_id, repo_type, sorted(files)))
        _FakeHub.last = {name: path.read_bytes() for name, path in files.items()}
        return self.commit


def _wire(monkeypatch: pytest.MonkeyPatch, plugin_env: Path) -> None:
    async def environment(*_: Any) -> PluginEnvironment:
        return PluginEnvironment(
            key="k",
            root=plugin_env,
            site=plugin_env / "site",
            packages=({"name": "toy-prm", "version": "0.1.0"},),
            requirements=("toy-prm==0.1.0 @ git+x",),
            reused=False,
        )

    async def inputs(spec: Any, entry: Any, secrets: Any, folder: Path) -> Any:
        role = spec.inputs[0].role
        dest = folder / "datasets" / f"{role}.parquet"
        if role == "steps":
            _steps_parquet(dest, step_type=True)
        else:
            dest.parent.mkdir(parents=True)
            pq.write_table(pa.table({"text": ["a", "b"]}), str(dest))
        return {role: dest}, {}, {role: 5 if role == "steps" else 2}

    async def sandbox() -> Sandbox:
        return Sandbox("bwrap", None)

    def no_bwrap(_: Sandbox, inner: list[str], *, env: dict[str, str], **__: Any) -> list[str]:
        # bubblewrap's --clearenv and --setenv, without its namespaces (not on this machine).
        return ["/usr/bin/env", "-i", *(f"{k}={v}" for k, v in env.items()), *inner]

    monkeypatch.setattr(plugin_runner, "_prepare_environment", environment)
    monkeypatch.setattr(plugin_runner, "_download_inputs", inputs)
    monkeypatch.setattr(plugin_runner, "choose_sandbox", sandbox)
    monkeypatch.setattr(plugin_runner, "sandbox_command", no_bwrap)
    monkeypatch.setattr(plugin_runner, "HFHubClient", _FakeHub)
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    _FakeHub.pushed = []
    _FakeHub.last = {}
    _FakeHub.private = True
    _FakeHub.commit = "c" * 40


def _execute(spec: plugin_runner.PluginRunSpec, *, from_environment: bool = False) -> str:
    secrets = RunSecrets(hf_token=SecretStr(TOKEN), from_environment=from_environment)
    return asyncio.run(plugin_runner._execute(spec, secrets, None))  # pyright: ignore[reportPrivateUsage]


def test_a_step_scoring_run_end_to_end(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    assert _execute(_spec(pin)) == f"acme/out@{_FakeHub.commit}"
    ((repo, kind, files),) = _FakeHub.pushed
    assert (repo, kind) == ("acme/out", "dataset")
    assert files == ["manifest.json", SCORES_FILENAME]
    manifest = json.loads(_FakeHub.last["manifest.json"])
    assert manifest["kind"] == "plugin"
    assert manifest["plugin"]["repo"] == "acme/toy-prm"
    assert manifest["plugin"]["commit"] == SHA
    assert manifest["plugin"]["score_kind"] == "probability"
    assert manifest["inputs"][0]["commit"] == SHA
    assert manifest["environment"]["packages"] == [{"name": "toy-prm", "version": "0.1.0"}]
    assert manifest["sandbox"]["isolation"] == "bwrap"
    assert manifest["output"]["scoring"] == {
        "score_kind": "probability",
        "rows": 5,
        "scored": 4,
        "unscored": {"unparseable": 0, "not_scored_type": 1},
    }
    assert set(manifest["output"]["files"]) == {SCORES_FILENAME}
    assert "diagnostics" not in manifest
    # The receipt in the job's workdir names the commit the push made and the manifest's digest.
    receipt = json.loads(Path("receipt.json").read_text())
    assert receipt["kind"] == "plugin"
    assert receipt["output"] == {"repo_id": "acme/out", "commit_sha": _FakeHub.commit}
    assert receipt["manifest_sha256"] == hashlib.sha256(_FakeHub.last["manifest.json"]).hexdigest()


def test_opt_in_aggregations_go_to_a_separate_diagnostic_file(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    _execute(_spec(pin, diagnostic_aggregations=["min", "last"]))
    ((_, _, files),) = _FakeHub.pushed
    assert plugin_runner.DIAGNOSTICS_FILENAME in files
    table = pq.read_table(pa.BufferReader(_FakeHub.last[plugin_runner.DIAGNOSTICS_FILENAME]))
    assert table.schema.metadata == {b"strata.diagnostic": b"true"}
    assert len(table) == 6
    scores = pq.read_table(pa.BufferReader(_FakeHub.last[SCORES_FILENAME]))
    assert "aggregation" not in scores.column_names
    manifest = json.loads(_FakeHub.last["manifest.json"])
    assert manifest["diagnostics"]["diagnostic"] is True


def test_the_scores_are_parsed_in_a_process_of_their_own(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    from strata_forge.plugins import checker, output

    def never(*_: Any) -> Any:
        msg = "the token-holding process parsed the plugin's scores"
        raise AssertionError(msg)

    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    commands: list[list[str]] = []
    wired = plugin_runner.sandbox_command

    def recording(sandbox: Sandbox, inner: list[str], **kwargs: Any) -> list[str]:
        commands.append(list(inner))
        return wired(sandbox, inner, **kwargs)

    monkeypatch.setattr(plugin_runner, "sandbox_command", recording)
    monkeypatch.setattr(output, "check_scores_file", never)
    monkeypatch.setattr(checker, "check_scores_file", never)
    _execute(_spec(pin, diagnostic_aggregations=["min"]))
    assert [c[3] for c in commands] == ["strata_forge.plugins.host", "strata_forge.plugins.checker"]
    assert plugin_runner.DIAGNOSTICS_FILENAME in _FakeHub.pushed[0][2]


@pytest.mark.parametrize(
    ("asked", "returned"), [((), ["min"]), (("min", "last"), ["min"]), (("min",), None)]
)
def test_diagnostics_other_than_those_asked_for_push_nothing(
    monkeypatch: pytest.MonkeyPatch,
    plugin_env: Path,
    pin: PluginPin,
    asked: tuple[str, ...],
    returned: list[str] | None,
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    real = plugin_runner._check_record  # pyright: ignore[reportPrivateUsage]

    def tampered(*args: Any) -> dict[str, Any]:
        record = {k: v for k, v in real(*args).items() if k != "diagnostics"}
        if returned is not None:
            record["diagnostics"] = {
                "path": plugin_runner.DIAGNOSTICS_FILENAME,
                "diagnostic": True,
                "aggregations": returned,
            }
        return record

    monkeypatch.setattr(plugin_runner, "_check_record", tampered)
    with pytest.raises(RunError, match="not asked for"):
        _execute(_spec(pin, diagnostic_aggregations=list(asked)))
    assert _FakeHub.pushed == []


def test_the_plugin_process_never_sees_a_credential(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    monkeypatch.setenv("STRATA_RUN_CONFIG", "{}")
    _execute(_spec(pin, entry="probe"))
    assert _FakeHub.last["env.txt"] == b""


def test_a_token_delivered_in_the_environment_is_refused(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    with pytest.raises(RunError, match="only from the secrets file"):
        _execute(_spec(pin), from_environment=True)
    monkeypatch.setattr(plugin_runner._StartingEnvironment, "token_present", True)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RunError, match="only from the secrets file"):
        _execute(_spec(pin))
    assert _FakeHub.pushed == []


def test_a_public_output_repository_is_never_written(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    _FakeHub.private = False
    with pytest.raises(RunError, match="public"):
        _execute(_spec(pin))
    assert _FakeHub.pushed == []


@pytest.mark.parametrize("entry", ["bad", "escape"])
def test_a_failing_plugin_pushes_nothing(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin, entry: str
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    with pytest.raises(RunError, match="exited with code 1"):
        _execute(_spec(pin, entry=entry))
    assert _FakeHub.pushed == []


def test_a_commit_url_is_not_reported_as_a_commit(
    monkeypatch: pytest.MonkeyPatch, plugin_env: Path, pin: PluginPin
) -> None:
    _allow(monkeypatch, pin)
    _wire(monkeypatch, plugin_env)
    _FakeHub.commit = "https://huggingface.co/datasets/acme/out/commit/abc"
    assert _execute(_spec(pin)) == "acme/out"
    assert json.loads(Path("receipt.json").read_text())["output"]["commit_sha"] is None


def test_a_git_source_s_build_runs_through_the_sandbox_and_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from strata_forge.plugins.environment import BuildJob

    seen: list[dict[str, Any]] = []

    def no_bwrap(
        _: Sandbox, inner: Any, *, paths: Any, env: dict[str, str], gpu: bool
    ) -> list[str]:
        seen.append({"paths": paths, "env": env, "gpu": gpu})
        return ["/usr/bin/env", "-i", *(f"{k}={v}" for k, v in env.items()), *inner]

    monkeypatch.setattr(plugin_runner, "sandbox_command", no_bwrap)
    source, out, deps = tmp_path / "source", tmp_path / "dist", tmp_path / "deps"
    for d in (source, out, deps):
        d.mkdir()
    build = plugin_runner._sandboxed_builder(Sandbox("bwrap", None), tmp_path)  # pyright: ignore[reportPrivateUsage]

    def job(code: str) -> BuildJob:
        return BuildJob(
            command=(sys.executable, "-I", "-c", code),
            env={"PATH": "/usr/bin:/bin"},
            readonly=(deps,),
            writable=(source, out),
            source=source,
            out=out,
        )

    asyncio.run(build(job(f"open({str(out / 'w.whl')!r}, 'w').write('x')")))
    assert (out / "w.whl").exists()
    (call,) = seen
    assert call["gpu"] is False
    assert call["paths"].writable == (source, out)
    assert deps in call["paths"].readonly
    assert Path(sys.prefix).resolve() in call["paths"].readonly  # the engine, read-only
    with pytest.raises(PluginManifestError, match=r"failed \(exit 3\).*broken backend"):
        asyncio.run(build(job("import sys; sys.stderr.write('broken backend'); sys.exit(3)")))
