"""The sandbox for real: bubblewrap, and bubblewrap as ``nobody`` through sudo, on a Linux machine
that has them. Skipped unless ``STRATA_SANDBOX_LIVE=1`` (CI runners have no usable bubblewrap);
run in a privileged Linux container with an init process (``docker run --privileged --init``:
without one, orphans the tests kill stay as zombies), once as a user without sudo (``bwrap``) and
once as one with passwordless sudo (``uid``). See ``docs/modules/plugins.md``.

Each test lays the run out the way the runner does (a private ``0700`` work directory, the
engine read-only, the plugin's directory, the job and the inputs read-only, one output) and runs
real code inside, which reports what it could and could not do.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from strata_forge.plugins import sandbox as sb

pytestmark = pytest.mark.skipif(
    os.environ.get("STRATA_SANDBOX_LIVE") != "1" or not sys.platform.startswith("linux"),
    reason="needs a real bubblewrap (set STRATA_SANDBOX_LIVE=1 on a Linux machine with bwrap)",
)

PROBE = r"""
import json, os, socket, sys
job, inputs, output, engine, canary = sys.argv[1:6]
report = {"uid": os.getuid()}
report["job"] = json.load(open(job))["hello"]
report["inputs"] = sorted(os.listdir(inputs))
for name, path in (("engine", os.path.join(engine, "planted.txt")), ("canary", canary)):
    try:
        if name == "engine":
            open(path, "w").write("x")
            report[name] = "WROTE"
        else:
            report[name] = open(path).read()
    except OSError as exc:
        report[name] = type(exc).__name__
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
    report["net"] = "OPEN"
except OSError as exc:
    report["net"] = type(exc).__name__
open(os.path.join(output, "report.json"), "w").write(json.dumps(report))
"""


def _layout() -> tuple[Path, dict[str, Path]]:
    work = Path(tempfile.mkdtemp(prefix="strata-plugin-live-"))  # 0700, as the runner's
    inputs = work / "inputs"
    (inputs / "datasets").mkdir(parents=True)
    (inputs / "datasets" / "steps.parquet").write_bytes(b"PAR1")
    output = work / "output"
    output.mkdir()
    job = work / "job.json"
    job.write_text(json.dumps({"hello": "job"}))
    canary = Path.home() / ".cache" / "huggingface" / "token"
    canary.parent.mkdir(parents=True, exist_ok=True)
    canary.write_text("hf_CANARYxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    return work, {"inputs": inputs, "output": output, "job": job, "canary": canary}


def _probe(sandbox: sb.Sandbox) -> dict[str, Any]:
    work, p = _layout()
    try:
        paths = sb.SandboxPaths(
            readonly=(*sb.engine_paths(), p["job"], p["inputs"]),
            writable=(p["output"],),
        )
        if sandbox.isolation == "uid":
            sb.chmod_for_uid(readable=[p["job"], p["inputs"]], writable=[p["output"]])
        inner = [
            sys.executable,
            "-I",
            "-c",
            PROBE,
            str(p["job"]),
            str(p["inputs"]),
            str(p["output"]),
            sys.prefix,
            str(p["canary"]),
        ]
        command = sb.sandbox_command(
            sandbox, inner, paths=paths, env={"PATH": "/usr/bin:/bin"}, gpu=False
        )
        done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads((p["output"] / "report.json").read_text())
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            subprocess.run(["sudo", "-n", "rm", "-rf", "--", str(work)], check=False)  # noqa: S603, S607


def test_the_sandbox_this_machine_offers_runs_plugin_code_apart() -> None:
    sandbox = asyncio.run(sb.choose_sandbox())
    report = _probe(sandbox)
    assert report["job"] == "job"
    assert report["inputs"] == ["datasets"]
    assert report["engine"] in {"OSError", "PermissionError"}  # read-only: EROFS or EACCES
    assert report["canary"] == "FileNotFoundError"  # the runner's home is not there at all
    assert report["net"] != "OPEN"
    if sandbox.isolation == "uid":
        assert report["uid"] == sb.NOBODY_UID
    else:
        assert report["uid"] == os.getuid()
    print(f"isolation={sandbox.isolation} report={report}")


def _wheel(directory: Path, files: dict[str, bytes]) -> tuple[Path, str, str]:
    """A wheel of ``files`` plus a dist-info for toy-prm 0.1.0: its path, sha256, files digest."""
    import base64
    import hashlib
    import zipfile

    from tests.unit.plugins.conftest import TOY_ENTRY_POINTS

    def h(b: bytes) -> str:
        return (
            "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode()
        )

    info = "toy_prm-0.1.0.dist-info"
    files = {
        **files,
        f"{info}/METADATA": b"Metadata-Version: 2.1\nName: toy-prm\nVersion: 0.1.0\n",
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": TOY_ENTRY_POINTS.encode(),
    }
    path = directory / "toy_prm-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr(
            f"{info}/RECORD",
            "".join(f"{n},{h(d)},{len(d)}\n" for n, d in files.items()) + f"{info}/RECORD,,\n",
        )
    lines = sorted((n, h(d)) for n, d in files.items() if not n.endswith("/WHEEL"))
    digest = hashlib.sha256("".join(f"{n},{x}\n" for n, x in lines).encode()).hexdigest()
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), digest


@pytest.mark.parametrize("planted_pth", [False, True])
def test_a_whole_run_in_the_real_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted_pth: bool
) -> None:
    """The review's attack, end to end: a wheel shipping a ``.pth`` that would plant a token
    stealer in the engine is refused before anything runs; the same build without it runs in the
    real sandbox and its scores are pushed."""
    from pydantic import SecretStr

    from strata_forge.pipelines import plugin_runner
    from strata_forge.pipelines._common import RunError, RunSecrets
    from strata_forge.plugins.pins import (
        ALLOWED_PLUGINS_ENV,
        AllowedPlugin,
        PluginPin,
        PluginSource,
        RunnerAllowance,
    )
    from tests.unit.pipelines import test_plugin_runner as t
    from tests.unit.plugins.conftest import SHA, TOY_MODULE

    marker = tmp_path / "pth-ran"
    files = {"toy_prm/__init__.py": TOY_MODULE.encode()}
    if planted_pth:
        engine_site = next(Path(p) for p in sys.path if p.endswith("site-packages"))
        files["zzz_evil.pth"] = (
            f"import os; open({str(marker)!r}, 'w').write('ran'); "
            f"open({str(engine_site / 'PWNED.txt')!r}, 'w').write('x')\n"
        ).encode()
    (tmp_path / "dl").mkdir()
    wheel, sha, digest = _wheel(tmp_path / "dl", files)
    pin = PluginPin(
        distribution="toy-prm",
        version="0.1.0",
        source=PluginSource(
            kind="hf",
            repo="acme/toy-prm",
            commit=SHA,
            wheel=wheel.name,
            wheel_sha256=sha,
            files_sha256=digest,
        ),
    )
    allowance = RunnerAllowance(
        plugin=AllowedPlugin(id="toy", label="Toy", pin=pin),
        entry_points=tuple(plugin_runner.EntryPointDecl.model_validate(e) for e in t.ENTRIES),
    )
    monkeypatch.setenv(ALLOWED_PLUGINS_ENV, json.dumps([allowance.model_dump(mode="json")]))
    monkeypatch.setenv("STRATA_PLUGIN_ENVS_DIR", str(Path.home() / ".strata-forge" / "plugin-envs"))

    class Hub(t._FakeHub):  # pyright: ignore[reportPrivateUsage]
        async def download_file(self, *_: Any, local_dir: Path, **__: Any) -> Path:
            target = Path(local_dir) / wheel.name
            shutil.copy(wheel, target)
            return target

    async def inputs(spec: Any, entry: Any, secrets: Any, folder: Path) -> Any:
        dest = folder / "datasets" / "steps.parquet"
        t._steps_parquet(dest, step_type=True)  # pyright: ignore[reportPrivateUsage]
        return {"steps": dest}, {}, {"steps": 5}

    monkeypatch.setattr(plugin_runner, "_download_inputs", inputs)
    monkeypatch.setattr(plugin_runner, "HFHubClient", Hub)
    fake = t._FakeHub  # pyright: ignore[reportPrivateUsage]
    fake.pushed, fake.last, fake.private = [], {}, True
    secrets = RunSecrets(hf_token=SecretStr("hf_" + "Q" * 34), from_environment=False)
    spec = t._spec(pin)  # pyright: ignore[reportPrivateUsage]
    if planted_pth:
        with pytest.raises(RunError, match=r"zzz_evil\.pth holds executable lines"):
            asyncio.run(plugin_runner._execute(spec, secrets, None))  # pyright: ignore[reportPrivateUsage]
        assert fake.pushed == []
    else:
        destination = asyncio.run(plugin_runner._execute(spec, secrets, None))  # pyright: ignore[reportPrivateUsage]
        assert destination.startswith("acme/out")
        manifest = json.loads(fake.last["manifest.json"])
        print(f"sandbox={manifest['sandbox']}")
    assert not marker.exists()


BACKEND = r"""
import json, os, socket, zipfile

def _attempts(out):
    report = {"uid": os.getuid()}
    engine = os.environ["PROBE_ENGINE"]
    try:
        open(os.path.join(engine, "planted-by-build.txt"), "w").write("x")
        report["engine"] = "WROTE"
    except OSError as exc:
        report["engine"] = type(exc).__name__
    try:
        report["canary"] = open(os.environ["PROBE_CANARY"]).read()
    except OSError as exc:
        report["canary"] = type(exc).__name__
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        report["net"] = "OPEN"
    except OSError as exc:
        report["net"] = type(exc).__name__
    open(os.path.join(out, "report.json"), "w").write(json.dumps(report))

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _attempts(os.environ["PROBE_OUT"])
    name = "toy_prm-0.1.0-py3-none-any.whl"
    info = "toy_prm-0.1.0.dist-info"
    with zipfile.ZipFile(os.path.join(wheel_directory, name), "w") as z:
        z.writestr("toy_prm/__init__.py", "")
        z.writestr(info + "/METADATA", "Metadata-Version: 2.1\nName: toy-prm\nVersion: 0.1.0\n")
        z.writestr(info + "/WHEEL", "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(info + "/RECORD", "")
    return name
"""


def test_a_git_build_backend_runs_only_inside_the_sandbox(tmp_path: Path) -> None:
    """A build backend is the plugin's code: it runs in the sandbox, offline, and can write
    nothing but its own output (an in-tree PEP 517 backend stands in for a hostile one)."""
    from strata_forge.pipelines import plugin_runner
    from strata_forge.plugins import environment as env_mod

    work = Path(tempfile.mkdtemp(prefix="strata-plugin-build-"))
    try:
        source, deps, out = work / "source", work / "deps", work / "dist"
        for d in (source, deps, out):
            d.mkdir()
        (source / "pyproject.toml").write_text(
            '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        )
        (source / "backend.py").write_text(BACKEND)
        _, p = _layout()
        base = Path.home() / ".strata-forge" / "plugin-envs"
        install_env = env_mod.install_environ(home=work / "home", pip_cache=work / "cache")
        tools = asyncio.run(env_mod._tools(base, env_mod._run, install_env))  # pyright: ignore[reportPrivateUsage]
        sandbox = asyncio.run(sb.choose_sandbox())
        job = env_mod.BuildJob(
            command=(
                str(tools / "bin" / "python"),
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                str(out),
                str(source),
            ),
            env={
                "PYTHONPATH": str(deps),
                "PATH": "/usr/bin:/bin",
                "PROBE_ENGINE": sys.prefix,
                "PROBE_CANARY": str(p["canary"]),
                "PROBE_OUT": str(out),
            },
            readonly=(tools, deps),
            writable=(source, out),
            source=source,
            out=out,
        )
        build = plugin_runner._sandboxed_builder(sandbox, work)  # pyright: ignore[reportPrivateUsage]
        asyncio.run(build(job))
        report = json.loads((out / "report.json").read_text())
        assert (out / "toy_prm-0.1.0-py3-none-any.whl").exists()
        assert report["engine"] in {"OSError", "PermissionError"}
        assert not (Path(sys.prefix) / "planted-by-build.txt").exists()
        assert report["canary"] == "FileNotFoundError"
        assert report["net"] != "OPEN"
        print(f"build isolation={sandbox.isolation} report={report}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            subprocess.run(["sudo", "-n", "rm", "-rf", "--", str(work)], check=False)  # noqa: S603, S607
