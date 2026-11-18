"""The plugin's own environment: keyed by everything it holds, built without running any plugin
code outside the sandbox, read from its files rather than run, and verified on every reuse."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.plugins import environment
from strata_forge.plugins.manifest import PluginManifestError
from strata_forge.plugins.pins import LockedDependency, PluginPin, PluginSource, files_digest

from .conftest import SHA, TOY_DIGEST, build_toy, toy_pin, write_dist

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

WHEEL = "toy_prm-0.1.0-py3-none-any.whl"
DEP = LockedDependency(name="dep-a", version="1.0", sha256=("d" * 64,))


class FakeMachine:
    """Stands in for ``python -m venv``, ``pip`` and ``git``: the tools venv is a directory, an
    install with ``-r`` writes the locked dependencies, an install of a wheel writes the toy
    distribution (or what ``plugin`` writes), and git leaves a source tree."""

    def __init__(self, *, plugin: Any = build_toy) -> None:
        self.plugin = plugin
        self.calls: list[list[str]] = []
        self.envs: list[Mapping[str, str]] = []

    async def __call__(self, args: Sequence[str], env: Mapping[str, str]) -> tuple[int, str]:
        self.calls.append(list(args))
        self.envs.append(env)
        if args[1:3] == ["-m", "venv"]:
            (Path(args[3]) / "bin").mkdir(parents=True)
            (Path(args[3]) / "bin" / "python").write_text("")
            return 0, ""
        if "--version" in args:
            return 0, "pip 26.0"
        if "git" in Path(args[0]).name:
            if "checkout" in args:
                dest = Path(args[args.index("-C") + 1])
                (dest / "pyproject.toml").write_text("[project]\nname='toy-prm'\n")
            return 0, ""
        if "install" in args:
            target = Path(args[args.index("--target") + 1])
            if "-r" in args:
                lock = Path(args[args.index("-r") + 1]).read_text()  # noqa: ASYNC240
                for line in lock.splitlines():
                    name, _, rest = line.partition("==")
                    write_dist(
                        target,
                        name=name.replace("-", "_"),
                        version=rest.split()[0],
                        direct_url=None,
                        entry_points="",
                    )
                return 0, "Successfully installed"
            self.plugin(target)
            return 0, "Successfully installed"
        return 1, f"unexpected command {args}"


def _hf_pin(sha: str, *, dependencies: tuple[LockedDependency, ...] = ()) -> PluginPin:
    return PluginPin(
        distribution="toy-prm",
        version="0.1.0",
        source=PluginSource(
            kind="hf",
            repo="acme/toy",
            commit=SHA,
            wheel=WHEEL,
            wheel_sha256=sha,
            files_sha256=TOY_DIGEST,
        ),
        dependencies=dependencies,
    )


def _wheel(tmp_path: Path, content: bytes = b"a wheel") -> tuple[Any, str]:
    path = tmp_path / "dl" / WHEEL
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)

    async def fetch(into: Path) -> Path:
        return path

    return fetch, hashlib.sha256(content).hexdigest()


def _ensure(
    pin: PluginPin, tmp_path: Path, run: Any, **kwargs: Any
) -> environment.PluginEnvironment:
    scratch = tmp_path / f"scratch-{len(list(tmp_path.glob('scratch-*')))}"
    scratch.mkdir(parents=True)
    return asyncio.run(
        environment.ensure_environment(
            pin, scratch=scratch, root=tmp_path / "envs", run=run, **kwargs
        )
    )


def test_the_key_changes_with_the_pin_the_engine_and_the_locks(pin: PluginPin) -> None:
    key = environment.environment_key(pin, "strata-forge==0.6.0")
    assert key == environment.environment_key(pin, "strata-forge==0.6.0")
    moved = pin.model_copy(update={"source": pin.source.model_copy(update={"commit": "b" * 40})})
    assert environment.environment_key(moved, "strata-forge==0.6.0") != key
    assert environment.environment_key(pin, "strata-forge==0.7.0") != key
    locked = pin.model_copy(update={"dependencies": (DEP,)})
    assert environment.environment_key(locked, "strata-forge==0.6.0") != key


def test_the_install_environment_holds_no_credential(tmp_path: Path) -> None:
    env = environment.install_environ(
        home=tmp_path / "home",
        pip_cache=tmp_path / "cache",
        environ={
            "PATH": "/bin",
            "HOME": "/home/u",
            "HF_TOKEN": "hf_x",
            "GITHUB_TOKEN": "ghp_x",
            "PIP_INDEX_URL": "https://user:pass@index/simple",
            "STRATA_SECRETS_FILE": "/run/s",
            "NETRC": "/home/u/.netrc",
        },
    )
    assert env["PATH"] == "/bin"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    for name in ("HF_TOKEN", "GITHUB_TOKEN", "PIP_INDEX_URL", "STRATA_SECRETS_FILE", "NETRC"):
        assert name not in env


def test_an_hf_build_is_unpacked_with_hash_pinned_wheels_and_reused(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)
    pin = _hf_pin(sha, dependencies=(DEP,))
    machine = FakeMachine()
    built = _ensure(pin, tmp_path, machine, fetch_wheel=fetch)
    assert not built.reused
    assert built.site == built.root / "site"
    assert {"name": "dep-a", "version": "1.0"} in built.packages
    deps, plugin = [c for c in machine.calls if "install" in c]
    # Dependencies: only hash-checked wheels, never built, never resolved further.
    for flag in ("--require-hashes", "--only-binary=:all:", "--no-deps"):
        assert flag in deps
    assert f"--hash=sha256:{'d' * 64}" in Path(deps[deps.index("-r") + 1]).read_text()
    # The plugin: the verified local wheel, offline.
    assert "--no-index" in plugin
    assert plugin[-1].endswith(WHEEL)
    # pip runs isolated from its own tools environment, never with the plugin on its path.
    assert all(c[1:3] == ["-I", "-m"] for c in (deps, plugin))
    for env in machine.envs:
        assert not any("TOKEN" in name for name in env)
    again = _ensure(pin, tmp_path, machine, fetch_wheel=fetch)
    assert again.reused
    assert again.key == built.key
    assert len([c for c in machine.calls if "install" in c]) == 2
    assert again.record()["requirements"] == list(built.requirements)


def test_an_hf_wheel_is_hash_checked_before_anything_installs_it(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)
    tampered = FakeMachine()
    with pytest.raises(PluginManifestError, match="not the pinned"):
        _ensure(_hf_pin("0" * 64), tmp_path, tampered, fetch_wheel=fetch)
    assert not [c for c in tampered.calls if "install" in c]
    with pytest.raises(PluginManifestError, match="downloaded"):
        _ensure(_hf_pin(sha), tmp_path, FakeMachine())


def test_an_environment_whose_files_changed_is_rebuilt(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)
    pin = _hf_pin(sha)
    machine = FakeMachine()
    built = _ensure(pin, tmp_path, machine, fetch_wheel=fetch)
    (built.site / "toy_prm" / "__init__.py").write_text("print('changed')\n")
    again = _ensure(pin, tmp_path, machine, fetch_wheel=fetch)
    assert not again.reused
    assert len([c for c in machine.calls if "install" in c]) == 2


def test_a_build_that_is_not_the_pin_is_refused_and_removed(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)
    pin = _hf_pin(sha)
    pin = pin.model_copy(
        update={"source": pin.source.model_copy(update={"files_sha256": "e" * 64})}
    )
    with pytest.raises(PluginManifestError, match="not the pinned"):
        _ensure(pin, tmp_path, FakeMachine(), fetch_wheel=fetch)
    assert not [
        p for p in (tmp_path / "envs").iterdir() if p.is_dir() and not p.name.startswith(".")
    ]


def _with_pth(site: Path) -> None:
    build_toy(site)
    (site / "zzz_evil.pth").write_text("import os; os.system('touch /tmp/pwned')\n")


def test_a_build_shipping_an_executable_pth_is_refused(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)
    with pytest.raises(PluginManifestError, match=r"\.pth"):
        _ensure(_hf_pin(sha), tmp_path, FakeMachine(plugin=_with_pth), fetch_wheel=fetch)


def test_pth_problems_tolerate_plain_path_lines(tmp_path: Path) -> None:
    (tmp_path / "paths.pth").write_text("# a comment\nsome/dir\n")
    assert environment.pth_problems(tmp_path) == []
    (tmp_path / "evil.pth").write_text("import\tos\n")
    assert environment.pth_problems(tmp_path) == ["evil.pth holds executable lines"]


def _requiring(site: Path) -> None:
    build_toy(site, requires=("not-a-real-dependency-xyz",))


def test_a_requirement_nothing_provides_is_named(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)
    probe = tmp_path / "probe"
    _requiring(probe)
    (dist,) = metadata.distributions(path=[str(probe)])
    digest = files_digest(dist)
    assert digest is not None
    pin = _hf_pin(sha)
    pinned = pin.model_copy(
        update={"source": pin.source.model_copy(update={"files_sha256": digest})}
    )
    with pytest.raises(PluginManifestError, match="requires not-a-real-dependency-xyz"):
        _ensure(pinned, tmp_path, FakeMachine(plugin=_requiring), fetch_wheel=fetch)


def test_a_git_source_is_built_by_the_sandbox_never_here(tmp_path: Path) -> None:
    jobs: list[environment.BuildJob] = []

    async def build(job: environment.BuildJob) -> None:
        jobs.append(job)
        (job.out / WHEEL).write_bytes(b"built")

    machine = FakeMachine()
    pin = toy_pin()
    env = _ensure(pin, tmp_path, machine, build=build)
    assert not env.reused
    (job,) = jobs
    # The build is pip's, offline, with only the pinned build requirements on its path.
    assert job.command[1:4] == ("-m", "pip", "wheel")
    for flag in ("--no-deps", "--no-build-isolation", "--no-index"):
        assert flag in job.command
    assert job.env["PYTHONPATH"] == str(job.readonly[1])
    assert set(job.writable) == {job.source, job.out}
    assert not any("TOKEN" in name for name in job.env)
    # git fetched the commit with every repository-controlled hook off, and nothing else ran.
    fetch = next(c for c in machine.calls if "fetch" in c)
    assert SHA in fetch
    assert "core.hooksPath=/dev/null" in fetch
    build_reqs = next(c for c in machine.calls if "install" in c and "-r" in c)
    assert "--require-hashes" in build_reqs
    with pytest.raises(PluginManifestError, match="sandbox"):
        _ensure(pin, tmp_path / "x", FakeMachine())


def test_a_build_that_leaves_no_single_wheel_is_refused(tmp_path: Path) -> None:
    async def build(job: environment.BuildJob) -> None:
        (job.out / "a.whl").write_bytes(b"1")
        (job.out / "b.whl").write_bytes(b"2")

    with pytest.raises(PluginManifestError, match="2 wheels"):
        _ensure(toy_pin(), tmp_path, FakeMachine(), build=build)


def test_a_failed_install_reports_its_output(tmp_path: Path) -> None:
    fetch, sha = _wheel(tmp_path)

    async def failing(args: Sequence[str], env: Mapping[str, str]) -> tuple[int, str]:
        if args[1:3] == ["-m", "venv"]:
            (Path(args[3]) / "bin").mkdir(parents=True)
            return 0, ""
        if "--version" in args:
            return 0, "pip"
        return 1, "ERROR: Could not find a version that satisfies the requirement"

    with pytest.raises(PluginManifestError, match=r"exit 1.*satisfies"):
        _ensure(_hf_pin(sha, dependencies=(DEP,)), tmp_path, failing, fetch_wheel=fetch)


# ── the security review's proof of concept, with a real pip ───────────────────────────────────


def _hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _real_wheel(marker: Path, *, pth: bool) -> tuple[bytes, str]:
    """A wheel shaped like the review's: a module, and optionally a ``.pth`` whose import line
    writes ``marker`` if anything ever executes it. Returns the bytes and the files digest."""
    info = "evil_plugin-0.1.0.dist-info"
    files: dict[str, bytes] = {
        "evil_plugin/__init__.py": b"def attack(ctx):\n    pass\n",
        f"{info}/METADATA": b"Metadata-Version: 2.1\nName: evil-plugin\nVersion: 0.1.0\n",
        f"{info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: hand\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{info}/entry_points.txt": b"[strata_forge.plugins]\nattack = evil_plugin:attack\n",
    }
    if pth:
        payload = f"import os; open({str(marker)!r}, 'w').write('ran')\n"
        files["zzz_evil_plugin.pth"] = payload.encode()
    record = "".join(f"{p},{_hash(b)},{len(b)}\n" for p, b in files.items())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for path, data in files.items():
            z.writestr(path, data)
        z.writestr(f"{info}/RECORD", record + f"{info}/RECORD,,\n")
    lines = sorted((p, _hash(b)) for p, b in files.items() if not p.endswith("/WHEEL"))
    digest = hashlib.sha256("".join(f"{p},{h}\n" for p, h in lines).encode()).hexdigest()
    return buffer.getvalue(), digest


@pytest.fixture(scope="module")
def shared_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One plugin-envs root for the real-pip tests, so pip's tools venv is created once."""
    return tmp_path_factory.mktemp("envs")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
@pytest.mark.parametrize("pth", [True, False])
def test_a_wheel_s_pth_never_runs_while_the_environment_is_built(
    tmp_path: Path, shared_root: Path, pth: bool
) -> None:
    marker = tmp_path / "pth-ran"
    data, digest = _real_wheel(marker, pth=pth)
    wheel = tmp_path / "dl" / "evil_plugin-0.1.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(data)

    async def fetch(into: Path) -> Path:
        return wheel

    pin = PluginPin(
        distribution="evil-plugin",
        version="0.1.0",
        source=PluginSource(
            kind="hf",
            repo="acme/evil-plugin",
            commit=SHA,
            wheel=wheel.name,
            wheel_sha256=hashlib.sha256(data).hexdigest(),
            files_sha256=digest,
        ),
    )

    def ensure(name: str) -> environment.PluginEnvironment:
        scratch = tmp_path / name
        scratch.mkdir()
        return asyncio.run(
            environment.ensure_environment(
                pin, scratch=scratch, fetch_wheel=fetch, root=shared_root
            )
        )

    if pth:
        with pytest.raises(PluginManifestError, match=r"zzz_evil_plugin\.pth holds executable"):
            ensure("first")
    else:
        env = ensure("first")
        assert env.packages == ({"name": "evil-plugin", "version": "0.1.0"},)
        assert ensure("again").reused  # a reuse reads the files again, and runs nothing either
    # Whatever happened, no process ever executed the .pth: not pip, not an inspection.
    assert not marker.exists()
