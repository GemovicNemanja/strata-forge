"""The plugin's own environment on the compute target, built by the runner from the pin.

Every allowed plugin build gets a dedicated directory of its own, never the engine's venv: the
pinned plugin and the hash-pinned dependencies it needs beyond the engine's own, installed with
``pip install --target``. The plugin's process is the ENGINE's interpreter, started inside the
sandbox, with this directory appended to its import path (after the engine's own, so nothing in
it can shadow the engine). The engine's environment is bound read-only there.

**No plugin code runs outside the sandbox, at any step.** The runner is the process holding the
run's credential, and anything executing as its uid outside a sandbox could read its files or
rewrite the engine it imports. So:

- **Dependencies are wheels, hash-pinned.** They are installed with ``--require-hashes``,
  ``--only-binary=:all:`` and ``--no-deps`` from the pin's own list
  (:class:`~strata_forge.plugins.pins.LockedDependency`): pip refuses any file whose hash is not
  listed, and installing a wheel unpacks it, it runs nothing.
- **An hf source** is a wheel the runner downloads (with the run's own credential, which a private
  repository may need), refuses unless its SHA-256 is the pinned one, and unpacks the same way.
- **A git source** is fetched anonymously at the commit, and its wheel is BUILT INSIDE THE
  SANDBOX (no network, only the pinned build requirements on its path, the source and an output
  directory writable, the engine read-only); the build backend is the plugin's code. The wheel it
  produces is only unpacked.
- **The environment is read, never run.** What it holds is read from its files in this process
  (``importlib.metadata``); no interpreter is ever started with it on its path outside the
  sandbox, and the plugin directory is never a site directory anywhere, so a ``.pth`` file in it
  is never processed. One with an executable line is refused all the same.
- **Pip itself** runs from a tools environment of its own (the standard library's ``venv`` and
  its bundled pip), never with the plugin directory on its path, with an allow-listed
  environment and an empty ``HOME``: no credential, no ``~/.netrc``, no pip or git configuration
  of the machine's.

Every build is checked before it is used, and again on every reuse: the pinned distribution and
version are there, every file matches its RECORD hash, and the files' digest is the pin's
``files_sha256`` (:func:`~strata_forge.plugins.pins.record_mismatch`); every distribution the
plugin requires is present (the plugin directory or the engine's environment). On reuse, the
whole directory must also hash to the digest recorded when it was built (:func:`tree_digest`):
the plugin's RECORD covers only its own files, and this covers the dependencies, their data
files and anything added beside them. The directory is
keyed by everything it holds (:func:`environment_key`), so a changed pin, engine or interpreter
gets a new one. The resolved set of installed distributions is recorded in the run's manifest.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, cast

from strata_forge import __version__
from strata_forge.plugins.manifest import PluginManifestError, normalize_distribution
from strata_forge.plugins.pins import installed_mismatch, record_mismatch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from strata_forge.plugins.pins import LockedDependency, PluginPin

__all__ = [
    "ENGINE_DISTRIBUTION",
    "PLUGIN_ENVS_DIR_ENV",
    "BuildJob",
    "PluginEnvironment",
    "engine_requirement",
    "ensure_environment",
    "environment_key",
    "install_environ",
    "pth_problems",
    "tree_digest",
    "verify_wheel",
]

#: Overrides where plugin environments are kept (default ``~/.strata-forge/plugin-envs``).
PLUGIN_ENVS_DIR_ENV = "STRATA_PLUGIN_ENVS_DIR"
#: The engine's distribution name.
ENGINE_DISTRIBUTION = "strata-forge"
#: The file that marks a finished environment, and records what it holds. Written last.
_MARKER = ".strata-plugin-env.json"
#: How much of a failed command's output the error keeps.
_OUTPUT_TAIL_CHARS = 2000
_INSTALL_ENV_NAMES = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER"})
_CREDENTIAL_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
_GIT_SAFE = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "protocol.file.allow=never",
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "submodule.recurse=false",
)


@dataclass(frozen=True, slots=True)
class PluginEnvironment:
    """A finished, verified plugin environment."""

    key: str
    root: Path
    #: The directory the plugin and its dependencies are installed in (``pip --target``).
    site: Path
    #: Every distribution installed there, ``{"name", "version"}``, sorted by name.
    packages: tuple[dict[str, str], ...]
    #: What it was built from: the plugin, then each locked dependency.
    requirements: tuple[str, ...]
    #: Whether this run reused an environment an earlier run built.
    reused: bool

    @property
    def site_paths(self) -> tuple[str, ...]:
        return (str(self.site),)

    def record(self) -> dict[str, object]:
        """What the run's manifest records about the environment."""
        return {
            "key": self.key,
            "reused": self.reused,
            "engine": engine_requirement(),
            "requirements": list(self.requirements),
            "packages": [dict(p) for p in self.packages],
        }


@dataclass(frozen=True, slots=True)
class BuildJob:
    """A git source's build, for the caller to run inside the sandbox: ``command`` with ``env``,
    reading ``readonly`` and writing only ``writable`` (the source tree and ``out``)."""

    command: tuple[str, ...]
    env: dict[str, str]
    readonly: tuple[Path, ...]
    writable: tuple[Path, ...]
    source: Path
    out: Path


type CommandRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[tuple[int, str]]]
type Builder = Callable[[BuildJob], Awaitable[None]]


def _direct_url(dist: metadata.Distribution) -> dict[str, object] | None:
    try:
        raw = dist.read_text("direct_url.json")
    except OSError, ValueError:
        return None
    if not raw:
        return None
    try:
        info: object = json.loads(raw)
    except ValueError:
        return None
    return cast("dict[str, object]", info) if isinstance(info, dict) else None


def engine_requirement() -> str:
    """What this engine build is: its PEP 610 source when it has one (a git commit, an archive, a
    local directory), else the exact release. Part of every environment's key: dependencies are
    locked against one engine's environment."""
    try:
        dist = metadata.distribution(ENGINE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return f"{ENGINE_DISTRIBUTION}=={__version__}"
    info = _direct_url(dist)
    url = info.get("url") if info is not None else None
    if info is None or not isinstance(url, str):
        return f"{ENGINE_DISTRIBUTION}=={dist.version}"
    vcs = info.get("vcs_info")
    if isinstance(vcs, dict):
        commit = cast("dict[str, object]", vcs).get("commit_id")
        if isinstance(commit, str):
            return f"{ENGINE_DISTRIBUTION} @ git+{url}@{commit}"
    return f"{ENGINE_DISTRIBUTION} @ {url}"


def _plugin_requirement(pin: PluginPin) -> str:
    source = pin.source
    if source.kind == "git":
        return f"{pin.distribution}=={pin.version} @ git+{source.install_url()}@{source.commit}"
    return (
        f"{pin.distribution}=={pin.version} @ {source.install_url()}#sha256={source.wheel_sha256}"
    )


def environment_key(pin: PluginPin, engine: str) -> str:
    """The environment's name: a hash of the full pin, the engine and the interpreter."""
    canonical = json.dumps(
        {
            "pin": pin.model_dump(mode="json"),
            "engine": engine,
            "python": sys.implementation.cache_tag,
            "platform": sysconfig.get_platform(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def envs_dir(environ: Mapping[str, str] | None = None) -> Path:
    source = os.environ if environ is None else environ
    override = source.get(PLUGIN_ENVS_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".strata-forge" / "plugin-envs"


def install_environ(
    *, home: Path, pip_cache: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment pip and git run in: an allow-list with every credential-shaped name
    removed, a fresh ``HOME`` (no ``.netrc``, no git or pip configuration), and no prompts."""
    source = os.environ if environ is None else environ
    env = {
        name: value
        for name, value in source.items()
        if name in _INSTALL_ENV_NAMES and not any(w in name.upper() for w in _CREDENTIAL_WORDS)
    }
    env.update(
        {
            "HOME": str(home),
            "PYTHONNOUSERSITE": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(pip_cache),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    return env


def verify_wheel(path: Path, sha256: str) -> None:
    """Refuse a downloaded wheel whose SHA-256 is not the pinned one."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        msg = (
            f"the downloaded wheel {path.name} has sha256 {digest.hexdigest()}, not the pinned "
            f"{sha256}; nothing was installed"
        )
        raise PluginManifestError(msg)


async def _run(args: Sequence[str], env: Mapping[str, str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        env=dict(env),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        out, _ = await proc.communicate()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


async def _checked(
    run: CommandRunner, args: Sequence[str], env: Mapping[str, str], what: str
) -> str:
    code, output = await run(args, env)
    if code != 0:
        tail = output.strip()[-_OUTPUT_TAIL_CHARS:]
        msg = f"{what} failed (exit {code})" + (f": {tail}" if tail else "")
        raise PluginManifestError(msg)
    return output


# ── reading an environment, without running it ─────────────────────────────────────────────────


def pth_problems(site: Path) -> list[str]:
    """Every ``.pth`` file in ``site`` with a line Python would EXECUTE (one starting with
    ``import``). The plugin directory is never processed as a site directory, so none of them
    would run; a build that ships one is refused all the same, as a build that tried."""
    found: list[str] = []
    for pth in sorted(site.glob("*.pth")):
        try:
            text = pth.read_text(encoding="utf-8", errors="replace")
        except OSError:
            found.append(f"{pth.name} cannot be read")
            continue
        if any(line.startswith(("import ", "import\t")) for line in text.splitlines()):
            found.append(f"{pth.name} holds executable lines")
    return found


def _packages(site: Path) -> tuple[dict[str, str], ...]:
    return tuple(
        sorted(
            (
                {"name": str(d.metadata["Name"]), "version": str(d.version)}
                for d in metadata.distributions(path=[str(site)])
            ),
            key=lambda p: p["name"].lower(),
        )
    )


def _missing_requirements(pin: PluginPin, site: Path) -> list[str]:
    """The plugin's requirements (``Requires-Dist``, markers evaluated here) that neither its own
    directory nor the engine's environment provides: a lock that left one out fails here, by
    name, rather than as an import error inside the sandbox."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # pragma: no cover -- packaging ships with every engine environment
        return []
    plugin = next(
        (
            d
            for d in metadata.distributions(path=[str(site)])
            if normalize_distribution(d.metadata["Name"] or "") == pin.distribution
        ),
        None,
    )
    if plugin is None:
        return []
    present = {
        normalize_distribution(d.metadata["Name"] or "")
        for d in [*metadata.distributions(path=[str(site)]), *metadata.distributions()]
    }
    missing: list[str] = []
    for raw in plugin.requires or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            missing.append(raw)
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        if normalize_distribution(req.name) not in present:
            missing.append(req.name)
    return missing


def _verify(pin: PluginPin, site: Path) -> str | None:
    paths = [str(site)]
    problem = installed_mismatch(pin, path=paths) or record_mismatch(pin, path=paths)
    if problem is not None:
        return problem
    pth = pth_problems(site)
    if pth:
        return "the plugin environment holds .pth files that would run code: " + "; ".join(pth)
    missing = _missing_requirements(pin, site)
    if missing:
        return (
            f"{pin.distribution} requires {', '.join(sorted(missing))}, which neither its pinned "
            "dependencies nor the engine's environment provide"
        )
    return None


def tree_digest(site: Path) -> str:
    """The SHA-256 over EVERY entry of the plugin directory: each directory, and each file's
    path, executable bit and content hash, sorted. The plugin's own RECORD covers only its own
    files; this covers its dependencies, their data files, bytecode and anything added beside
    them, so a reused directory must be byte for byte the one this runner built. A link or a
    special file anywhere refuses the directory (``pip --target`` writes neither).

    Each entry is its fields as the file system's own bytes (``os.fsencode``), each ended by a NUL,
    which no path component can hold: a name with a newline cannot pose as two entries, and a name
    that is not valid text is hashed as it is rather than refused."""
    digest = hashlib.sha256()
    entries = sorted((os.fsencode(p.relative_to(site)), p) for p in site.rglob("*"))
    for rel, path in entries:
        if path.is_symlink():
            msg = f"the plugin environment holds a link ({rel!a}); it is not reused"
            raise PluginManifestError(msg)
        if path.is_dir():
            digest.update(b"d\0" + rel + b"\0")
            continue
        if not path.is_file():
            msg = f"the plugin environment holds a special file ({rel!a}); it is not reused"
            raise PluginManifestError(msg)
        content = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                content.update(chunk)
        mode = b"x" if path.stat().st_mode & 0o111 else b"-"
        digest.update(b"f\0" + rel + b"\0" + mode + b"\0" + content.hexdigest().encode() + b"\0")
    return digest.hexdigest()


def _tree_matches(site: Path, recorded: str) -> bool:
    """Whether the directory still hashes to ``recorded``; any failure to hash it means no."""
    try:
        return tree_digest(site) == recorded
    except OSError, ValueError, PluginManifestError:
        return False


def _recorded(marker: Path, key: str) -> tuple[tuple[str, ...], str] | None:
    """The requirements a finished environment was built from and the digest of its whole
    directory, or ``None`` when the marker is missing, unreadable, names another key, or predates
    the directory digest (an unfinished, foreign or older directory, rebuilt)."""
    try:
        data: object = json.loads(marker.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(data, dict):
        return None
    record = cast("dict[str, object]", data)
    requirements = record.get("requirements")
    if record.get("key") != key or not isinstance(requirements, list):
        return None
    items = cast("list[object]", requirements)
    tree = record.get("tree_sha256")
    if not all(isinstance(i, str) for i in items) or not isinstance(tree, str) or not tree:
        return None
    return tuple(cast("list[str]", items)), tree


# ── building one ────────────────────────────────────────────────────────────────────────────────


def _lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _unlock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


async def _tools(base: Path, run: CommandRunner, env: Mapping[str, str]) -> Path:
    """The interpreter pip runs from: a venv of the engine's own interpreter holding only the
    pip the standard library bundles (``ensurepip``, offline), shared by every plugin build on
    the machine and never holding anything of a plugin's."""
    tools = base / ".tools" / sys.implementation.cache_tag
    python = tools / "bin" / "python"
    done = tools / ".ready"
    fd = await asyncio.to_thread(_lock, base / ".tools.lock")
    try:
        if not done.exists():
            shutil.rmtree(tools, ignore_errors=True)
            await _checked(
                run,
                [sys.executable, "-I", "-m", "venv", str(tools)],
                env,
                "creating pip's environment",
            )
            await _checked(run, [str(python), "-I", "-m", "pip", "--version"], env, "checking pip")
            done.write_text("ok\n", encoding="utf-8")
    finally:
        _unlock(fd)
    return tools


def _lock_file(path: Path, locked: Sequence[LockedDependency]) -> Path:
    path.write_text("".join(f"{d.requirement_line()}\n" for d in locked), encoding="utf-8")
    return path


async def _install_locked(
    run: CommandRunner,
    pip: Sequence[str],
    locked: Sequence[LockedDependency],
    target: Path,
    lock: Path,
    env: Mapping[str, str],
    what: str,
) -> None:
    """Install hash-pinned wheels into ``target``: nothing unlisted, nothing built, nothing run."""
    if not locked:
        return
    await _checked(
        run,
        [
            *pip,
            "install",
            "--no-input",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
            "--target",
            str(target),
            "-r",
            str(_lock_file(lock, locked)),
        ],
        env,
        what,
    )


async def _fetch_source(
    run: CommandRunner, pin: PluginPin, dest: Path, env: Mapping[str, str]
) -> None:
    """The source tree at the pinned commit, fetched anonymously. git runs none of the
    repository's code: hooks, submodules, filters and local transports are all off."""
    await asyncio.to_thread(dest.mkdir, parents=True)
    git = shutil.which("git") or "git"
    url = pin.source.install_url()
    commit = pin.source.commit
    steps: list[tuple[list[str], str]] = [
        ([git, *_GIT_SAFE, "init", "-q", str(dest)], "preparing the source"),
        (
            [
                git,
                *_GIT_SAFE,
                "-C",
                str(dest),
                "fetch",
                "-q",
                "--depth",
                "1",
                "--no-tags",
                url,
                commit,
            ],
            f"fetching {pin.source.repo} at {commit[:12]}",
        ),
        (
            [git, *_GIT_SAFE, "-C", str(dest), "checkout", "-q", "--detach", "FETCH_HEAD"],
            "checking out the source",
        ),
    ]
    for args, what in steps:
        await _checked(run, args, env, what)
    shutil.rmtree(dest / ".git", ignore_errors=True)


async def _built_wheel(
    pin: PluginPin,
    *,
    run: CommandRunner,
    pip: Sequence[str],
    tools: Path,
    scratch: Path,
    env: Mapping[str, str],
    build: Builder,
) -> Path:
    """A git source's wheel, built inside the sandbox from the fetched commit."""
    source, deps, out = scratch / "source", scratch / "build-deps", scratch / "dist"
    await _fetch_source(run, pin, source, env)
    deps.mkdir()
    out.mkdir()
    await _install_locked(
        run,
        pip,
        pin.build_requires,
        deps,
        scratch / "build-requires.txt",
        env,
        "installing the pinned build requirements",
    )
    await build(
        BuildJob(
            command=(
                str(tools / "bin" / "python"),
                "-m",
                "pip",
                "wheel",
                # Verbose, so the backend's own output streams: a build that writes nothing for
                # the runner's idle limit is treated as stalled and stopped.
                "-v",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--no-input",
                "--wheel-dir",
                str(out),
                str(source),
            ),
            env={
                # PYTHONPATH, not -I: the build backend's own process inherits it.
                "PYTHONPATH": str(deps),
                "PYTHONNOUSERSITE": "1",
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_NO_INPUT": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_CACHE_DIR": "1",
                "PATH": env.get("PATH", "/usr/bin:/bin"),
            },
            readonly=(tools, deps),
            writable=(source, out),
            source=source,
            out=out,
        )
    )
    wheels = [p for p in out.iterdir() if p.suffix == ".whl" and p.is_file() and not p.is_symlink()]
    if len(wheels) != 1:
        msg = f"the build of {pin.distribution} produced {len(wheels)} wheels, not one"
        raise PluginManifestError(msg)
    return wheels[0]


def _readable(tree: Path) -> None:
    """Readable by every uid (the ``nobody`` sandbox reads it), writable by the runner only."""
    for root, _, files in tree.walk():
        root.chmod(0o755)
        for name in files:
            path = root / name
            if not path.is_symlink():
                executable = path.stat().st_mode & 0o111
                path.chmod(0o755 if executable else 0o644)


async def ensure_environment(
    pin: PluginPin,
    *,
    scratch: Path,
    fetch_wheel: Callable[[Path], Awaitable[Path]] | None = None,
    build: Builder | None = None,
    root: Path | None = None,
    run: CommandRunner = _run,
    phase: Callable[[str], None] | None = None,
) -> PluginEnvironment:
    """The verified environment for ``pin``: reused when an earlier run built it and it still
    verifies, else (re)built.

    ``scratch`` is an empty directory of the caller's for downloads and the build (the caller
    removes it). ``fetch_wheel`` downloads an ``hf`` source's wheel into the directory it is given
    (with the run's credential); ``build`` runs a git source's :class:`BuildJob` inside the
    sandbox. Builds of one environment are serialised with a lock file, so two runs on one machine
    never install into the same directory at once.
    """
    engine = engine_requirement()
    key = environment_key(pin, engine)
    base = envs_dir() if root is None else root
    target = base / key
    site = target / "site"
    marker = target / _MARKER
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = install_environ(home=home, pip_cache=scratch / "pip-cache")
    fd = await asyncio.to_thread(_lock, base / f"{key}.lock")
    try:
        recorded = _recorded(marker, key)
        # Reused only when every file in it is the one this runner built (the whole-directory
        # digest: dependencies and data files included), and the pinned build still verifies.
        if (
            recorded is not None
            and await asyncio.to_thread(_tree_matches, site, recorded[1])
            and _verify(pin, site) is None
        ):
            return PluginEnvironment(
                key=key,
                root=target,
                site=site,
                packages=_packages(site),
                requirements=recorded[0],
                reused=True,
            )
        shutil.rmtree(target, ignore_errors=True)
        tools = await _tools(base, run, env)
        pip = (str(tools / "bin" / "python"), "-I", "-m", "pip")
        if phase is not None:
            phase(f"installing {pin.distribution} {pin.version} into its own environment")
        if pin.source.kind == "hf":
            if fetch_wheel is None or pin.source.wheel_sha256 is None:
                msg = "an hf plugin source needs its wheel downloaded and its sha256 pinned"
                raise PluginManifestError(msg)
            (scratch / "wheel").mkdir()
            wheel = await fetch_wheel(scratch / "wheel")
            await asyncio.to_thread(verify_wheel, wheel, pin.source.wheel_sha256)
        else:
            if build is None:
                msg = "a git plugin source is built inside the sandbox, and no sandbox was given"
                raise PluginManifestError(msg)
            wheel = await _built_wheel(
                pin, run=run, pip=pip, tools=tools, scratch=scratch, env=env, build=build
            )
        site.mkdir(parents=True)
        await _install_locked(
            run,
            pip,
            pin.dependencies,
            site,
            scratch / "requirements.txt",
            env,
            f"installing the pinned dependencies of {pin.distribution}",
        )
        await _checked(
            run,
            [
                *pip,
                "install",
                "--no-input",
                "--no-deps",
                "--no-index",
                "--target",
                str(site),
                str(wheel),
            ],
            env,
            f"installing {pin.distribution} {pin.version}",
        )
        problem = _verify(pin, site)
        if problem is not None:
            shutil.rmtree(target, ignore_errors=True)
            raise PluginManifestError(problem)
        await asyncio.to_thread(_readable, target)
        requirements = (
            _plugin_requirement(pin),
            *(d.requirement_line() for d in pin.dependencies),
        )
        tree = await asyncio.to_thread(tree_digest, site)
        marker.write_text(
            json.dumps({"key": key, "requirements": list(requirements), "tree_sha256": tree}),
            encoding="utf-8",
        )
        return PluginEnvironment(
            key=key,
            root=target,
            site=site,
            packages=_packages(site),
            requirements=requirements,
            reused=False,
        )
    finally:
        _unlock(fd)
