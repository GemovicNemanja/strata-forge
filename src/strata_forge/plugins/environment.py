"""The plugin's own environment on the compute target, built by the runner from the pin.

Every allowed plugin build runs from a dedicated virtual environment, never from the engine's
venv: a rebuild of the engine's venv cannot wipe it, and the plugin's dependencies cannot change
the engine's. The environment holds the pinned plugin and the SAME engine build the runner is
(the child process imports :mod:`strata_forge.plugins.host` from it), resolved together, so the
two agree on every shared dependency.

- **Keyed by what it holds.** :func:`environment_key` hashes the whole pin (distribution, version,
  source, commit, the wheel's hash) and the engine's own install requirement, so a new plugin
  commit or a new engine build gets a new environment and an existing one is never reused for
  anything else. Environments live under :data:`PLUGIN_ENVS_DIR_ENV` (default
  ``~/.strata-forge/plugin-envs``), beside the engine's venv rather than inside it.
- **Installed without credentials.** The install runs with an allow-listed environment and a
  fresh, empty ``HOME``: no Hub or GitHub token, no ``~/.netrc``, no pip or git configuration of
  the machine's, and pip never prompts. A git source is therefore installed anonymously; an
  ``hf`` source's wheel is downloaded by the runner first (with the run's own credential, which
  may be needed for a private repository), checked against its pinned SHA-256 BEFORE anything
  installs it, and installed from that verified local file.
- **Verified after install, and on every reuse.** The pinned build must be what the environment
  holds (:func:`~strata_forge.plugins.pins.installed_mismatch`, PEP 610), and the engine there
  must be the runner's own version. An environment that fails either check, or whose build never
  finished, is removed and rebuilt.
- **Recorded.** The resolved set of installed distributions (name and version) is kept with the
  environment and copied into the run's manifest.

Building the plugin from source runs its build backend on the target. That is the plugin's code
running on the owner's own machine, with no credential, which is where it would run anyway.
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
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, cast

from strata_forge import __version__
from strata_forge.plugins.manifest import PluginManifestError
from strata_forge.plugins.pins import installed_mismatch, record_mismatch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from strata_forge.plugins.pins import PluginPin

__all__ = [
    "ENGINE_DISTRIBUTION",
    "PLUGIN_ENVS_DIR_ENV",
    "PluginEnvironment",
    "engine_requirement",
    "ensure_environment",
    "environment_key",
    "install_environ",
    "plugin_requirement",
    "verify_wheel",
]

#: Overrides where plugin environments are kept (default ``~/.strata-forge/plugin-envs``).
PLUGIN_ENVS_DIR_ENV = "STRATA_PLUGIN_ENVS_DIR"
#: The engine's distribution name, installed into every plugin environment at the runner's build.
ENGINE_DISTRIBUTION = "strata-forge"
#: The engine extra every plugin environment carries: the host reads and writes parquet.
_ENGINE_EXTRA = "hf"
#: The file that marks a finished environment, and records what it holds. Written last.
_MARKER = ".strata-plugin-env.json"
#: How much of a failed install's output the error keeps.
_OUTPUT_TAIL_CHARS = 2000
_INSTALL_ENV_NAMES = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER"})
_CREDENTIAL_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
_SITE_PATHS = (
    "import json, sysconfig;"
    "print(json.dumps(sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})))"
)


@dataclass(frozen=True, slots=True)
class PluginEnvironment:
    """A finished, verified plugin environment."""

    key: str
    root: Path
    python: Path
    #: The environment's site directories, where its distributions' metadata is read.
    site_paths: tuple[str, ...]
    #: Every installed distribution, ``{"name", "version"}``, sorted by name.
    packages: tuple[dict[str, str], ...]
    #: The two requirements it was built from (engine, plugin).
    requirements: tuple[str, str]
    #: Whether this run reused an environment an earlier run built.
    reused: bool

    def interpreter_home(self) -> Path | None:
        """The base interpreter's installation the environment runs on (``pyvenv.cfg``'s
        ``home`` is its ``bin``): a sandbox must expose it read-only beside the environment."""
        try:
            text = (self.root / "pyvenv.cfg").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home" and value.strip():
                return Path(value.strip()).parent
        return None

    def record(self) -> dict[str, object]:
        """What the run's manifest records about the environment."""
        return {
            "key": self.key,
            "reused": self.reused,
            "requirements": list(self.requirements),
            "packages": [dict(p) for p in self.packages],
        }


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
    """The requirement that installs THIS engine build again: its PEP 610 source when it has
    one (a git commit, an archive, a local directory), else the exact release."""
    name = f"{ENGINE_DISTRIBUTION}[{_ENGINE_EXTRA}]"
    try:
        dist = metadata.distribution(ENGINE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return f"{name}=={__version__}"
    info = _direct_url(dist)
    url = info.get("url") if info is not None else None
    if info is None or not isinstance(url, str):
        return f"{name}=={dist.version}"
    vcs = info.get("vcs_info")
    if isinstance(vcs, dict):
        commit = cast("dict[str, object]", vcs).get("commit_id")
        if isinstance(commit, str):
            return f"{name} @ git+{url}@{commit}"
    return f"{name} @ {url}"


def plugin_requirement(pin: PluginPin, wheel: Path | None) -> str:
    """The requirement that installs exactly the pinned build: the git repository at the commit,
    or the verified local wheel with its pinned hash (which pip checks again and records)."""
    source = pin.source
    if source.kind == "git":
        return f"{pin.distribution} @ git+{source.install_url()}@{source.commit}"
    if wheel is None:
        msg = "an hf plugin source installs from its downloaded, verified wheel"
        raise PluginManifestError(msg)
    return f"{pin.distribution} @ {wheel.resolve().as_uri()}#sha256={source.wheel_sha256}"


def environment_key(pin: PluginPin, engine: str) -> str:
    """The environment's name: a hash of the full pin and the engine requirement."""
    canonical = json.dumps({"pin": pin.model_dump(mode="json"), "engine": engine}, sort_keys=True)
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
    """The environment the install runs in: an allow-list with every credential-shaped name
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


type CommandRunner = Callable[[Sequence[str], Mapping[str, str]], Awaitable[tuple[int, str]]]


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


async def _inspect(
    run: CommandRunner, python: Path, env: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    sites_raw = await _checked(
        run, [str(python), "-I", "-c", _SITE_PATHS], env, "reading the plugin environment"
    )
    listed = await _checked(
        run,
        [str(python), "-I", "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        env,
        "listing the plugin environment",
    )
    try:
        sites = tuple(
            str(p) for p in cast("list[object]", json.loads(sites_raw.strip().splitlines()[-1]))
        )
        rows = cast("list[dict[str, object]]", json.loads(listed.strip().splitlines()[-1]))
    except (ValueError, IndexError, TypeError) as exc:
        msg = "the plugin environment could not be read"
        raise PluginManifestError(msg) from exc
    packages = tuple(
        sorted(
            ({"name": str(r.get("name")), "version": str(r.get("version"))} for r in rows),
            key=lambda p: p["name"].lower(),
        )
    )
    return sites, packages


def _verify(pin: PluginPin, sites: Sequence[str]) -> str | None:
    mismatch = installed_mismatch(pin, path=sites) or record_mismatch(pin, path=sites)
    if mismatch is not None:
        return mismatch
    engines = [
        d
        for d in metadata.distributions(path=list(sites))
        if (d.metadata["Name"] or "").lower() == ENGINE_DISTRIBUTION
    ]
    if not engines:
        return f"the plugin environment holds no {ENGINE_DISTRIBUTION}"
    if engines[0].version != __version__:
        return (
            f"the plugin environment holds {ENGINE_DISTRIBUTION} {engines[0].version}, "
            f"not the runner's {__version__}"
        )
    return None


def _recorded_requirement(marker: Path, engine: str) -> str | None:
    """The plugin requirement a finished environment was built from, or ``None`` when the marker
    is missing, unreadable or names another engine (an unfinished or foreign directory)."""
    try:
        data: object = json.loads(marker.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(data, dict):
        return None
    requirements = cast("dict[str, object]", data).get("requirements")
    if not isinstance(requirements, list):
        return None
    items = cast("list[object]", requirements)
    if len(items) != 2 or items[0] != engine or not isinstance(items[1], str):
        return None
    return items[1]


def _lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


async def ensure_environment(
    pin: PluginPin,
    *,
    fetch_wheel: Callable[[], Awaitable[Path]] | None = None,
    root: Path | None = None,
    run: CommandRunner = _run,
    phase: Callable[[str], None] | None = None,
) -> PluginEnvironment:
    """The verified environment for ``pin``: reused when an earlier run built it and it still
    verifies, else (re)built.

    ``fetch_wheel`` downloads an ``hf`` source's wheel (with the run's credential); it is called
    only when a build is needed, and its file is hash-checked before the install sees it. Builds
    of one environment are serialised with a lock file, so two runs on one machine never install
    into the same directory at once.
    """
    engine = engine_requirement()
    key = environment_key(pin, engine)
    base = envs_dir() if root is None else root
    target = base / key
    python = target / "bin" / "python"
    scratch = Path(tempfile.mkdtemp(prefix="strata-plugin-install-"))
    env = install_environ(home=scratch / "home", pip_cache=base / ".pip-cache")
    (scratch / "home").mkdir()
    fd = await asyncio.to_thread(_lock, base / f"{key}.lock")
    try:
        marker = target / _MARKER
        recorded = _recorded_requirement(marker, engine) if python.exists() else None
        if recorded is not None:
            try:
                sites, packages = await _inspect(run, python, env)
            except PluginManifestError:
                sites, packages = None, ()
            if sites is not None and _verify(pin, sites) is None:
                return PluginEnvironment(
                    key=key,
                    root=target,
                    python=python,
                    site_paths=sites,
                    packages=packages,
                    requirements=(engine, recorded),
                    reused=True,
                )
        shutil.rmtree(target, ignore_errors=True)
        wheel: Path | None = None
        if pin.source.kind == "hf":
            if fetch_wheel is None or pin.source.wheel_sha256 is None:
                msg = "an hf plugin source needs its wheel downloaded and its sha256 pinned"
                raise PluginManifestError(msg)
            wheel = await fetch_wheel()
            await asyncio.to_thread(verify_wheel, wheel, pin.source.wheel_sha256)
        requirement = plugin_requirement(pin, wheel)
        if phase is not None:
            phase(f"installing {pin.distribution} {pin.version} into its own environment")
        await _checked(
            run, [sys.executable, "-m", "venv", str(target)], env, "creating the plugin environment"
        )
        await _checked(
            run,
            [str(python), "-I", "-m", "pip", "install", "--no-input", engine, requirement],
            env,
            f"installing {pin.distribution} {pin.version}",
        )
        sites, packages = await _inspect(run, python, env)
        problem = _verify(pin, sites)
        if problem is not None:
            shutil.rmtree(target, ignore_errors=True)
            raise PluginManifestError(problem)
        marker.write_text(
            json.dumps(
                {"key": key, "requirements": [engine, requirement], "packages": list(packages)}
            ),
            encoding="utf-8",
        )
        return PluginEnvironment(
            key=key,
            root=target,
            python=python,
            site_paths=sites,
            packages=packages,
            requirements=(engine, requirement),
            reused=False,
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        shutil.rmtree(scratch, ignore_errors=True)
