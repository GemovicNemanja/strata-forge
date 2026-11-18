"""The plugin's sandbox: the strongest one the machine has or none at all, and a bubblewrap
command that binds nothing of the runner's but read-only what the code needs."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from strata_forge.plugins import sandbox as sb

if TYPE_CHECKING:
    from collections.abc import Sequence

ENGINE = Path("/home/u/engine")
PLUGIN = Path("/home/u/.strata-forge/plugin-envs/k/site")
JOB = Path("/work/strata-plugin-x/job.json")
INPUTS = Path("/work/strata-plugin-x/inputs")
OUTPUT = Path("/work/strata-plugin-x/output")
PATHS = sb.SandboxPaths(
    readonly=(
        ENGINE,
        ENGINE / "lib/python3.14/site-packages",
        Path("/usr/lib/python3"),
        PLUGIN,
        JOB,
        INPUTS,
    ),
    writable=(OUTPUT,),
)
INNER = ["/home/u/engine/bin/python", "-I", "-m", "strata_forge.plugins.host"]


def _linux(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools: set[str],
    works: set[str],
    scope: int | None = 1,
) -> None:
    monkeypatch.setattr(sb.sys, "platform", "linux")

    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in tools else None

    monkeypatch.setattr(sb.shutil, "which", which)
    monkeypatch.setattr(sb, "ptrace_scope", lambda: scope)

    async def succeeds(args: Sequence[str]) -> bool:
        return ("sudo" if args[0].endswith("sudo") else "bwrap") in works

    monkeypatch.setattr(sb, "_succeeds", succeeds)


def _choose() -> sb.Sandbox:
    return asyncio.run(sb.choose_sandbox())


def test_another_uid_is_preferred_when_sudo_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux(monkeypatch, tools={"bwrap", "sudo", "setpriv"}, works={"sudo", "bwrap"})
    assert _choose() == sb.Sandbox("uid", 1)


def test_bubblewrap_alone_when_sudo_would_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux(monkeypatch, tools={"bwrap", "sudo", "setpriv"}, works={"bwrap"})
    assert _choose().isolation == "bwrap"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_a_machine_with_no_sandbox_runs_no_plugin(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    _linux(monkeypatch, tools=set(), works=set())
    monkeypatch.setattr(sb.sys, "platform", platform)
    with pytest.raises(sb.SandboxUnavailableError, match="install bubblewrap"):
        _choose()


def _flag_pairs(args: list[str], flag: str) -> list[tuple[str, str]]:
    return [(args[i + 1], args[i + 2]) for i, a in enumerate(args) if a == flag]


def test_bubblewrap_binds_nothing_of_the_runner_s_but_what_the_code_needs() -> None:
    args = sb.sandbox_command(
        sb.Sandbox("bwrap", 1),
        INNER,
        paths=PATHS,
        env={"PATH": "/usr/bin", "HOME": "/x"},
        gpu=False,
    )
    assert args[-len(INNER) :] == INNER
    for flag in (
        "--unshare-pid",
        "--unshare-net",
        "--die-with-parent",
        "--clearenv",
        "--new-session",
    ):
        assert flag in args
    readonly = {src for src, _ in _flag_pairs(args, "--ro-bind")}
    # The engine once (not again for a path inside it), nothing under a system directory again.
    assert readonly == {str(ENGINE), str(PLUGIN), str(JOB), str(INPUTS)}
    assert _flag_pairs(args, "--bind") == [(str(OUTPUT), str(OUTPUT))]
    assert not any(src.startswith("/home") for src, _ in _flag_pairs(args, "--ro-bind-try"))
    env = dict(_flag_pairs(args, "--setenv"))
    assert env["HOME"] == "/tmp"
    assert env["HF_HOME"] == "/tmp/hf"
    assert "--dev-bind" not in args
    assert "--share-net" not in args
    assert "setpriv" not in " ".join(args)


def test_every_directory_made_to_hold_a_bind_is_reachable_by_nobody() -> None:
    # As root, bubblewrap makes a bind's missing parents 0700, and the nobody uid could reach
    # nothing beneath them: each one is made explicitly, 0755, before anything inside it.
    args = sb.sandbox_command(sb.Sandbox("uid", 1), INNER, paths=PATHS, env={}, gpu=False)
    exists = {Path("/"), Path("/tmp"), *(Path(d) for d in sb._SYSTEM_DIRS)}  # pyright: ignore[reportPrivateUsage]
    made: list[str] = []
    for i, arg in enumerate(args):
        if arg == "--dir":
            assert args[i - 2 : i] == ["--perms", "0755"]
            target = Path(args[i + 1])
            made.append(str(target))
        elif arg in ("--ro-bind", "--bind"):
            target = Path(args[i + 2])
        else:
            continue
        assert target.parent in exists, f"{target} before its parent"
        exists.add(target)
    assert set(made) == {
        "/home",
        "/home/u",
        "/home/u/.strata-forge",
        "/home/u/.strata-forge/plugin-envs",
        "/home/u/.strata-forge/plugin-envs/k",
        "/work",
        "/work/strata-plugin-x",
    }


def test_the_uid_sandbox_drops_to_nobody_inside_bubblewrap() -> None:
    args = sb.sandbox_command(sb.Sandbox("uid", 2), INNER, paths=PATHS, env={}, gpu=False)
    assert Path(args[0]).name == "sudo"
    assert args[1] == "-n"
    tail = args[-len(INNER) - 5 : -len(INNER)]
    assert Path(tail[0]).name == "setpriv"
    assert tail[1:] == ["--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs"]
    # Still root, the supervisor ties the sandbox's life to the runner's, then drops to nobody.
    supervisor = args[-len(INNER) - 10 : -len(INNER) - 5]
    assert supervisor == [sys.executable, "-I", "-m", "strata_forge.plugins.supervisor", "--"]
    assert args.index("--") < args.index(sys.executable)


def test_the_engine_is_bound_read_only_from_its_own_installation() -> None:
    import sys

    found = sb.engine_paths()
    assert Path(sys.prefix).resolve() in found
    assert Path(sys.base_prefix).resolve() in found


def test_chmod_for_uid_opens_only_what_it_is_given(tmp_path: Path) -> None:
    readable, writable, private = tmp_path / "r", tmp_path / "w", tmp_path / "p"
    for d in (readable, writable, private):
        d.mkdir(mode=0o700)
        (d / "f").write_text("x")
        (d / "f").chmod(0o600)
    tool = readable / "tool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o700)
    sb.chmod_for_uid(readable=[readable], writable=[writable])
    assert readable.stat().st_mode & 0o777 == 0o755
    assert (readable / "f").stat().st_mode & 0o777 == 0o644
    assert tool.stat().st_mode & 0o777 == 0o755
    assert writable.stat().st_mode & 0o777 == 0o777
    assert (private / "f").stat().st_mode & 0o777 == 0o600


def test_the_choice_is_recorded() -> None:
    assert sb.Sandbox("bwrap", 1).record() == {"isolation": "bwrap", "ptrace_scope": 1}
