"""The plugin's sandbox: the strongest one the machine has, refused rather than weakened, and a
bubblewrap command that binds nothing of the runner's."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.plugins import sandbox as sb

if TYPE_CHECKING:
    from collections.abc import Sequence

PATHS = sb.SandboxPaths(
    environment=Path("/home/u/.strata-forge/plugin-envs/k"),
    interpreter_home=Path("/home/u/.local/share/uv/python/cpython-3.14"),
    job=Path("/tmp/strata-plugin-x/job.json"),
    inputs=Path("/tmp/strata-plugin-x/inputs"),
    output=Path("/tmp/strata-plugin-x/output"),
)
INNER = ["/home/u/.strata-forge/plugin-envs/k/bin/python", "-I", "-m", "strata_forge.plugins.host"]


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


def _choose(**kwargs: Any) -> sb.Sandbox:
    return asyncio.run(sb.choose_sandbox(**kwargs))


def test_another_uid_is_preferred_when_sudo_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux(monkeypatch, tools={"bwrap", "sudo", "setpriv"}, works={"sudo", "bwrap"})
    assert _choose(allow_unisolated=False) == sb.Sandbox("uid", 1)


def test_bubblewrap_alone_when_sudo_would_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux(monkeypatch, tools={"bwrap", "sudo", "setpriv"}, works={"bwrap"})
    assert _choose(allow_unisolated=False).isolation == "bwrap"


def test_no_sandbox_is_refused_unless_the_operator_allowed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _linux(monkeypatch, tools=set(), works=set())
    with pytest.raises(sb.SandboxUnavailableError, match="install bubblewrap"):
        _choose(allow_unisolated=False)
    assert _choose(allow_unisolated=True, environ={"PATH": "/bin"}).isolation == "none"


@pytest.mark.parametrize("scope", [0, None])
def test_unisolated_is_refused_where_any_process_can_ptrace(
    monkeypatch: pytest.MonkeyPatch, scope: int | None
) -> None:
    _linux(monkeypatch, tools=set(), works=set(), scope=scope)
    with pytest.raises(sb.SandboxUnavailableError, match="ptrace"):
        _choose(allow_unisolated=True, environ={"PATH": "/bin"})


def test_unisolated_is_refused_while_the_runner_s_environment_holds_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _linux(monkeypatch, tools=set(), works=set())
    with pytest.raises(sb.SandboxUnavailableError, match="GITHUB_TOKEN"):
        _choose(allow_unisolated=True, environ={"PATH": "/bin", "GITHUB_TOKEN": "x"})


def _flag_pairs(args: list[str], flag: str) -> list[tuple[str, str]]:
    return [(args[i + 1], args[i + 2]) for i, a in enumerate(args) if a == flag]


def test_bubblewrap_binds_nothing_of_the_runner_s_but_what_the_plugin_needs() -> None:
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
    assert readonly == {
        str(PATHS.environment),
        str(PATHS.job),
        str(PATHS.inputs),
        str(PATHS.interpreter_home),
    }
    assert _flag_pairs(args, "--bind") == [(str(PATHS.output), str(PATHS.output))]
    bound = {src for src, _ in _flag_pairs(args, "--ro-bind-try")}
    assert not any(
        b.startswith("/home")
        for b in bound | readonly - {str(PATHS.environment), str(PATHS.interpreter_home)}
    )
    env = dict(_flag_pairs(args, "--setenv"))
    assert env["HOME"] == "/tmp"
    assert env["HF_HOME"] == "/tmp/hf"
    assert "--dev-bind" not in args
    assert "setpriv" not in " ".join(args)


def test_the_uid_sandbox_drops_to_nobody_inside_bubblewrap() -> None:
    args = sb.sandbox_command(sb.Sandbox("uid", 2), INNER, paths=PATHS, env={}, gpu=False)
    assert Path(args[0]).name == "sudo"
    assert args[1] == "-n"
    tail = args[-len(INNER) - 5 : -len(INNER)]
    assert Path(tail[0]).name == "setpriv"
    assert tail[1:] == ["--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs"]


def test_no_sandbox_runs_the_host_directly() -> None:
    assert sb.sandbox_command(sb.Sandbox("none", 1), INNER, paths=PATHS, env={}, gpu=False) == INNER


def test_the_choice_is_recorded() -> None:
    assert sb.Sandbox("bwrap", 1).record() == {"isolation": "bwrap", "ptrace_scope": 1}
