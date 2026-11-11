"""``strata-forge doctor`` — environment, configuration, and service-reachability checks.

Ordinary diagnostic failures report fixed statuses and exit 0. The explicit
offline mode observes package/configuration presence without service probes.
Neither mode verifies provider authentication or execution readiness.
"""

from __future__ import annotations

import re
import socket
from contextlib import suppress
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

from strata_forge.config.settings import Settings
from strata_forge.core.repro import (
    _TRACKED_PACKAGES,  # pyright: ignore[reportPrivateUsage] -- fixed diagnostic package list
    env_snapshot,
)

__all__ = ["doctor"]


_PROBE_TIMEOUT_SECONDS = 1.0
_VERSION = re.compile(
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+){0,3}(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?\Z"
)


def doctor(
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            "--presence-only",
            help="Report configuration and package presence without probing services.",
        ),
    ] = False,
) -> None:
    """Diagnose the environment, configuration, and reachability of declared services."""
    console = Console()
    try:
        settings = Settings()
    except Exception:
        settings = None
    try:
        snapshot = env_snapshot()
    except Exception:
        snapshot = None

    _render_header(console, snapshot or {})
    if settings is None:
        console.print("[yellow]Settings unavailable; review the configured value types.[/yellow]")
    else:
        _render_settings(console, settings)
    if snapshot is None:
        console.print("[yellow]Tracked packages unavailable; metadata could not be read.[/yellow]")
    else:
        _render_packages(console, snapshot)
    if settings is None:
        console.print(
            "[yellow]Service checks unavailable: configuration could not be read.[/yellow]"
        )
    else:
        _render_services(console, settings, offline=offline)
    _render_pending_checks(console)


# ---------------------------------------------------------------------------
# Section renderers — each takes the Console and renders a chunk of the report.
# ---------------------------------------------------------------------------


def _render_header(console: Console, snapshot: dict[str, str | None]) -> None:
    console.print("\n[bold cyan]strata-forge doctor[/bold cyan]")
    system = snapshot.get("system")
    label = system if system in {"Linux", "Darwin", "Windows"} else "other system"
    console.print(f"Python {_shown_version(snapshot.get('python')) or 'unknown'} on {label}\n")


def _render_settings(console: Console, settings: Settings) -> None:
    table = Table(title="Settings", show_header=False, title_justify="left", box=None)
    table.add_row("profile", _configured(settings.profile))
    table.add_row("logging.level", settings.logging.level)
    table.add_row("logging.format", settings.logging.format)
    table.add_row("diagnostic.enabled", str(settings.diagnostic.enabled))
    table.add_row("diagnostic.path", _configured(settings.diagnostic.path))
    table.add_row("storage.default_backend", settings.storage.default_backend)
    console.print(table)
    console.print()


def _render_packages(console: Console, snapshot: dict[str, str | None]) -> None:
    table = Table(title="Tracked packages", title_justify="left")
    table.add_column("Package")
    table.add_column("Version")
    for name in sorted(_TRACKED_PACKAGES):
        value = snapshot.get("pkg." + name)
        rendered = (
            "[dim]not installed[/dim]"
            if value is None
            else _shown_version(value) or "[yellow]present; version unavailable[/yellow]"
        )
        table.add_row(name, rendered)
    console.print(table)
    console.print()


def _render_services(console: Console, settings: Settings, *, offline: bool = False) -> None:
    table = Table(
        title="Service configuration (not probed)" if offline else "Service reachability",
        title_justify="left",
    )
    table.add_column("Service")
    table.add_column("Status")

    rows: list[tuple[str, str]] = []

    if settings.langfuse.enabled:
        state = (
            _presence(settings.langfuse.host)
            if offline
            else _status(_probe_url(settings.langfuse.host))
        )
        rows.append(("Langfuse", state))
    else:
        rows.append(("Langfuse", "[yellow]keys not set[/yellow]"))

    state = _presence(settings.redis.url) if offline else _status(_probe_url(settings.redis.url))
    rows.append(("Redis", state))

    state = _presence(settings.qdrant.url) if offline else _status(_probe_url(settings.qdrant.url))
    rows.append(("Qdrant", state))

    for service, status in rows:
        table.add_row(service, status)
    console.print(table)
    console.print()


def _render_pending_checks(console: Console) -> None:
    console.print(
        "[dim]Presence and TCP reachability do not verify provider authentication, "
        "model compatibility or execution readiness.[/dim]"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(ok: bool) -> str:
    return "[green]reachable[/green]" if ok else "[red]unreachable[/red]"


def _configured(value: str) -> str:
    return "configured" if value.strip() else "not configured"


def _presence(value: str) -> str:
    return _configured(value) + "; not probed"


def _shown_version(value: str | None) -> str | None:
    if value is None or len(value) > 160:
        return None
    release, separator, _ = value.partition("+")
    if _VERSION.fullmatch(release) is None:
        return None
    return release + (" (local build)" if separator else "")


def _probe_url(url: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Try a TCP connect to the host:port implied by ``url``.

    Returns ``True`` iff the connect succeeded inside ``timeout`` seconds.
    Any DNS / refused / unreachable / parse error returns ``False`` quietly.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
    if not host:
        return False
    with suppress(OSError), socket.create_connection((host, port), timeout=timeout):
        return True
    return False
