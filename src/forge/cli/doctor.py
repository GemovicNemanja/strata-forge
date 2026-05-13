"""``forge doctor`` — environment, configuration, and service-reachability checks.

The command always exits 0; it's a diagnostic, not a gate. Operators read the
output to confirm everything is wired up; CI can grep it.
"""

from __future__ import annotations

import socket
from contextlib import suppress
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table

from forge.config.settings import Settings
from forge.core.repro import env_snapshot

__all__ = ["doctor"]


_PROBE_TIMEOUT_SECONDS = 1.0


def doctor() -> None:
    """Diagnose the environment, configuration, and reachability of declared services."""
    settings = Settings()
    snapshot = env_snapshot()
    console = Console()

    _render_header(console, snapshot)
    _render_settings(console, settings)
    _render_packages(console, snapshot)
    _render_services(console, settings)
    _render_pending_checks(console)


# ---------------------------------------------------------------------------
# Section renderers — each takes the Console and renders a chunk of the report.
# ---------------------------------------------------------------------------


def _render_header(console: Console, snapshot: dict[str, str | None]) -> None:
    console.print("\n[bold cyan]forge doctor[/bold cyan]")
    console.print(f"Python {snapshot.get('python')} on {snapshot.get('platform')}\n")


def _render_settings(console: Console, settings: Settings) -> None:
    table = Table(title="Settings", show_header=False, title_justify="left", box=None)
    table.add_row("profile", settings.profile)
    table.add_row("logging.level", settings.logging.level)
    table.add_row("logging.format", settings.logging.format)
    table.add_row("diagnostic.enabled", str(settings.diagnostic.enabled))
    table.add_row("diagnostic.path", settings.diagnostic.path)
    table.add_row("storage.default_backend", settings.storage.default_backend)
    console.print(table)
    console.print()


def _render_packages(console: Console, snapshot: dict[str, str | None]) -> None:
    table = Table(title="Tracked packages", title_justify="left")
    table.add_column("Package")
    table.add_column("Version")
    for key, value in sorted(snapshot.items()):
        if not key.startswith("pkg."):
            continue
        name = key.removeprefix("pkg.")
        rendered = value if value is not None else "[dim]not installed[/dim]"
        table.add_row(name, rendered)
    console.print(table)
    console.print()


def _render_services(console: Console, settings: Settings) -> None:
    table = Table(title="Service reachability", title_justify="left")
    table.add_column("Service")
    table.add_column("URL")
    table.add_column("Status")

    rows: list[tuple[str, str, str]] = []

    if settings.langfuse.enabled:
        ok = _probe_url(settings.langfuse.host)
        rows.append(("Langfuse", settings.langfuse.host, _status(ok)))
    else:
        rows.append(("Langfuse", settings.langfuse.host, "[yellow]keys not set[/yellow]"))

    ok = _probe_url(settings.redis.url)
    rows.append(("Redis", settings.redis.url, _status(ok)))

    ok = _probe_url(settings.qdrant.url)
    rows.append(("Qdrant", settings.qdrant.url, _status(ok)))

    for service, url, status in rows:
        table.add_row(service, url, status)
    console.print(table)
    console.print()


def _render_pending_checks(console: Console) -> None:
    console.print(
        "[dim]Provider auth probes and model-registry consistency checks land "
        "alongside the LLM module.[/dim]"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(ok: bool) -> str:
    return "[green]reachable[/green]" if ok else "[red]unreachable[/red]"


def _probe_url(url: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Try a TCP connect to the host:port implied by ``url``.

    Returns ``True`` iff the connect succeeded inside ``timeout`` seconds.
    Any DNS / refused / unreachable / parse error returns ``False`` quietly.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with suppress(OSError), socket.create_connection((host, port), timeout=timeout):
        return True
    return False
