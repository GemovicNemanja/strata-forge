"""``strata-forge doctor`` — environment, configuration, credential, and reachability checks.

The command always exits 0; it's a diagnostic, not a gate. Operators read the
output to confirm everything is wired up; CI can grep it.
"""

from __future__ import annotations

import os
import socket
from contextlib import suppress
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table

from strata_forge.config.settings import Settings
from strata_forge.core.repro import env_snapshot

__all__ = ["doctor"]


_PROBE_TIMEOUT_SECONDS = 1.0

# Provider → environment variables LiteLLM needs before a live call can work.
# Only presence is ever reported; values are never read into the output.
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "vertex": ("GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
}


def doctor() -> None:
    """Diagnose the environment, configuration, credentials, and declared services."""
    settings = Settings()
    snapshot = env_snapshot()
    console = Console()

    _render_header(console, snapshot)
    _render_settings(console, settings)
    _render_credentials(console)
    _render_packages(console, snapshot)
    _render_services(console, settings)


# ---------------------------------------------------------------------------
# Section renderers — each takes the Console and renders a chunk of the report.
# ---------------------------------------------------------------------------


def _render_header(console: Console, snapshot: dict[str, str | None]) -> None:
    console.print("\n[bold cyan]strata-forge doctor[/bold cyan]")
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


def _render_credentials(console: Console) -> None:
    """Report which provider credentials are present. Values are never printed."""
    table = Table(title="Provider credentials", title_justify="left")
    table.add_column("Provider")
    table.add_column("Environment variables")
    table.add_column("Status")

    for provider, variables in _PROVIDER_ENV_VARS.items():
        missing = [name for name in variables if not os.environ.get(name)]
        if not missing:
            status = "[green]set[/green]"
        elif len(missing) == len(variables):
            status = "[yellow]not set[/yellow]"
        else:
            status = "[yellow]partial[/yellow]"
        table.add_row(provider, ", ".join(variables), status)

    console.print(table)
    console.print(
        "[dim]Presence only: doctor reads whether each variable is defined, never its "
        "value, and a defined key is not proof it is valid. 'partial' means some but "
        "not all of the listed variables are defined.[/dim]"
    )
    console.print()


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
