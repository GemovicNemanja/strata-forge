"""Pure ASCII Hub identifier checks for immutable worker input boundaries."""

from __future__ import annotations

import re

_REPO = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)


def validate_hub_repo(value: str) -> str:
    """Validate a namespaced ASCII Hub ID without importing or contacting its SDK."""
    if (
        len(value) > 215
        or not _REPO.fullmatch(value)
        or ".." in value
        or "--" in value
        or value.endswith(".git")
    ):
        raise ValueError("hub_repo_id_invalid")
    return value


def validate_hub_ref(value: str) -> str:
    """Validate a bounded canonical namespace/repo@full-SHA reference."""
    repo, separator, sha = value.rpartition("@")
    if not separator or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("hub_ref_invalid")
    validate_hub_repo(repo)
    return value
