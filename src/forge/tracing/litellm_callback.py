"""Idempotent registration of LiteLLM's Langfuse callback.

LiteLLM ships built-in Langfuse integration: appending ``"langfuse"``
to :data:`litellm.success_callback` and :data:`litellm.failure_callback`
makes the runtime emit a Langfuse trace for every completion. The
callback reads its credentials from the standard ``LANGFUSE_*`` env
vars at call time.

This module wraps that wiring so applications can flip tracing on with
a single :func:`install_litellm_callback` call at startup. The
function is **idempotent** — calling it multiple times doesn't
duplicate the callback in the lists — and **silent-no-op when Langfuse
isn't configured**, so library code can call it unconditionally.

:func:`is_litellm_callback_installed` is the diagnostic counterpart;
``forge doctor`` uses it to surface the "I configured Langfuse but I'm
not seeing traces" failure mode.
"""

from __future__ import annotations

from typing import Any

from forge.config import get_settings

__all__ = [
    "LITELLM_CALLBACK_NAME",
    "install_litellm_callback",
    "is_litellm_callback_installed",
]


LITELLM_CALLBACK_NAME = "langfuse"
"""The exact string LiteLLM matches against in its callback lists.

Module-level so the diagnostic and the installer share the same source
of truth — if LiteLLM ever renames its built-in Langfuse callback, the
edit is one place.
"""


def _ensure(callback_list: list[Any], name: str) -> None:
    """Append ``name`` to ``callback_list`` exactly once."""
    if name not in callback_list:
        callback_list.append(name)


def install_litellm_callback() -> bool:
    """Wire LiteLLM's Langfuse callback on success + failure paths.

    Idempotent: calling multiple times leaves
    :data:`litellm.success_callback` / :data:`litellm.failure_callback`
    with exactly one ``"langfuse"`` entry each. Silently no-ops when
    Langfuse isn't configured (no public/secret key in
    :class:`forge.config.LangfuseConfig`).

    Returns:
        ``True`` when the callback is installed (whether by this call
        or already), ``False`` when Langfuse isn't configured and the
        function was a no-op.
    """
    config = get_settings().langfuse
    if not config.enabled:
        return False

    import litellm  # pyright: ignore[reportMissingTypeStubs]

    success_list: list[Any] = getattr(litellm, "success_callback", None) or []
    failure_list: list[Any] = getattr(litellm, "failure_callback", None) or []
    _ensure(success_list, LITELLM_CALLBACK_NAME)
    _ensure(failure_list, LITELLM_CALLBACK_NAME)

    # Reassign in case `getattr` returned None and we built a fresh list.
    litellm.success_callback = success_list  # pyright: ignore[reportAttributeAccessIssue]
    litellm.failure_callback = failure_list  # pyright: ignore[reportAttributeAccessIssue]
    return True


def is_litellm_callback_installed() -> bool:
    """Report whether the Langfuse callback is currently registered.

    Returns ``True`` only when ``"langfuse"`` appears in **both**
    :data:`litellm.success_callback` and
    :data:`litellm.failure_callback`. The asymmetric case (success but
    not failure, or vice versa) is a half-installed state — usually
    the result of someone mutating the lists by hand — and worth
    surfacing as "not installed" so the diagnostic isn't misleading.
    """
    import litellm  # pyright: ignore[reportMissingTypeStubs]

    success: list[Any] = getattr(litellm, "success_callback", None) or []
    failure: list[Any] = getattr(litellm, "failure_callback", None) or []
    return LITELLM_CALLBACK_NAME in success and LITELLM_CALLBACK_NAME in failure
