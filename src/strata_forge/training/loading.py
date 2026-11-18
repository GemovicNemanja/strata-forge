"""The arguments every model and tokenizer load in :mod:`strata_forge.training` passes.

A fine-tuning run downloads code-adjacent artifacts from a repo someone else controls, so each
``from_pretrained`` call states its terms instead of inheriting the library's defaults:

- **No remote code.** ``trust_remote_code=False``, explicitly: a repo's own Python never runs.
- **Safetensors weights only.** ``use_safetensors=True`` on every model load: a pickle checkpoint
  (``*.bin`` / ``*.pt``) can execute code when it is unpickled, so a repo that ships only those
  fails to load rather than loading unsafely.
- **A pinned revision, when there is one.** A full commit id reads exactly that commit.
- **The caller's credential.** The token travels as an argument of the load, never on a config
  object, so it cannot reach TRL's arguments or the ``training_args.bin`` TRL saves beside a
  checkpoint.

Nothing here imports a heavy dependency.
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = ["HubToken", "model_load_kwargs", "tokenizer_load_kwargs"]


type HubToken = str | Literal[False] | None
"""A Hub credential as ``from_pretrained`` takes it.

A token string; ``False`` to send none at all; or ``None`` for the library's own lookup (an
``HF_TOKEN`` variable or a cached login on the machine).
"""


def tokenizer_load_kwargs(token: HubToken, revision: str | None = None) -> dict[str, Any]:
    """Keyword arguments for ``AutoTokenizer.from_pretrained``: no remote code, this credential.

    ``revision`` pins the repo commit the load reads; ``None`` reads the default branch.
    """
    kwargs: dict[str, Any] = {"trust_remote_code": False, "token": token}
    if revision is not None:
        kwargs["revision"] = revision
    return kwargs


def model_load_kwargs(token: HubToken, revision: str | None = None) -> dict[str, Any]:
    """Keyword arguments for a model's ``from_pretrained``: no remote code, safetensors only."""
    return {**tokenizer_load_kwargs(token, revision), "use_safetensors": True}
