"""The arguments every model and tokenizer load in :mod:`strata_forge.training` passes.

A fine-tuning run downloads code-adjacent artifacts from a repo someone else controls, so each
``from_pretrained`` call states its terms instead of inheriting the library's defaults:

- **No remote code.** ``trust_remote_code=False``, explicitly: a repo's own Python never runs.
- **Safetensors weights only.** ``use_safetensors=True`` on every model load: a pickle checkpoint
  (``*.bin`` / ``*.pt``) can execute code when it is unpickled, so a repo that ships only those
  fails to load rather than loading unsafely.
- **The caller's credential.** The token travels as an argument of the load, never on a config
  object, so it cannot reach TRL's arguments or the ``training_args.bin`` TRL saves beside a
  checkpoint.
- **Nothing silently re-initialised.** :func:`check_loading_info` reads the report a load returns
  with ``output_loading_info=True`` and refuses one that left a weight at a random initial value.

Nothing here imports a heavy dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from strata_forge.core.errors import ForgeError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "HubToken",
    "LoadRefusedError",
    "check_loading_info",
    "model_load_kwargs",
    "tokenizer_load_kwargs",
]

# How many key names a refusal quotes: enough to see the pattern, few enough to stay a message.
_QUOTED_KEYS = 8


type HubToken = str | Literal[False] | None
"""A Hub credential as ``from_pretrained`` takes it.

A token string; ``False`` to send none at all; or ``None`` for the library's own lookup (an
``HF_TOKEN`` variable or a cached login on the machine).
"""


def tokenizer_load_kwargs(token: HubToken) -> dict[str, Any]:
    """Keyword arguments for ``AutoTokenizer.from_pretrained``: no remote code, this credential."""
    return {"trust_remote_code": False, "token": token}


def model_load_kwargs(token: HubToken) -> dict[str, Any]:
    """Keyword arguments for a model's ``from_pretrained``: no remote code, safetensors only."""
    return {**tokenizer_load_kwargs(token), "use_safetensors": True}


class LoadRefusedError(ForgeError):
    """A model load left weights uninitialised or reported errors, so its outputs mean nothing."""


def check_loading_info(info: Mapping[str, Any], *, head_prefixes: tuple[str, ...]) -> list[str]:
    """Refuse a load that did not fill every weight; return the checkpoint keys it ignored.

    ``info`` is the report ``from_pretrained(..., output_loading_info=True)`` returns. A MISSING
    key is a weight the checkpoint did not supply, so the library initialised it at random: a
    classifier loaded from a checkpoint without its head reads noise and still answers. A
    MISMATCHED key is a weight whose shape disagreed and was re-initialised. Both refuse, as do
    reported load errors.

    An UNEXPECTED key is a checkpoint tensor the model class has no slot for. Most are harmless
    (a pooler the class does not use, a stored ``position_ids`` buffer), so they are returned for
    the caller to record, and refused only when one starts with one of ``head_prefixes``: a head
    tensor the class did not take means the head it built is not the one the checkpoint trained.
    """
    missing = sorted(str(key) for key in info.get("missing_keys") or ())
    mismatched = sorted(_key_name(key) for key in info.get("mismatched_keys") or ())
    errors = [str(message) for message in info.get("error_msgs") or ()]
    unexpected = sorted(str(key) for key in info.get("unexpected_keys") or ())
    unexpected_head = [key for key in unexpected if key.startswith(head_prefixes)]
    problems = [
        f"{label}: {_quote(keys)}"
        for label, keys in (
            ("missing weights", missing),
            ("weights with a mismatched shape", mismatched),
            ("checkpoint head tensors the model did not load", unexpected_head),
            ("load errors", errors),
        )
        if keys
    ]
    if problems:
        msg = "the model did not load cleanly, so its outputs would be meaningless; " + "; ".join(
            problems
        )
        raise LoadRefusedError(msg)
    return unexpected


def _key_name(entry: object) -> str:
    # transformers reports a shape mismatch as (key, checkpoint shape, model shape).
    parts = cast("tuple[object, ...]", entry) if isinstance(entry, tuple) else (entry,)
    return str(parts[0]) if parts else ""


def _quote(keys: list[str]) -> str:
    shown = ", ".join(keys[:_QUOTED_KEYS])
    more = len(keys) - _QUOTED_KEYS
    return f"{shown} (and {more} more)" if more > 0 else shown
