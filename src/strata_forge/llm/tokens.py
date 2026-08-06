"""Token-count estimation helpers.

Uses ``tiktoken`` (OpenAI's BPE tokenizer) as the primary backend — it's
already installed transitively via the OpenAI / LiteLLM stack and produces
stable counts for OpenAI models. For Anthropic and Google models we fall
back to tiktoken's ``cl100k_base`` encoding as a close-enough approximation:
not exact, but local, fast, and adequate for pre-call budget checks.
Post-call exact counts come from the provider's response payload.

The tokenizer is loaded on first call (lazy) and cached, so importing this
module doesn't pay the tokenizer's startup cost.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from strata_forge.core.errors import RegistryError
from strata_forge.llm.registry import registry as _global_registry

if TYPE_CHECKING:
    import tiktoken

__all__ = ["count_tokens"]


# Encoding selection per OpenAI model family. The GPT-5 family (incl. 5.5)
# uses ``o200k_base``; older GPT-4 era models used ``cl100k_base``.
_OPENAI_ENCODING_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("gpt-5", "o200k_base"),
    ("gpt-4", "cl100k_base"),
)

# Used for anything we can't tag to a more specific encoding.
_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(name: str) -> tiktoken.Encoding:
    """Cached tiktoken encoder lookup. First call imports tiktoken."""
    import tiktoken

    return tiktoken.get_encoding(name)


def _count_with_encoding(text: str, encoding_name: str) -> int:
    if not text:
        return 0
    encoding = _get_encoding(encoding_name)
    return len(encoding.encode(text))


def _encoding_for_model(model: str | None) -> str:
    """Pick the encoding name based on the model's registry entry.

    Unknown models default to ``cl100k_base`` — a reasonable fallback that
    over-estimates slightly for newer OpenAI models but never crashes.
    """
    if model is None:
        return _DEFAULT_ENCODING
    try:
        entry = _global_registry.get(model)
    except RegistryError:
        return _DEFAULT_ENCODING

    if entry.vendor == "openai":
        for prefix, encoding_name in _OPENAI_ENCODING_BY_PREFIX:
            if entry.name.startswith(prefix):
                return encoding_name
        return "o200k_base"
    # Anthropic + Google: cl100k_base is a workable approximation. Operators
    # who need exact pre-call counts for those vendors should use the post-
    # call usage values from the response, or the vendor's tokenizer SDK.
    return _DEFAULT_ENCODING


def count_tokens(text: str, *, model: str | None = None) -> int:
    """Estimate the number of tokens in ``text`` for the given model.

    Args:
        text: The text to tokenize.
        model: Optional logical model name; selects the most-appropriate
            tokenizer. When ``None`` or unknown, falls back to a default
            encoding that approximates most modern LLM tokenizers.

    Returns:
        Token count. Returns 0 for an empty string.
    """
    return _count_with_encoding(text, _encoding_for_model(model))
