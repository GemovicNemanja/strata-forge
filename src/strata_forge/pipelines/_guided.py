"""Guided JSON for batch generation: the run's schema, its checks, and the one repair turn.

A run may declare ONE JSON Schema (``hyperparams.guided_json``). Every request then carries it as
the model server's structured-output constraint, so decoding can only produce text the schema's
grammar admits, and every output is still validated here, because the constraint is not the
whole schema: a reply cut off at ``max_tokens`` is not JSON at all, and a grammar backend does not
enforce every keyword a schema can state. An output that fails validation gets exactly ONE
repair turn, the original conversation plus the invalid reply and a fixed message naming what
was wrong, under the same constraint. The record says, per row, whether the final output is
valid and whether it took the repair, and the manifest counts both.

The schema is data from the wire, so it is held to three rules before any machine is touched:
it is a JSON object of bounded size, it is a valid schema for the draft it names (2020-12 when it
names none), and every ``$ref`` in it points inside the schema itself. The last rule matters
because ``jsonschema`` still RETRIEVES a remote reference by default (its deprecated
``_warn_for_remote_retrieve`` opens the URL); validators built here also get an empty
``referencing.Registry``, so nothing is ever fetched either way.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import referencing

from strata_forge.pipelines._chat_template import nests_deeper_than

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

__all__ = [
    "MAX_REPLY_DEPTH",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_JSON_BYTES",
    "REPAIR_PROMPT",
    "OutputValidator",
    "repair_message",
    "repair_prompt_digest",
    "schema_digest",
    "schema_problem",
]

#: The schema's size as canonical JSON. The whole run spec rides to the machine in one bounded
#: blob, and a schema is a contract, not a corpus.
MAX_SCHEMA_JSON_BYTES = 16 * 1024
#: How deep a schema's objects and arrays may nest. Every walk over a schema (the size measure,
#: the reference check, the draft check, validation) recurses, so the depth is bounded first,
#: iteratively, and a schema past it is refused rather than allowed to raise ``RecursionError``.
MAX_SCHEMA_DEPTH = 32
#: How deep a reply may nest before it is recorded invalid without being validated: validation
#: walks the reply recursively, and one row's reply must never end the run.
MAX_REPLY_DEPTH = 64
#: The one repair turn. ``{error}`` is the validator's message for the reply, capped.
REPAIR_PROMPT = (
    "Your previous reply does not satisfy the required JSON schema: {error}\n"
    "Reply again with only the corrected JSON."
)
# A validation message is quoted into the repair turn and the results; cap it.
_MAX_ERROR_CHARS = 300
_REF_KEYS = ("$ref", "$dynamicRef", "$recursiveRef")


def _canonical(schema: Mapping[str, JsonValue]) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def schema_digest(schema: Mapping[str, JsonValue]) -> str:
    """The sha256 of the schema as canonical JSON (keys sorted at every depth)."""
    return hashlib.sha256(_canonical(schema).encode()).hexdigest()


def repair_prompt_digest() -> str:
    """The sha256 of :data:`REPAIR_PROMPT`, so a manifest names the exact repair turn used."""
    return hashlib.sha256(REPAIR_PROMPT.encode()).hexdigest()


def _foreign_refs(value: object) -> list[str]:
    """Every reference in ``value`` that does not point inside the schema (``#…``)."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in cast("dict[str, object]", value).items():
            if key in _REF_KEYS and isinstance(item, str) and not item.startswith("#"):
                found.append(item)
            found.extend(_foreign_refs(item))
    elif isinstance(value, list):
        for item in cast("list[object]", value):
            found.extend(_foreign_refs(item))
    return found


def _validator_class(schema: Mapping[str, JsonValue]) -> Any:  # pyright: ignore[reportAny] - jsonschema ships no types
    return jsonschema.validators.validator_for(
        dict(schema), default=jsonschema.validators.Draft202012Validator
    )


def schema_problem(schema: Mapping[str, JsonValue]) -> str | None:
    """Why ``schema`` cannot be a run's guided JSON schema, or ``None`` when it can."""
    if not schema:
        return "guided_json is an empty schema, which admits anything; omit it instead"
    if nests_deeper_than(dict(schema), MAX_SCHEMA_DEPTH):
        return f"guided_json nests more than {MAX_SCHEMA_DEPTH} levels of objects and arrays"
    try:
        return _bounded_schema_problem(schema)
    except RecursionError:
        # Within the depth cap a schema cannot reach the interpreter's limit; this is the guard
        # that keeps a spec check from ever surfacing as a server error.
        return "guided_json is nested too deeply to check"


def _bounded_schema_problem(schema: Mapping[str, JsonValue]) -> str | None:
    if len(_canonical(schema).encode()) > MAX_SCHEMA_JSON_BYTES:
        return f"guided_json is at most {MAX_SCHEMA_JSON_BYTES} bytes of JSON"
    foreign = _foreign_refs(dict(schema))
    if foreign:
        return f"guided_json may reference only its own definitions (#...), not {foreign[:3]}"
    try:
        _validator_class(schema).check_schema(dict(schema))
    except jsonschema.exceptions.SchemaError as exc:
        return f"guided_json is not a valid JSON Schema: {str(exc.message)[:_MAX_ERROR_CHARS]}"
    return None


def repair_message(error: str) -> str:
    """The repair turn for a reply that failed with ``error``."""
    return REPAIR_PROMPT.format(error=error)


def _no_constant(name: str) -> object:
    # `json.loads` accepts NaN and Infinity, which are not JSON; a reader of the output would not.
    msg = f"{name} is not valid JSON"
    raise ValueError(msg)


class OutputValidator:
    """Checks a model's reply against the run's schema; never fetches a reference."""

    def __init__(self, schema: Mapping[str, JsonValue]) -> None:
        cls: Any = _validator_class(schema)
        self.draft: str = str(cls.__name__)
        # Empty: a reference that is not inside the schema resolves to nothing, never to a fetch.
        registry: referencing.Registry[Any] = referencing.Registry()
        self._validator: Any = cls(dict(schema), registry=registry)

    def error(self, text: str | None) -> str | None:
        """Why ``text`` is not a valid instance of the schema, or ``None`` when it is."""
        if text is None:
            return "the reply has no text"
        too_deep = f"the reply nests more than {MAX_REPLY_DEPTH} levels of objects and arrays"
        try:
            instance: object = json.loads(text, parse_constant=_no_constant)
        except RecursionError:
            return too_deep
        except ValueError as exc:
            return f"not valid JSON ({str(exc)[:_MAX_ERROR_CHARS]})"
        if nests_deeper_than(instance, MAX_REPLY_DEPTH):
            return too_deep
        try:
            errors: Any = self._validator.iter_errors(instance)
            # jsonschema ships no types; the error it returns is read for two plain attributes.
            best = cast("Any", jsonschema.exceptions.best_match)(errors)  # pyright: ignore[reportUnknownMemberType]
        except RecursionError:
            return "the reply is nested too deeply to validate against the schema"
        if best is None:
            return None
        parts: list[object] = list(best.absolute_path)
        message: str = str(best.message)
        path = "/".join(str(part) for part in parts)
        where = f" at /{path}" if path else ""
        return f"{message[:_MAX_ERROR_CHARS]}{where}"
