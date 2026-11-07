"""Bounded allowlisted value projections, never whole kwargs, credentials or HTTP bytes."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from strata_forge.core.errors import ForgeError
from strata_forge.prompts.composition_errors import PromptCompositionError
from strata_forge.prompts.portable import MAX_JSON_DEPTH, canonical_bytes
from strata_forge.tracing.prompt_provider_types import (
    MAX_PROVIDER_PROJECTION_BYTES,
    ProviderProjectionMetadata,
    unavailable_projection,
)

if TYPE_CHECKING:
    from strata_forge.prompts.portable import PortableValue
    from strata_forge.tracing.prompt_provider_types import ProviderMethod

CHAT_FIELDS = (
    "messages",
    "temperature",
    "max_tokens",
    "top_p",
    "tools",
    "response_format",
    "seed",
    "stop",
    "tool_choice",
    "parallel_tool_calls",
)
RESPONSES_FIELDS = (
    "input",
    "instructions",
    "tools",
    "temperature",
    "max_output_tokens",
    "top_p",
    "text",
    "reasoning",
    "include",
    "store",
    "parallel_tool_calls",
    "truncation",
)
RESPONSE_FIELDS = (
    "choices",
    "output",
    "usage",
    "model",
    "status",
    "error",
    "incomplete_details",
    "object",
)
MAX_PROJECTION_DEPTH = 32
MAX_TAGGED_DEPTH = 56
PROJECTION_FRAME_DEPTH_RESERVATION = 8
MAX_PROJECTION_NODES = 65_536
MAX_PROJECTION_MAP_FIELDS = 4096
MAX_PROJECTION_INTEGER_DIGITS = 1024


class ProviderValueError(ForgeError):
    """A fixed pure-codec refusal; the adapter must preserve the owning model call."""

    def __init__(self, code: Literal["projection_unsupported", "projection_limit"]) -> None:
        self.code = "projection_limit" if code == "projection_limit" else "projection_unsupported"
        super().__init__(self.code)


@dataclass(slots=True, repr=False)
class _Budget:
    used: int = 0
    nodes: int = 0
    ancestors: set[int] = field(default_factory=lambda: set[int]())

    def charge(self, size: int) -> None:
        self.used += size
        if self.used > MAX_PROVIDER_PROJECTION_BYTES:
            raise ProviderValueError("projection_limit")

    def node(self, depth: int) -> None:
        self.nodes += 1
        if depth > MAX_PROJECTION_DEPTH or self.nodes > MAX_PROJECTION_NODES:
            raise ProviderValueError("projection_limit")


def _quoted_size(value: str) -> int:
    if len(value) > MAX_PROVIDER_PROJECTION_BYTES:
        raise ProviderValueError("projection_limit")
    size = 2
    for char in value:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            raise ProviderValueError("projection_unsupported")
        if code in {34, 92, 8, 9, 10, 12, 13}:
            size += 2
        elif code < 32:
            size += 6
        elif code < 128:
            size += 1
        elif code < 2048:
            size += 2
        elif code < 65536:
            size += 3
        else:
            size += 4
        if size > MAX_PROVIDER_PROJECTION_BYTES:
            raise ProviderValueError("projection_limit")
    return size


def _tag(value: object, budget: _Budget, depth: int = 0, tagged_depth: int = 0) -> PortableValue:
    if tagged_depth > MAX_TAGGED_DEPTH:
        raise ProviderValueError("projection_limit")
    budget.node(depth)
    if value is None:
        budget.charge(len(b'["null"]'))
        return ["null"]
    if type(value) is bool:
        budget.charge(len(b'["bool",false]') if not value else len(b'["bool",true]'))
        return ["bool", value]
    if type(value) is str:
        budget.charge(len(b'["string",]') + _quoted_size(value))
        return ["string", value]
    if type(value) is int:
        if value.bit_length() > 3402:
            raise ProviderValueError("projection_limit")
        text = str(value)
        if len(text.lstrip("-")) > MAX_PROJECTION_INTEGER_DIGITS:
            raise ProviderValueError("projection_limit")
        budget.charge(len(b'["integer",]') + len(text) + 2)
        return ["integer", text]
    if type(value) is float:
        if not math.isfinite(value):
            raise ProviderValueError("projection_unsupported")
        text = struct.pack(">d", value).hex()
        budget.charge(len(b'["binary64",""]') + 16)
        return ["binary64", text]
    if type(value) not in (list, dict):
        raise ProviderValueError("projection_unsupported")
    identity = id(value)
    if identity in budget.ancestors:
        raise ProviderValueError("projection_unsupported")
    budget.ancestors.add(identity)
    try:
        if type(value) is list:
            items = cast("list[object]", value)
            if len(items) > MAX_PROJECTION_NODES:
                raise ProviderValueError("projection_limit")
            budget.charge(len(b'["array",[]]') + max(0, len(items) - 1))
            return ["array", [_tag(item, budget, depth + 1, tagged_depth + 2) for item in items]]
        mapping = cast("dict[object, object]", value)
        if len(mapping) > MAX_PROJECTION_MAP_FIELDS:
            raise ProviderValueError("projection_limit")
        budget.charge(len(b'["object",[]]') + max(0, len(mapping) - 1))
        entries: list[PortableValue] = []
        for key, item in mapping.items():
            if type(key) is not str:
                raise ProviderValueError("projection_unsupported")
            budget.node(depth + 1)
            budget.charge(_quoted_size(key) + 3)
            entries.append([key, _tag(item, budget, depth + 1, tagged_depth + 3)])
        return ["object", entries]
    finally:
        budget.ancestors.remove(identity)


def _encode(tree: PortableValue) -> bytes:
    encoded: bytes | None = None
    failure: Literal["projection_unsupported", "projection_limit"] | None = None
    try:
        if MAX_TAGGED_DEPTH + PROJECTION_FRAME_DEPTH_RESERVATION > MAX_JSON_DEPTH:
            raise ProviderValueError("projection_limit")
        encoded = canonical_bytes(tree)
    except PromptCompositionError as error:
        failure = (
            "projection_limit"
            if type(error) is PromptCompositionError
            and error.diagnostic.code in {"portable_depth_limit", "portable_bytes_limit"}
            else "projection_unsupported"
        )
    if failure is not None:
        raise ProviderValueError(failure)
    if encoded is None or len(encoded) > MAX_PROVIDER_PROJECTION_BYTES:
        raise ProviderValueError("projection_limit")
    return encoded


def encode_provider_value(value: object) -> bytes:
    """Encode one exact tagged value; this content-bearing pure helper issues no authority."""
    budget = _Budget()
    tree = _tag(value, budget)
    return _encode(tree)


def _mapping(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ProviderValueError("projection_unsupported")
    mapping = cast("dict[object, object]", value)
    if len(mapping) > MAX_PROJECTION_MAP_FIELDS:
        raise ProviderValueError("projection_limit")
    # Check keys before any selected lookup: a custom key's equality/hash is not our observer.
    for key in mapping:
        if type(key) is not str:
            raise ProviderValueError("projection_unsupported")
        _quoted_size(key)
    return cast("dict[str, object]", value)


def _project(
    method: ProviderMethod,
    values: object,
    names: tuple[str, ...],
    *,
    model: object = None,
    response: bool = False,
) -> ProviderProjectionMetadata:
    present: int | None = None
    omitted: int | None = None
    status: Literal["unsupported", "limit"] = "unsupported"
    encoded: bytes | None = None
    try:
        mapping = _mapping(values)
        present = sum(name in mapping for name in names)
        omitted = len(mapping) - present
        budget = _Budget()
        table: list[PortableValue] = []
        for name in names:
            table.append(
                [name, ["present", _tag(mapping[name], budget)] if name in mapping else ["absent"]]
            )
        payload: dict[str, PortableValue] = {
            "domain": "strata-provider-response-projection-v1"
            if response
            else "strata-provider-input-projection-v1",
            "profile": "forge-provider-client-input/v1",
            "method": method,
            "fields": table,
        }
        if not response:
            payload["provider_model_id"] = _tag(model, budget)
        encoded = _encode(payload)
    except ProviderValueError as error:
        status = "limit" if error.code == "projection_limit" else "unsupported"
    except Exception:
        # An observer implementation failure never becomes a provider-call error.
        status = "unsupported"
    if encoded is None:
        return ProviderProjectionMetadata(
            profile="forge-provider-client-input/v1",
            status=status,
            canonical_tagged_sha256=None,
            encoded_size_bytes=None,
            selected_present_fields_count=present,
            omitted_fields_count=omitted,
        )
    return ProviderProjectionMetadata(
        profile="forge-provider-client-input/v1",
        status="observed",
        canonical_tagged_sha256=hashlib.sha256(encoded).hexdigest(),
        encoded_size_bytes=len(encoded),
        selected_present_fields_count=present,
        omitted_fields_count=omitted,
    )


def project_chat(
    provider_model_id: object, messages: object, parameters: object
) -> ProviderProjectionMetadata:
    """Observe fixed fields without reading any unselected parameter value."""
    try:
        selected = _mapping(parameters)
    except ProviderValueError as error:
        return unavailable_projection(
            "limit" if error.code == "projection_limit" else "unsupported"
        )
    # Preserve the actual required argument if an impossible duplicate is passed by a direct caller.
    if "messages" in selected:
        return unavailable_projection("unsupported")
    fields: dict[str, object] = {"messages": messages}
    for name in CHAT_FIELDS[1:]:
        if name in selected:
            fields[name] = selected[name]
    result = _project("acompletion", fields, CHAT_FIELDS, model=provider_model_id)
    return ProviderProjectionMetadata.model_validate(
        {
            **result.model_dump(),
            "omitted_fields_count": len(selected)
            - sum(name in selected for name in CHAT_FIELDS[1:]),
        }
    )


def project_responses(provider_model_id: object, request: object) -> ProviderProjectionMetadata:
    return _project("aresponses", request, RESPONSES_FIELDS, model=provider_model_id)


def project_response(method: ProviderMethod, raw: object) -> ProviderProjectionMetadata:
    if type(method) is not str or method not in {"acompletion", "aresponses"}:
        return unavailable_projection("unsupported")
    if type(raw) is not dict:
        return unavailable_projection("opaque")
    return _project(method, cast("dict[str, object]", raw), RESPONSE_FIELDS, response=True)
