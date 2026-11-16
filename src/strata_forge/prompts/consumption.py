"""Pinned local build consumption; descriptive source labels never authorize native reads."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from strata_forge.prompts.composition import PromptBuild, PromptRender, render_build
from strata_forge.prompts.composition_errors import PromptCompositionError

MAX_CONSUMED_BUILD_BYTES = 8 * 1024 * 1024
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)


class _Local(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class LocalBuildReference(_Local):
    """A caller-declared native locator, without a session, grant or software attestation."""

    project_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    surface_resource_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    resource_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    version_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    manifest_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")


class LocalBuildSource(_Local):
    """Explicit local content and partition assertions; no URL or remote loader is implied."""

    kind: Literal["caller_bytes"]
    partition: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    reference: LocalBuildReference | None


class ConsumedBuild(_Local):
    """Immutable local bytes with a verified content pin and descriptive source snapshot."""

    canonical: bytes = Field(repr=False)
    build_id: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=1, le=MAX_CONSUMED_BUILD_BYTES)
    source: LocalBuildSource

    @model_validator(mode="after")
    def exact_content(self) -> ConsumedBuild:
        if (
            len(self.canonical) != self.size_bytes
            or hashlib.sha256(self.canonical).hexdigest() != self.build_id
        ):
            raise PromptCompositionError("consumption_pin_mismatch")
        PromptBuild(canonical=self.canonical)
        return self


def _source(value: object) -> LocalBuildSource:
    result: LocalBuildSource | None = None
    if type(value) is LocalBuildSource:
        try:
            if (
                set(value.__dict__) == set(LocalBuildSource.model_fields)
                and not value.__pydantic_extra__
                and (value.reference is None or type(value.reference) is LocalBuildReference)
            ):
                result = LocalBuildSource.model_validate(value)
        except ValidationError, ValueError, TypeError:
            result = None
    if result is None:
        raise PromptCompositionError("consumption_source_invalid")
    return result


def consume_build(
    raw: bytes,
    *,
    expected_build_id: str,
    expected_size_bytes: int,
    source: LocalBuildSource,
) -> ConsumedBuild:
    """Validate exact caller-owned bytes and their mandatory pin, without fetching anything."""
    if (
        type(raw) is not bytes
        or type(expected_build_id) is not str
        or _SHA.fullmatch(expected_build_id) is None
        or type(expected_size_bytes) is not int
        or not 1 <= expected_size_bytes <= MAX_CONSUMED_BUILD_BYTES
    ):
        raise PromptCompositionError("consumption_invalid")
    checked_source = _source(source)
    if len(raw) != expected_size_bytes or hashlib.sha256(raw).hexdigest() != expected_build_id:
        raise PromptCompositionError("consumption_pin_mismatch")
    # ConsumedBuild calls the sole existing schema validator, without producer attestation.
    return ConsumedBuild(
        canonical=raw,
        build_id=expected_build_id,
        size_bytes=expected_size_bytes,
        source=checked_source,
    )


async def render_consumed(build: ConsumedBuild, values: object) -> PromptRender:
    """Use the installed modular renderer after rechecking the local immutable snapshot."""
    if (
        type(build) is not ConsumedBuild
        or set(build.__dict__) != set(ConsumedBuild.model_fields)
        or build.__pydantic_extra__
    ):
        raise PromptCompositionError("consumption_invalid")
    checked = consume_build(
        build.canonical,
        expected_build_id=build.build_id,
        expected_size_bytes=build.size_bytes,
        source=build.source,
    )
    return await render_build(PromptBuild(canonical=checked.canonical), values)
