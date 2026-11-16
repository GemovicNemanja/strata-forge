"""Bounded structural diagnostics for portable modular prompt composition."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.prompts.template import PromptError


class PromptOrigin(BaseModel):
    """A content coordinate and one exact expansion occurrence, never prompt text."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    coordinate: str = Field(min_length=1, max_length=160, pattern=r"^[\x21-\x7e]+$")
    occurrence_path: tuple[Annotated[int, Field(ge=0, le=16_384)], ...] = Field(
        default=(), max_length=34
    )


class CompositionDiagnostic(BaseModel):
    """A stable code with bounded passage references and no caller input values."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    severity: Literal["error", "advisory"] = "error"
    origins: tuple[PromptOrigin, ...] = Field(default=(), max_length=16_384)


class PromptCompositionError(PromptError):
    """Refuse structurally invalid content using a closed diagnostic, not source values."""

    def __init__(self, code: str, origins: tuple[PromptOrigin, ...] = ()) -> None:
        self.diagnostic = CompositionDiagnostic(code=code, origins=origins)
        super().__init__(code)
