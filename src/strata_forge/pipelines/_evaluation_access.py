"""Trusted pin-bound access decisions for evaluation reads; absence always denies."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.core.errors import ValidationError

__all__ = ["EvaluationError", "ReadDecision", "ReadPolicy", "authorize_reads"]


class EvaluationError(ValidationError):
    """A content-free evaluation refusal code safe to surface through run telemetry."""


class ReadDecision(BaseModel):
    """A trusted resolver's decision bound to the exact immutable requested input."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )
    ref: str = Field(min_length=43, max_length=256)
    repo_type: Literal["model", "dataset"]
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    authorized: bool
    custody: Literal["unsealed", "sealed", "locked", "unknown"]
    policy_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")


class ReadPolicy(Protocol):
    """Resolve canonical, historical, marker and lineage facts outside the inert spec.

    Implementations are installed trusted code, never a spec/CLI callback or import path.
    Positive custody facts are monotone; missing metadata is unknown, not unsealed.
    """

    async def decide(
        self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
    ) -> ReadDecision: ...


async def authorize_reads(
    reads: tuple[tuple[str, Literal["model", "dataset"]], ...],
    policy: ReadPolicy | None,
    *,
    spec_sha256: str,
) -> tuple[ReadDecision, ...]:
    """Authorize all named inputs before any model, row or prediction materialization."""
    if policy is None:
        raise EvaluationError("custody_policy_unavailable")
    decisions: list[ReadDecision] = []
    for ref, repo_type in reads:
        try:
            decision = ReadDecision.model_validate(
                await policy.decide(ref, repo_type=repo_type, spec_sha256=spec_sha256)
            )
        except Exception:
            raise EvaluationError("custody_resolution_failed") from None
        if decision.ref != ref or decision.repo_type != repo_type:
            raise EvaluationError("custody_ref_mismatch")
        if decision.spec_sha256 != spec_sha256:
            raise EvaluationError("custody_spec_mismatch")
        if not decision.authorized or decision.custody != "unsealed":
            raise EvaluationError("custody_denied")
        decisions.append(decision)
    return tuple(decisions)
