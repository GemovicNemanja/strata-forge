"""Private checkpoint-completeness checks shared by trained-artifact loading paths."""

from __future__ import annotations

import re
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError

MAX_AUDIT_KEYS = 32
MAX_KEY_CHARACTERS = 192
__all__ = [
    "HeadKeyDecision",
    "LoadingAudit",
    "ModelLoadingError",
    "PaddingPolicy",
    "_guard_loading_result",
]
_KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
type LoadingCode = Literal[
    "missing_weights",
    "unexpected_weights",
    "mismatched_weights",
    "load_errors",
    "conversion_errors",
    "unsupported_loading_info",
    "pretrained_load_failed",
]


def _valid_key(key: str) -> bool:
    return len(key) <= MAX_KEY_CHARACTERS and _KEY.fullmatch(key) is not None


class LoadingAudit(BaseModel):
    """Bounded parameter names and fixed decisions; upstream diagnostic text is excluded."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    status: Literal["accepted", "refused"]
    missing_keys: tuple[str, ...] = Field(default=(), max_length=MAX_AUDIT_KEYS)
    missing_count: int = Field(default=0, ge=0)
    missing_omitted: int = Field(default=0, ge=0)
    unexpected_keys: tuple[str, ...] = Field(default=(), max_length=MAX_AUDIT_KEYS)
    unexpected_count: int = Field(default=0, ge=0)
    unexpected_omitted: int = Field(default=0, ge=0)
    mismatched_keys: tuple[str, ...] = Field(default=(), max_length=MAX_AUDIT_KEYS)
    mismatched_count: int = Field(default=0, ge=0)
    mismatched_omitted: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    conversion_error_count: int = Field(default=0, ge=0)
    conversion_diagnostics: Literal["reported", "not_returned"] = "not_returned"
    codes: tuple[LoadingCode, ...] = Field(default=(), max_length=7)

    @field_validator("missing_keys", "unexpected_keys", "mismatched_keys")
    @classmethod
    def _parameter_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(names)) != len(names) or any(not _valid_key(name) for name in names):
            raise ValueError("loading audit requires unique bounded parameter names")
        return names

    @model_validator(mode="after")
    def _consistent_counts(self) -> LoadingAudit:
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("loading audit refusal codes must be unique")
        if (
            self.missing_count < len(self.missing_keys)
            or self.unexpected_count < len(self.unexpected_keys)
            or self.mismatched_count < len(self.mismatched_keys)
        ):
            raise ValueError("loading audit names exceed their recorded counts")
        for category in ("missing", "unexpected", "mismatched"):
            omitted = f"{category}_omitted"
            expected = getattr(self, f"{category}_count") - len(getattr(self, f"{category}_keys"))
            if omitted in self.model_fields_set and getattr(self, omitted) != expected:
                raise ValueError("loading audit omission counts must match bounded names")
            object.__setattr__(self, omitted, expected)
        if (self.status == "accepted") != (not self.codes):
            raise ValueError("accepted loading audits cannot contain refusal codes")
        if self.status == "accepted" and any(
            (
                self.missing_count,
                self.unexpected_count,
                self.mismatched_count,
                self.error_count,
                self.conversion_error_count,
            )
        ):
            raise ValueError("trained-artifact loading cannot accept structural diagnostics")
        for count, code in (
            (self.missing_count, "missing_weights"),
            (self.unexpected_count, "unexpected_weights"),
            (self.mismatched_count, "mismatched_weights"),
            (self.error_count, "load_errors"),
            (self.conversion_error_count, "conversion_errors"),
        ):
            if (count > 0) != (code in self.codes):
                raise ValueError("loading audit codes must match structural counts")
        return self

    @classmethod
    def refusal(
        cls, code: Literal["unsupported_loading_info", "pretrained_load_failed"]
    ) -> LoadingAudit:
        """Construct a fixed-code refusal when upstream provided no usable loading record."""
        return cls(status="refused", codes=(code,))


class PaddingPolicy(BaseModel):
    """The actual tokenizer padding and model-config alignment used by a scorer."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    source: Literal["tokenizer_pad", "eos_fallback"]
    padding_side: Literal["right"] = "right"
    pad_token_id: int = Field(ge=0)
    original_model_pad_token_id: int | None
    model_config_aligned: Literal[True] = True


class HeadKeyDecision(BaseModel):
    """Exact key decision for a future trusted initializer; never an inference exemption."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    initializer_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    head_kind: Literal["encoder", "decoder"]
    created_keys: tuple[str, ...] = Field(min_length=1, max_length=MAX_AUDIT_KEYS)
    removed_keys: tuple[str, ...] = Field(default=(), max_length=MAX_AUDIT_KEYS)

    @field_validator("created_keys", "removed_keys")
    @classmethod
    def _exact_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(names)) != len(names) or any(not _valid_key(name) for name in names):
            raise ValueError("head keys must be exact unique parameter names, without wildcards")
        return names

    @model_validator(mode="after")
    def _trusted_contract(self) -> HeadKeyDecision:
        if self.head_kind == "decoder" and self.initializer_id != "decoder_logit_head":
            raise ValueError("decoder head initialization requires decoder_logit_head")
        if set(self.created_keys) & set(self.removed_keys):
            raise ValueError("created and removed keys must be disjoint")
        return self

    def accepts(self, audit: LoadingAudit) -> bool:
        """Compare complete observed key sets; execution still needs a trusted init receipt."""
        return (
            set(audit.codes) <= {"missing_weights", "unexpected_weights"}
            and audit.missing_count == len(audit.missing_keys)
            and audit.unexpected_count == len(audit.unexpected_keys)
            and set(audit.missing_keys) == set(self.created_keys)
            and set(audit.unexpected_keys) == set(self.removed_keys)
            and not (audit.mismatched_count or audit.error_count or audit.conversion_error_count)
        )


class ModelLoadingError(ValidationError):
    """A checkpoint was refused, with only a bounded safe audit available to the caller."""

    def __init__(self, audit: LoadingAudit) -> None:
        self.audit = audit
        names = (*audit.missing_keys, *audit.unexpected_keys, *audit.mismatched_keys)[:4]
        suffix = f"; parameter keys: {', '.join(names)}" if names else ""
        super().__init__(f"model_loading_refused: {', '.join(audit.codes)}{suffix}")


def _keys(raw: object, *, mismatched: bool = False) -> tuple[str, ...] | None:
    if type(raw) not in (list, tuple, set, frozenset):
        return None
    names: list[str] = []
    for item in cast("list[object]", raw):
        if (
            mismatched
            and type(item) in (list, tuple)
            and len(cast("tuple[object, ...]", item)) == 3
        ):
            item = cast("tuple[object, ...]", item)[0]
        if type(item) is not str or not _valid_key(item):
            return None
        names.append(item)
    if len(set(names)) != len(names):
        return None
    return tuple(sorted(names))


def _audit_loading_info(raw: object) -> LoadingAudit:
    required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    # Loader diagnostics are plain containers; a custom mapping can execute methods or
    # expose arbitrary exception text while validation attempts to inspect it.
    if type(raw) is not dict:
        return LoadingAudit.refusal("unsupported_loading_info")
    info = cast("dict[object, object]", raw)
    if (
        any(type(key) is not str for key in info)
        or not required <= info.keys()
        or info.keys() - required - {"conversion_errors"}
    ):
        return LoadingAudit.refusal("unsupported_loading_info")
    missing = _keys(info["missing_keys"])
    unexpected = _keys(info["unexpected_keys"])
    mismatched = _keys(info["mismatched_keys"], mismatched=True)
    errors = info["error_msgs"]
    conversions = info.get("conversion_errors", {})
    if (
        missing is None
        or unexpected is None
        or mismatched is None
        or type(errors) not in (list, tuple)
        or not all(type(item) is str for item in cast("list[object]", errors))
        or type(conversions) is not dict
    ):
        return LoadingAudit.refusal("unsupported_loading_info")
    codes: list[LoadingCode] = []
    error_count = len(cast("list[object]", errors))
    conversion_count = len(cast("dict[object, object]", conversions))
    for present, code in (
        (bool(missing), "missing_weights"),
        (bool(unexpected), "unexpected_weights"),
        (bool(mismatched), "mismatched_weights"),
        (error_count > 0, "load_errors"),
        (conversion_count > 0, "conversion_errors"),
    ):
        if present:
            codes.append(cast("LoadingCode", code))
    return LoadingAudit(
        status="refused" if codes else "accepted",
        missing_keys=missing[:MAX_AUDIT_KEYS],
        missing_count=len(missing),
        unexpected_keys=unexpected[:MAX_AUDIT_KEYS],
        unexpected_count=len(unexpected),
        mismatched_keys=mismatched[:MAX_AUDIT_KEYS],
        mismatched_count=len(mismatched),
        error_count=error_count,
        conversion_error_count=conversion_count,
        conversion_diagnostics="reported" if "conversion_errors" in info else "not_returned",
        codes=tuple(codes),
    )


def _guard_loading_result(result: object) -> tuple[object, LoadingAudit]:
    if type(result) is not tuple or len(cast("tuple[object, ...]", result)) != 2:
        raise ModelLoadingError(LoadingAudit.refusal("unsupported_loading_info"))
    model, info = cast("tuple[object, object]", result)
    audit = _audit_loading_info(info)
    if model is None or audit.status != "accepted":
        raise ModelLoadingError(
            audit if audit.status == "refused" else LoadingAudit.refusal("unsupported_loading_info")
        )
    return model, audit
