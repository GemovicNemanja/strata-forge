"""Fixed run-scoped policy RPC over a private, bounded authority frame."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from strata_forge.pipelines import EVAL_AUTHORITY_SECRET
from strata_forge.pipelines._evaluation_access import EvaluationError, ReadDecision
from strata_forge.pipelines._hub_ids import validate_hub_ref, validate_hub_repo
from strata_forge.storage import HFHubClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from contextlib import AbstractAsyncContextManager

    from strata_forge.storage import CommitRef

AUTHORITY_SECRET = EVAL_AUTHORITY_SECRET
_PREFIX = "/api/internal/evaluation-policy"
_TOKEN = re.compile(r"strata_worker_[A-Za-z0-9_-]{43}")
_SHA = re.compile(r"[a-f0-9]{40}")
_MAX_RESPONSE = 16_384


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    version: Literal[1]

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("authority_version_invalid")
        return value

    @field_validator("authorized", "accepted", mode="before", check_fields=False)
    @classmethod
    def _true_bool(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("authority_flag_invalid")
        return value


class EvaluationAuthority(_Strict):
    """A masked credential frame delivered only through the authenticated Task secret file."""

    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_origin: str = Field(min_length=1, max_length=2048)
    worker_credential: SecretStr

    @field_validator("worker_credential")
    @classmethod
    def _credential(cls, value: SecretStr) -> SecretStr:
        if not _TOKEN.fullmatch(value.get_secret_value()):
            raise ValueError("authority_credential_invalid")
        return value

    @field_validator("policy_origin")
    @classmethod
    def _origin(cls, value: str) -> str:
        parsed = None
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            parsed = None
            port = None
        if (
            parsed is None
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or any(ord(c) <= 32 for c in value)
            or port == 0
        ):
            raise ValueError("authority_origin_invalid")
        if parsed.scheme == "https":
            return value.rstrip("/")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if parsed.scheme != "http" or not loopback:
            raise ValueError("authority_origin_invalid")
        return value.rstrip("/")


class _Bound(_Strict):
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_version: Literal["custody-v1"]


class _ReadResponse(_Bound):
    ref: str
    repo_type: Literal["model", "dataset"]
    authorized: Literal[True]
    custody: Literal["unsealed"]


class OutputTarget(BaseModel):
    """A dataset destination; never a caller authorization flag."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    repo_type: Literal["dataset"] = "dataset"
    repo_id: str

    @field_validator("repo_id")
    @classmethod
    def _repo(cls, value: str) -> str:
        return validate_hub_repo(value)


class ApprovedOutput(BaseModel):
    """The server's separately authorized existing or create-only plan."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    requested: OutputTarget
    canonical_id: str
    known_aliases: tuple[str, ...] = Field(default=(), max_length=16)
    mode: Literal["existing_private", "create_only"]
    expected_parent: str | None
    privacy: Literal["required_private"]

    @model_validator(mode="after")
    def _binding(self) -> ApprovedOutput:
        validate_hub_repo(self.canonical_id)
        for alias in self.known_aliases:
            validate_hub_repo(alias)
        if self.requested.repo_id.lower() not in {
            self.canonical_id.lower(),
            *(a.lower() for a in self.known_aliases),
        }:
            raise ValueError("publication_identity_invalid")
        if (self.expected_parent is None) != (self.mode == "create_only") or (
            self.expected_parent is not None and not _SHA.fullmatch(self.expected_parent)
        ):
            raise ValueError("publication_parent_invalid")
        return self


class _PlanResponse(_Bound):
    output: ApprovedOutput


class PublicationFence(_Bound):
    """Ready exact destination/parent under the server's active Run publication fence."""

    repo_type: Literal["dataset"]
    requested_id: str
    canonical_id: str
    mode: Literal["existing_private", "create_only"]
    expected_parent: str = Field(pattern=r"^[a-f0-9]{40}$")

    @field_validator("requested_id", "canonical_id")
    @classmethod
    def _repo(cls, value: str) -> str:
        return validate_hub_repo(value)


class _ReceiptResponse(_Bound):
    target: OutputTarget
    expected_parent: str = Field(pattern=r"^[a-f0-9]{40}$")
    commit_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    accepted: Literal[True]


class PublicationPolicy(Protocol):
    """Separate destination authority; read/lineage grants never authorize row writes."""

    async def plan(self, *, requested_repo_id: str, spec_sha256: str) -> ApprovedOutput: ...
    def publication(
        self, *, requested_repo_id: str, spec_sha256: str
    ) -> AbstractAsyncContextManager[PublicationFence]: ...
    async def record_receipt(self, fence: PublicationFence, commit: CommitRef) -> None: ...


def parse_authority(value: SecretStr, *, run_id: str, spec_sha256: str) -> EvaluationAuthority:
    """Parse a bounded private frame without retaining an unsafe parser exception context."""
    raw = value.get_secret_value()
    frame: EvaluationAuthority | None = None
    try:
        if len(raw.encode("utf-8")) <= 8192:
            frame = EvaluationAuthority.model_validate_json(raw)
    except ValueError, UnicodeError:
        frame = None
    if frame is None:
        raise EvaluationError("authority_frame_invalid")
    if frame.run_id != run_id or frame.spec_sha256 != spec_sha256:
        raise EvaluationError("authority_frame_mismatch")
    return frame


class EvaluationPolicyRPC:
    """A fixed installed adapter; requests never use cookies, redirects or caller-selected paths."""

    def __init__(
        self,
        authority: EvaluationAuthority,
        *,
        token: str | Literal[False],
        client: httpx.AsyncClient | None = None,
        hub: HFHubClient | None = None,
    ) -> None:
        self.authority = authority
        self._client = client
        self._owned_client = client is None
        self._hub = hub or HFHubClient(token=token)
        self._token = token
        self._ready: PublicationFence | None = None

    async def __aenter__(self) -> EvaluationPolicyRPC:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(20, connect=5, write=5, pool=5),
                follow_redirects=False,
                trust_env=False,
            )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owned_client and self._client is not None:
            await self._client.aclose()

    def _request(self, digest: str) -> dict[str, object]:
        if digest != self.authority.spec_sha256:
            raise EvaluationError("authority_spec_mismatch")
        return {"version": 1, "run_id": self.authority.run_id, "spec_sha256": digest}

    async def _post[Response: _Bound](
        self,
        path: Literal["/read", "/output/plan", "/output/ready", "/output/receipt"],
        body: dict[str, object],
        response_type: type[Response],
    ) -> Response:
        encoded = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > 4096 or self._client is None:
            raise EvaluationError("authority_request_invalid")
        self._client.cookies.clear()
        self._client.headers.pop("Cookie", None)
        result: Response | None = None
        error = "authority_rpc_failed"
        try:
            async with (
                asyncio.timeout(30),
                self._client.stream(
                    "POST",
                    self.authority.policy_origin + _PREFIX + path,
                    content=encoded,
                    headers={
                        "Authorization": "Bearer "
                        + self.authority.worker_credential.get_secret_value(),
                        "Content-Type": "application/json",
                        "Accept-Encoding": "identity",
                    },
                    follow_redirects=False,
                    auth=None,
                ) as response,
            ):
                if response.status_code != 200:
                    error = "authority_rpc_denied"
                elif response.headers.get("content-encoding", "identity") != "identity":
                    error = "authority_response_invalid"
                else:
                    data = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if len(data) + len(chunk) > _MAX_RESPONSE:
                            raise EvaluationError("authority_response_too_large")
                        data.extend(chunk)
                    result = response_type.model_validate_json(bytes(data))
        except Exception:
            result = None
        if result is None:
            raise EvaluationError(error)
        if (
            result.run_id != self.authority.run_id
            or result.spec_sha256 != self.authority.spec_sha256
        ):
            raise EvaluationError("authority_response_mismatch")
        return result

    async def decide(
        self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
    ) -> ReadDecision:
        validate_hub_ref(ref)
        result = await self._post(
            "/read",
            {**self._request(spec_sha256), "repo_type": repo_type, "ref": ref},
            _ReadResponse,
        )
        if result.ref != ref or result.repo_type != repo_type or result.authorized is not True:
            raise EvaluationError("authority_response_mismatch")
        return ReadDecision(
            ref=ref,
            repo_type=repo_type,
            spec_sha256=spec_sha256,
            authorized=True,
            custody="unsealed",
            policy_version=result.policy_version,
        )

    @asynccontextmanager
    async def publication(
        self, *, requested_repo_id: str, spec_sha256: str
    ) -> AsyncGenerator[PublicationFence]:
        if self._ready is not None:
            raise EvaluationError("publication_already_active")
        if self._token is False:
            raise EvaluationError("output_credential_required")
        plan = await self.plan(requested_repo_id=requested_repo_id, spec_sha256=spec_sha256)
        body = {**self._request(spec_sha256), "target": plan.requested.model_dump(mode="json")}
        if plan.mode == "create_only":
            failed = False
            try:
                await self._hub.create_repo(
                    plan.canonical_id, repo_type="dataset", private=True, exist_ok=False
                )
            except Exception:
                failed = True
            if failed:
                raise EvaluationError("publication_create_failed")
        ready = await self._post("/output/ready", body, PublicationFence)
        if (
            ready.requested_id != requested_repo_id
            or ready.canonical_id != plan.canonical_id
            or ready.mode != plan.mode
            or (plan.expected_parent is not None and ready.expected_parent != plan.expected_parent)
        ):
            raise EvaluationError("publication_identity_mismatch")
        current_private, current_parent = False, None
        try:
            current_private = await self._hub.repo_private(ready.canonical_id, repo_type="dataset")
            current_parent = await self._hub.resolve_revision(
                ready.canonical_id, repo_type="dataset"
            )
        except Exception:
            current_private, current_parent = False, None
        if current_private is not True or current_parent != ready.expected_parent:
            raise EvaluationError("publication_destination_changed")
        self._ready = ready
        try:
            yield ready
        finally:
            self._ready = None

    async def plan(self, *, requested_repo_id: str, spec_sha256: str) -> ApprovedOutput:
        target = OutputTarget(repo_id=requested_repo_id)
        body = {**self._request(spec_sha256), "target": target.model_dump(mode="json")}
        plan = (await self._post("/output/plan", body, _PlanResponse)).output
        if plan.requested != target:
            raise EvaluationError("publication_identity_mismatch")
        return plan

    async def record_receipt(self, fence: PublicationFence, commit: CommitRef) -> None:
        if fence != self._ready or fence.run_id != self.authority.run_id:
            raise EvaluationError("publication_receipt_unbound")
        target = OutputTarget(repo_id=fence.canonical_id)
        body = {
            **self._request(fence.spec_sha256),
            "target": target.model_dump(mode="json"),
            "expected_parent": fence.expected_parent,
            "commit_sha": commit.sha,
        }
        receipt = await self._post("/output/receipt", body, _ReceiptResponse)
        if (
            receipt.target != target
            or receipt.expected_parent != fence.expected_parent
            or receipt.commit_sha != commit.sha
            or receipt.accepted is not True
        ):
            raise EvaluationError("publication_receipt_mismatch")
