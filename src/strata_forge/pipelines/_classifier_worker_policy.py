"""Fixed classifier-training policy RPC over its separate private Task channel."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from strata_forge.pipelines._hub_ids import validate_hub_ref
from strata_forge.pipelines.classifier_runner import TrainingReadDecision, TrainingReadRequest
from strata_forge.pipelines.classifier_spec import ClassifierWorkerError

AUTHORITY_MAX_BYTES = 8192
POLICY_MAX_BYTES = 16_384
POLICY_SECONDS = 30.0
_RETRY_BACKOFF = (0.25, 0.5)
_CREDENTIAL = re.compile(r"strata_worker_[A-Za-z0-9_-]{43}")

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def _close_owned(close: Callable[[], Awaitable[None]]) -> None:
    async def invoke() -> BaseException | None:
        try:
            await close()
        except BaseException as error:
            return error
        return None

    task = asyncio.create_task(invoke())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    error = task.result()
    if isinstance(error, Exception):
        raise error
    if error is not None:
        raise ClassifierWorkerError("classifier_policy_close_failed")


class ClassifierWorkerAuthority(BaseModel):
    """Private run/spec credential; it is never part of the inert training specification."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )
    version: Literal[1]
    purpose: Literal["classifier_training"]
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_origin: str = Field(min_length=1, max_length=2048)
    worker_credential: SecretStr

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("classifier_authority_invalid")
        return value

    @field_validator("worker_credential")
    @classmethod
    def _credential(cls, value: SecretStr) -> SecretStr:
        if _CREDENTIAL.fullmatch(value.get_secret_value()) is None:
            raise ValueError("classifier_authority_invalid")
        return value

    @field_validator("policy_origin")
    @classmethod
    def _origin(cls, value: str) -> str:
        parsed = None
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            port = 0
        if (
            parsed is None
            or not parsed.hostname
            or port == 0
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or not value.isascii()
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("classifier_authority_invalid")
        if parsed.scheme != "https":
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = False
            if parsed.scheme != "http" or not loopback:
                raise ValueError("classifier_authority_invalid")
        return value.rstrip("/")


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ValueError("classifier_policy_invalid")
        value[key] = item
    return value


def _constant(_: str) -> object:
    raise ValueError("classifier_policy_invalid")


def _json(raw: bytes, cap: int) -> object:
    if not raw or len(raw) > cap:
        raise ValueError("classifier_policy_invalid")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def parse_classifier_authority(
    value: SecretStr, *, run_id: str, spec_sha256: str
) -> ClassifierWorkerAuthority:
    """Bind the separate private frame before any source or model operation."""
    frame: ClassifierWorkerAuthority | None = None
    try:
        raw = value.get_secret_value().encode("utf-8")
        frame = ClassifierWorkerAuthority.model_validate(_json(raw, AUTHORITY_MAX_BYTES))
    except ValueError, TypeError, UnicodeError, RecursionError:
        frame = None
    if frame is None or frame.run_id != run_id or frame.spec_sha256 != spec_sha256:
        raise ClassifierWorkerError("classifier_authority_invalid")
    return frame


class ClassifierPolicyRPC:
    """Recheck current server policy without cookies, redirects or a cached positive grant."""

    def __init__(
        self, authority: ClassifierWorkerAuthority, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.authority = ClassifierWorkerAuthority.model_validate(authority.model_dump())
        if client is not None and (client.follow_redirects or client.trust_env):
            raise ClassifierWorkerError("classifier_authority_invalid")
        self._client = client or httpx.AsyncClient(trust_env=False, follow_redirects=False)
        self._owns_client = client is None
        self._closed = False

    async def __aenter__(self) -> Self:
        if self._closed:
            raise ClassifierWorkerError("classifier_read_denied")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._closed = True
        if self._owns_client:
            await _close_owned(self._client.aclose)

    async def decide(self, request: TrainingReadRequest) -> TrainingReadDecision:
        """Only identical transient requests retry inside one original thirty-second budget."""
        result: TrainingReadDecision | None = None
        try:
            current = TrainingReadRequest.model_validate(request.model_dump())
            validate_hub_ref(current.ref)
            if (
                self._closed
                or current.run_id != self.authority.run_id
                or current.spec_sha256 != self.authority.spec_sha256
                or current.purpose != self.authority.purpose
            ):
                raise ValueError("classifier_policy_invalid")
            payload = current.model_dump(mode="json")
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(body) > AUTHORITY_MAX_BYTES:
                raise ValueError("classifier_policy_invalid")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + POLICY_SECONDS
            async with asyncio.timeout_at(deadline):
                for attempt in range(3):
                    raw: bytes | None = None
                    with suppress(httpx.TimeoutException):
                        raw = await self._read(body, deadline - loop.time())
                    if raw is not None:
                        response = TrainingReadDecision.model_validate(_json(raw, POLICY_MAX_BYTES))
                        if (
                            response.authorized is not True
                            or response.custody != "unsealed"
                            or any(
                                getattr(response, key) != item
                                for key, item in current.model_dump().items()
                            )
                        ):
                            raise ValueError("classifier_policy_invalid")
                        result = response
                        break
                    if attempt < 2:
                        await asyncio.sleep(_RETRY_BACKOFF[attempt])
        except asyncio.CancelledError:
            raise
        except Exception:
            result = None
        if result is None:
            raise ClassifierWorkerError("classifier_read_denied")
        return result

    async def _read(self, body: bytes, remaining: float) -> bytes | None:
        if remaining <= 0:
            raise TimeoutError
        endpoint = (
            self.authority.policy_origin
            + "/api/worker-runs/"
            + self.authority.run_id
            + "/classifier-training/reads"
        )
        request = httpx.Request(
            "POST",
            endpoint,
            content=body,
            extensions={"timeout": httpx.Timeout(remaining).as_dict()},
            headers={
                "Authorization": "Bearer " + self.authority.worker_credential.get_secret_value(),
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        response = await self._client.send(request, stream=True, auth=None, follow_redirects=False)
        try:
            if response.status_code == 503:
                return None
            if response.status_code != 200:
                raise ValueError("classifier_policy_invalid")
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("classifier_policy_invalid")
            if (
                response.headers.get("content-type", "").split(";", 1)[0].lower()
                != "application/json"
            ):
                raise ValueError("classifier_policy_invalid")
            length = response.headers.get("content-length")
            if length is not None and (
                not length.isascii() or not length.isdecimal() or int(length) > POLICY_MAX_BYTES
            ):
                raise ValueError("classifier_policy_invalid")
            raw = bytearray()
            async for chunk in response.aiter_raw(chunk_size=POLICY_MAX_BYTES + 1):
                if len(raw) + len(chunk) > POLICY_MAX_BYTES:
                    raise ValueError("classifier_policy_invalid")
                raw.extend(chunk)
            return bytes(raw)
        finally:
            await _close_owned(response.aclose)
