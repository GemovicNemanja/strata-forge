"""Classifier credentials and policy codecs never borrow evaluation or ambient authentication."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from strata_forge.pipelines import _classifier_worker_policy as policy
from strata_forge.pipelines.classifier_runner import TrainingReadRequest
from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec, ClassifierWorkerError
from tests.unit.pipelines.test_classifier_spec import payload

# Malformed wire fixtures deliberately contain objects outside the production schemas.


def frame(**changes: Any) -> dict[str, Any]:
    spec = ClassifierTrainSpec.model_validate(payload())
    return {
        "version": 1,
        "purpose": "classifier_training",
        "run_id": spec.run_id,
        "spec_sha256": spec.digest(),
        "policy_origin": "https://policy.example",
        "worker_credential": "strata_worker_" + "x" * 43,
        **changes,
    }


def authority(**changes: Any) -> policy.ClassifierWorkerAuthority:
    spec = ClassifierTrainSpec.model_validate(payload())
    return policy.parse_classifier_authority(
        SecretStr(json.dumps(frame(**changes))), run_id=spec.run_id, spec_sha256=spec.digest()
    )


def request() -> TrainingReadRequest:
    spec = ClassifierTrainSpec.model_validate(payload())
    return TrainingReadRequest(
        run_id=spec.run_id, spec_sha256=spec.digest(), ref=spec.base.ref, repo_type="model"
    )


def accepted(**changes: Any) -> dict[str, Any]:
    return {
        **request().model_dump(),
        "authorized": True,
        "custody": "unsealed",
        "policy_version": "fictional-training-v1",
        **changes,
    }


def reply(
    value: Any, *, status: int = 200, raw: bytes | None = None, **headers: str
) -> httpx.Response:
    body = json.dumps(value).encode() if raw is None else raw
    return httpx.Response(
        status,
        stream=httpx.ByteStream(body),
        headers={"content-type": "application/json", **headers},
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"purpose": "evaluation"},
        {"run_id": "b" * 32},
        {"spec_sha256": "c" * 64},
        {"attempt_id": "d" * 32},
        {"worker_credential": "hf_fictional_private_canary"},
        {"policy_origin": "http://localhost:8000"},
        {"policy_origin": "http://policy.example"},
        {"policy_origin": "https://user:fictional-secret@policy.example"},
        {"policy_origin": "https://policy.example/path"},
        {"policy_origin": "https://policy.example?secret=canary"},
        {"policy_origin": "https://policy.example#canary"},
        {"policy_origin": "https://policy.example:0"},
    ],
)
def test_frame_refuses_wrong_domain_binding_or_origin_without_echo(changes: dict[str, Any]) -> None:
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_authority_invalid$") as error:
        authority(**changes)
    assert "fictional-secret" not in str(error.value)


@pytest.mark.parametrize(
    "origin", ["https://policy.example", "http://127.0.0.1:8144", "http://[::1]:8144"]
)
def test_frame_accepts_only_operator_https_or_literal_loopback(origin: str) -> None:
    value = authority(policy_origin=origin)
    assert value.policy_origin == origin
    assert frame()["worker_credential"] not in repr(value)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[]",
        b'{"version":1,"version":1}',
        b"x" * (policy.AUTHORITY_MAX_BYTES + 1),
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_frame_rejects_incomplete_duplicate_or_oversized_json(raw: bytes) -> None:
    spec = ClassifierTrainSpec.model_validate(payload())
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_authority_invalid$"):
        policy.parse_classifier_authority(
            SecretStr(raw.decode()), run_id=spec.run_id, spec_sha256=spec.digest()
        )


async def test_exact_fixed_read_request_bypasses_client_cookies_default_auth_and_parameters() -> (
    None
):
    seen = []

    def transport(actual: httpx.Request) -> httpx.Response:
        seen.append(actual)
        assert (
            str(actual.url)
            == "https://policy.example/api/worker-runs/"
            + request().run_id
            + "/classifier-training/reads"
        )
        assert actual.method == "POST"
        assert json.loads(actual.content) == request().model_dump()
        assert actual.headers["authorization"] == "Bearer " + frame()["worker_credential"]
        assert "cookie" not in actual.headers
        return reply(accepted())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transport),
        trust_env=False,
        cookies={"ambient": "private"},
        auth=("ambient", "secret"),
        params={"ambient": "private"},
    ) as client:
        async with policy.ClassifierPolicyRPC(authority(), client=client) as rpc:
            result = await rpc.decide(request())
            assert result.authorized
        assert not client.is_closed
    assert len(seen) == 1


@pytest.mark.parametrize("status", [301, 401, 403, 422, 500])
async def test_denial_and_redirect_never_retry(status: int) -> None:
    seen = []

    def transport(actual: httpx.Request) -> httpx.Response:
        seen.append(actual)
        return reply({}, status=status, location="https://other.example/private")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False) as client,
        policy.ClassifierPolicyRPC(authority(), client=client) as rpc,
    ):
        with pytest.raises(ClassifierWorkerError, match=r"^classifier_read_denied$"):
            await rpc.decide(request())
    assert len(seen) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"purpose": "evaluation"},
        {"run_id": "e" * 32},
        {"spec_sha256": "f" * 64},
        {"ref": "fixture/other@" + "1" * 40},
        {"repo_type": "dataset"},
        {"authorized": False},
        {"authorized": 1},
        {"custody": "locked"},
        {"private_rows": ["private canary"]},
    ],
)
async def test_closed_reply_must_match_every_request_field(changes: dict[str, Any]) -> None:
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: reply(accepted(**changes))),
            trust_env=False,
        ) as client,
        policy.ClassifierPolicyRPC(authority(), client=client) as rpc,
    ):
        with pytest.raises(ClassifierWorkerError, match=r"^classifier_read_denied$"):
            await rpc.decide(request())


@pytest.mark.parametrize(
    "wire",
    [
        b"x" * (policy.POLICY_MAX_BYTES + 1),
        b'{"version":1,"version":1}',
        b'{"version":NaN}',
        b"[]",
    ],
)
async def test_response_byte_and_json_bounds_precede_delivery(wire: bytes) -> None:
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: reply(None, raw=wire)),
            trust_env=False,
        ) as client,
        policy.ClassifierPolicyRPC(authority(), client=client) as rpc,
    ):
        with pytest.raises(ClassifierWorkerError, match=r"^classifier_read_denied$"):
            await rpc.decide(request())


async def test_only_identical_transient_requests_retry_at_most_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "_RETRY_BACKOFF", (0, 0))
    bodies = []

    def transport(actual: httpx.Request) -> httpx.Response:
        bodies.append(actual.content)
        return reply({}, status=503) if len(bodies) < 3 else reply(accepted())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False) as client,
        policy.ClassifierPolicyRPC(authority(), client=client) as rpc,
    ):
        assert (await rpc.decide(request())).authorized
    assert len(bodies) == 3
    assert len(set(bodies)) == 1


async def test_mismatched_request_refuses_before_http() -> None:
    calls = []
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: calls.append(r) or reply(accepted())),
            trust_env=False,
        ) as client,
        policy.ClassifierPolicyRPC(authority(), client=client) as rpc,
    ):
        altered = request().model_copy(update={"run_id": "e" * 32})
        with pytest.raises(ClassifierWorkerError, match=r"^classifier_read_denied$"):
            await rpc.decide(altered)
    assert not calls


async def test_repeated_cancel_drains_actual_close_before_propagation() -> None:
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close() -> None:
        entered.set()
        await release.wait()
        closed.set()

    task = asyncio.create_task(policy._close_owned(close))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


@pytest.mark.parametrize(
    "fault", [RuntimeError, SystemExit, KeyboardInterrupt, asyncio.CancelledError]
)
async def test_repeated_cancel_wins_after_close_fault_or_control_is_physically_contained(
    fault: type[BaseException],
) -> None:
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close() -> None:
        entered.set()
        await release.wait()
        finished.set()
        raise fault("fictional private close failure")

    task = asyncio.create_task(policy._close_owned(close))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_ordinary_close_failure_remains_an_error_without_caller_cancellation() -> None:
    async def close() -> None:
        raise RuntimeError("fictional close failure")

    with pytest.raises(RuntimeError, match=r"^fictional close failure$"):
        await policy._close_owned(close)
