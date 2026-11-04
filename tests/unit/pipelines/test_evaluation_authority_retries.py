"""Transient policy retries preserve exact authority, bounds and publication idempotency."""

from __future__ import annotations

import asyncio
from typing import cast

import httpx
import pytest

from strata_forge.pipelines import _evaluation_authority as authority_module
from strata_forge.pipelines._evaluation_access import EvaluationError
from strata_forge.pipelines._evaluation_authority import EvaluationPolicyRPC
from strata_forge.storage import CommitRef, HFHubClient
from tests.unit.pipelines.test_evaluation_authority import (
    COMMIT,
    DIGEST,
    REF,
    TOKEN,
    Hub,
    Server,
    frame,
)


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority_module, "_RPC_BACKOFF", (0.0, 0.0))


def test_default_retry_schedule_and_overall_budget_are_bounded() -> None:
    assert vars(authority_module)["_RPC_BACKOFF"] == (0.25, 0.5)
    assert vars(authority_module)["_RPC_SECONDS"] == 30.0


@pytest.mark.parametrize("failures", [1, 2])
async def test_503_recovery_reuses_exact_request_and_clears_each_cookie(
    fast_retry: None, failures: int
) -> None:
    server = Server()
    requests: list[tuple[str, bytes, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), request.content, request.headers["Authorization"]))
        result = server(request)
        if len(requests) <= failures:
            return httpx.Response(503, headers={"Set-Cookie": "session=never-send; Path=/"})
        return result

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            cookies={"session": "initial-cookie"},
            headers={"Cookie": "ambient=never-send"},
        ) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        decision = await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert decision.ref == REF
    assert len(requests) == failures + 1
    assert len(set(requests)) == 1


@pytest.mark.parametrize(
    "timeout_type", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout]
)
async def test_only_transport_timeouts_can_recover(
    fast_retry: None, timeout_type: type[httpx.TimeoutException]
) -> None:
    server = Server()
    requests: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.content)
        result = server(request)
        if len(requests) == 1:
            raise timeout_type(TOKEN, request=request)
        return result

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        assert (await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)).authorized
    assert len(requests) == 2
    assert requests[0] == requests[1]


@pytest.mark.parametrize("failure", ["503", "timeout"])
async def test_retry_exhaustion_never_sends_a_fourth_request_or_keeps_raw_context(
    fast_retry: None, failure: str
) -> None:
    server = Server()

    def handler(request: httpx.Request) -> httpx.Response:
        server(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(TOKEN, request=request)
        return httpx.Response(503, content=TOKEN.encode())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError) as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert len(server.requests) == 3
    assert str(caught.value) == (
        "authority_rpc_denied" if failure == "503" else "authority_rpc_failed"
    )
    assert caught.value.__context__ is None
    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize(
    "status", [301, 307, 308, 400, 401, 403, 404, 409, 422, 429, 500, 502, 504]
)
async def test_non_503_status_is_never_retried(fast_retry: None, status: int) -> None:
    server = Server()
    server.status = status
    server.headers = {"Location": "https://other.example", "Set-Cookie": "session=no"}
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError, match="authority_rpc_denied") as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert len(server.requests) == 1
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "error_type", [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError]
)
async def test_other_transport_failures_are_never_retried(
    fast_retry: None, error_type: type[httpx.RequestError]
) -> None:
    server = Server()

    def handler(request: httpx.Request) -> httpx.Response:
        server(request)
        raise error_type(TOKEN, request=request)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError, match="authority_rpc_failed") as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert len(server.requests) == 1
    assert caught.value.__context__ is None


@pytest.mark.parametrize("status", [307, 401, 403, 409, 422, 200])
async def test_close_timeout_cannot_make_denial_or_invalid_json_retryable(
    fast_retry: None, status: int
) -> None:
    server = Server()

    class ClosingTimeout(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"invalid " + TOKEN.encode()

        async def aclose(self) -> None:
            raise httpx.ReadTimeout(TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        server(request)
        return httpx.Response(status, stream=ClosingTimeout())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError) as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert len(server.requests) == 1
    assert caught.value.__context__ is None
    assert TOKEN not in str(caught.value)


async def test_timeout_while_reading_incomplete_200_response_can_retry(fast_retry: None) -> None:
    server = Server()

    class IncompleteReply(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"version":1,'
            raise httpx.ReadTimeout(TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        result = server(request)
        return (
            httpx.Response(200, stream=IncompleteReply()) if len(server.requests) == 1 else result
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        assert (await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)).authorized
    assert len(server.requests) == 2


@pytest.mark.parametrize("invalid", ["json", "binding", "oversized", "encoding", "denied"])
async def test_recovery_does_not_retry_an_invalid_or_denied_reply(
    fast_retry: None, invalid: str
) -> None:
    server = Server()

    def handler(request: httpx.Request) -> httpx.Response:
        if not server.requests:
            server(request)
            return httpx.Response(503)
        if invalid == "json":
            server.content = b"invalid " + TOKEN.encode()
        elif invalid == "binding":
            server.updates = {"run_id": "f" * 32}
        elif invalid == "oversized":
            server.content = b"x" * 16_385
        elif invalid == "encoding":
            server.headers = {"Content-Encoding": "gzip"}
        else:
            server.status = 403
        return server(request)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError) as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert len(server.requests) == 2
    assert TOKEN not in str(caught.value)
    assert caught.value.__context__ is None


async def test_total_budget_includes_backoff_without_renewing_per_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority_module, "_RPC_SECONDS", 0.01)
    server = Server()
    server.status = 503
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError, match="authority_rpc_failed") as caught:
            await asyncio.wait_for(
                policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST), timeout=1
            )
    assert len(server.requests) == 1
    assert caught.value.__context__ is None


async def test_total_budget_bounds_a_stalled_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority_module, "_RPC_SECONDS", 0.01)
    server = Server()

    async def handler(request: httpx.Request) -> httpx.Response:
        server(request)
        await asyncio.Event().wait()
        pytest.fail("the overall deadline must cancel the transport")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError, match="authority_rpc_failed") as caught:
            await asyncio.wait_for(
                policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST), timeout=1
            )
    assert len(server.requests) == 1
    assert caught.value.__context__ is None


@pytest.mark.parametrize("during", ["transport", "backoff"])
async def test_cancellation_propagates_without_any_retry(during: str) -> None:
    server, entered = Server(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        result = server(request)
        entered.set()
        if during == "transport":
            await asyncio.Event().wait()
        return httpx.Response(503) if during == "backoff" else result

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        task = asyncio.create_task(policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(server.requests) == 1


async def test_lost_receipt_reply_replays_exact_commit_without_repeating_hub_creation(
    fast_retry: None,
) -> None:
    server, hub = Server("create_only"), Hub()
    receipts: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        result = server(request)
        if request.url.path.endswith("/receipt"):
            receipts.append(request.content)
            if len(receipts) == 1:
                # The service accepted the exact receipt, but the worker lost its reply.
                raise httpx.ReadTimeout(TOKEN, request=request)
        return result

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(
            frame(),
            token="hf_synthetic",
            client=client,
            hub=cast("HFHubClient", cast("object", hub)),
        ) as policy,
        policy.publication(requested_repo_id="example/alias", spec_sha256=DIGEST) as fence,
    ):
        await policy.record_receipt(
            fence, CommitRef(sha=COMMIT, url="https://example.invalid/commit")
        )
    assert len(receipts) == 2
    assert receipts[0] == receipts[1]
    assert [path for path, _ in server.requests] == [
        "/output/plan",
        "/output/ready",
        "/output/receipt",
        "/output/receipt",
    ]
    assert [event[0] for event in hub.events] == ["create", "private", "parent"]
