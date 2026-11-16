"""Fixed RPC protocol conformance with fake HTTP/Hub; no credentials or remote writes."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from strata_forge.pipelines._evaluation_access import EvaluationError
from strata_forge.pipelines._evaluation_authority import (
    ApprovedOutput,
    EvaluationAuthority,
    EvaluationPolicyRPC,
    parse_authority,
)
from strata_forge.storage import CommitRef, HFHubClient

RUN = "a" * 32
DIGEST = "b" * 64
PARENT = "c" * 40
COMMIT = "d" * 40
TOKEN = "strata_worker_" + "x" * 43
REF = "example/source@" + "e" * 40
BASE: dict[str, Any] = {"version": 1, "run_id": RUN, "spec_sha256": DIGEST}


def frame(**updates: Any) -> EvaluationAuthority:
    return EvaluationAuthority.model_validate(
        {**BASE, "policy_origin": "https://policy.example", "worker_credential": TOKEN, **updates}
    )


def plan(mode: str = "existing_private", **updates: Any) -> dict[str, Any]:
    return {
        "requested": {"repo_type": "dataset", "repo_id": "example/alias"},
        "canonical_id": "example/results",
        "known_aliases": ["example/alias"],
        "mode": mode,
        "expected_parent": None if mode == "create_only" else PARENT,
        "privacy": "required_private",
        **updates,
    }


def ready(**updates: Any) -> dict[str, Any]:
    return {
        **BASE,
        "policy_version": "custody-v1",
        "repo_type": "dataset",
        "requested_id": "example/alias",
        "canonical_id": "example/results",
        "mode": "existing_private",
        "expected_parent": PARENT,
        **updates,
    }


class Hub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object] | tuple[str, str, dict[str, Any]]] = []
        self.private: bool = True
        self.parent = PARENT
        self.create_error = False
        self.metadata_error = False

    async def create_repo(self, repo: str, **kwargs: Any) -> str:
        self.events.append(("create", repo, kwargs))
        if self.create_error:
            raise RuntimeError(TOKEN)
        return repo

    async def repo_private(self, repo: str, **kwargs: Any) -> bool:
        self.events.append(("private", repo))
        if self.metadata_error:
            raise RuntimeError(TOKEN)
        return self.private

    async def resolve_revision(self, repo: str, **kwargs: Any) -> str:
        self.events.append(("parent", repo))
        return self.parent


class Server:
    """Only echoes the agreed fixed protocol; tests mutate one bound field at a time."""

    def __init__(self, mode: str = "existing_private") -> None:
        self.mode = mode
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.updates: dict[str, Any] = {}
        self.status = 200
        self.content: bytes | None = None
        self.headers: dict[str, str] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.host == "policy.example"
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        assert "cookie" not in request.headers
        assert request.headers["Accept-Encoding"] == "identity"
        body = json.loads(request.content)
        assert {key: body[key] for key in BASE} == BASE
        path = request.url.path.removeprefix("/api/internal/evaluation-policy")
        self.requests.append((path, body))
        response: dict[str, Any] = {**BASE, "policy_version": "custody-v1"}
        if path == "/read":
            assert set(body) == {*BASE, "ref", "repo_type"}
            response.update(
                ref=body["ref"], repo_type=body["repo_type"], authorized=True, custody="unsealed"
            )
        elif path == "/output/plan":
            assert body["target"] == {"repo_type": "dataset", "repo_id": "example/alias"}
            response["output"] = plan(self.mode)
        elif path == "/output/ready":
            response.update(ready(mode=self.mode))
        elif path == "/output/receipt":
            response.update(
                target=body["target"],
                expected_parent=body["expected_parent"],
                commit_sha=body["commit_sha"],
                accepted=True,
            )
        else:
            raise AssertionError(path)
        response.update(self.updates)
        return httpx.Response(
            self.status,
            json=response if self.content is None else None,
            content=self.content,
            headers=self.headers,
        )


@pytest.mark.parametrize(
    "origin",
    [
        "https://policy.example",
        "https://policy.example:8443/",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
    ],
)
def test_valid_private_frame(origin: str) -> None:
    value = frame(policy_origin=origin)
    parsed = parse_authority(
        SecretStr(json.dumps({**value.model_dump(mode="json"), "worker_credential": TOKEN})),
        run_id=RUN,
        spec_sha256=DIGEST,
    )
    assert parsed == value
    assert TOKEN not in repr(parsed)
    assert TOKEN not in parsed.model_dump_json()


@pytest.mark.parametrize(
    "updates",
    [
        {"version": True},
        {"version": 2},
        {"version": "1"},
        {"version": None},
        {"run_id": "a" * 31},
        {"spec_sha256": "b" * 63},
        {"authorized": True},
        {"worker_credential": "strata_worker_" + "x" * 42},
        {"worker_credential": TOKEN + "x"},
        {"policy_origin": "http://localhost:8000"},
        {"policy_origin": "http://policy.example"},
        {"policy_origin": "https://user:pass@policy.example"},
        {"policy_origin": "https://policy.example/path"},
        {"policy_origin": "https://policy.example?x=1"},
        {"policy_origin": "https://policy.example#x"},
        {"policy_origin": "https://policy.example:0"},
        {"policy_origin": "https://policy.example:bad"},
        {"policy_origin": "https://policy.example\n"},
        {"policy_origin": ""},
    ],
)
def test_invalid_frame(updates: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        frame(**updates)


@pytest.mark.parametrize(
    "raw", ["not json", "{}", "x" * 8193, '{"worker_credential":"' + TOKEN + '",']
)
def test_parse_failure_has_no_secret_exception_context(raw: str) -> None:
    with pytest.raises(EvaluationError, match="authority_frame_invalid") as caught:
        parse_authority(SecretStr(raw), run_id=RUN, spec_sha256=DIGEST)
    assert caught.value.__context__ is None
    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize(("run", "digest"), [("f" * 32, DIGEST), (RUN, "f" * 64)])
def test_frame_binds_actual_run_and_spec(run: str, digest: str) -> None:
    raw = {**BASE, "policy_origin": "https://policy.example", "worker_credential": TOKEN}
    with pytest.raises(EvaluationError, match="authority_frame_mismatch"):
        parse_authority(SecretStr(json.dumps(raw)), run_id=run, spec_sha256=digest)


@pytest.mark.asyncio
async def test_read_and_same_name_domains_are_distinct_and_no_cookies() -> None:
    server = Server()
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(server), cookies={"session": "never-send"}
        ) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        for kind in ("model", "dataset"):
            decision = await policy.decide(REF, repo_type=kind, spec_sha256=DIGEST)
            assert decision.repo_type == kind
            assert decision.ref == REF
            assert decision.authorized
            assert decision.custody == "unsealed"
    assert [body["repo_type"] for _, body in server.requests] == ["model", "dataset"]


@pytest.mark.parametrize(
    "updates",
    [
        {"run_id": "f" * 32},
        {"spec_sha256": "f" * 64},
        {"ref": "example/other@" + "e" * 40},
        {"repo_type": "model"},
        {"authorized": False},
        {"authorized": 1},
        {"version": True},
        {"custody": "unknown"},
        {"custody": "sealed"},
        {"custody": "locked"},
        {"policy_version": "other"},
        {"extra": TOKEN},
    ],
)
@pytest.mark.asyncio
async def test_read_rejects_denial_unknown_and_bad_echo(updates: dict[str, Any]) -> None:
    server = Server()
    server.updates = updates
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError) as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert TOKEN not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("status", "content", "headers"),
    [
        (307, None, {"Location": "https://elsewhere.example"}),
        (401, TOKEN.encode(), {}),
        (409, None, {}),
        (500, None, {}),
        (200, b"not json " + TOKEN.encode(), {}),
        (200, b"x" * 16385, {}),
        (200, None, {"Content-Encoding": "gzip"}),
    ],
)
@pytest.mark.asyncio
async def test_transport_fails_closed_without_redirects_or_raw_errors(
    status: int, content: bytes | None, headers: dict[str, str]
) -> None:
    server = Server()
    server.status, server.content, server.headers = status, content, headers
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError) as caught:
            await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
    assert len(server.requests) == 1
    assert TOKEN not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_wrong_spec_refuses_before_network() -> None:
    server = Server()
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError, match="authority_spec_mismatch"):
            await policy.decide(REF, repo_type="dataset", spec_sha256="f" * 64)
    assert not server.requests


@pytest.mark.parametrize("mode", ["existing_private", "create_only"])
@pytest.mark.asyncio
async def test_publication_plan_ready_actual_commit_receipt(mode: str) -> None:
    server, hub = Server(mode), Hub()
    commit = CommitRef(
        sha=COMMIT, url="https://huggingface.co/datasets/example/results/commit/" + COMMIT
    )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(
            frame(),
            token="hf_synthetic",
            client=client,
            hub=cast("HFHubClient", cast("object", hub)),
        ) as policy,
    ):
        async with policy.publication(
            requested_repo_id="example/alias", spec_sha256=DIGEST
        ) as fence:
            assert fence.expected_parent == PARENT
            assert fence.canonical_id == "example/results"
            await policy.record_receipt(fence, commit)
            with pytest.raises(EvaluationError, match="publication_already_active"):
                async with policy.publication(
                    requested_repo_id="example/alias", spec_sha256=DIGEST
                ):
                    pytest.fail("nested publication")
        with pytest.raises(EvaluationError, match="publication_receipt_unbound"):
            await policy.record_receipt(fence, commit)
    assert [path for path, _ in server.requests] == [
        "/output/plan",
        "/output/ready",
        "/output/receipt",
    ]
    if mode == "create_only":
        assert hub.events[0] == (
            "create",
            "example/results",
            {"repo_type": "dataset", "private": True, "exist_ok": False},
        )
    else:
        assert not any(event[0] == "create" for event in hub.events)
    assert server.requests[-1][1]["commit_sha"] == COMMIT
    assert server.requests[-1][1]["expected_parent"] == PARENT


@pytest.mark.asyncio
async def test_create_conflict_never_falls_through_to_ready() -> None:
    server, hub = Server("create_only"), Hub()
    hub.create_error = True
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(
            frame(),
            token="hf_synthetic",
            client=client,
            hub=cast("HFHubClient", cast("object", hub)),
        ) as policy,
    ):
        with pytest.raises(EvaluationError, match="publication_create_failed") as caught:
            async with policy.publication(requested_repo_id="example/alias", spec_sha256=DIGEST):
                pytest.fail("create conflict")
    assert [path for path, _ in server.requests] == ["/output/plan"]
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "setting",
    [
        "public",
        "parent",
        "metadata_error",
        "ready_parent",
        "ready_canonical",
        "ready_requested",
        "ready_mode",
    ],
)
@pytest.mark.asyncio
async def test_publication_refuses_changed_or_unknown_destination(setting: str) -> None:
    server, hub = Server(), Hub()
    if setting == "public":
        hub.private = False
    elif setting == "parent":
        hub.parent = "f" * 40
    elif setting == "metadata_error":
        hub.metadata_error = True
    elif setting == "ready_parent":
        server.updates = {"expected_parent": "f" * 40}
    elif setting == "ready_canonical":
        server.updates = {"canonical_id": "example/other"}
    elif setting == "ready_requested":
        server.updates = {"requested_id": "example/other"}
    elif setting == "ready_mode":
        server.updates = {"mode": "create_only"}
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(
            frame(),
            token="hf_synthetic",
            client=client,
            hub=cast("HFHubClient", cast("object", hub)),
        ) as policy,
    ):
        with pytest.raises(EvaluationError):
            async with policy.publication(requested_repo_id="example/alias", spec_sha256=DIGEST):
                pytest.fail("unsafe destination")
    assert not any(path == "/output/receipt" for path, _ in server.requests)


@pytest.mark.parametrize(
    "updates",
    [
        {"expected_parent": None},
        {"expected_parent": "c" * 39},
        {"canonical_id": "other/repo", "known_aliases": []},
        {"privacy": "public"},
        {"mode": "create_only", "expected_parent": PARENT},
        {"known_aliases": ["bad/../path"]},
    ],
)
def test_plan_contract_rejects_ambiguous_authority(updates: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ApprovedOutput.model_validate_json(json.dumps(plan(**updates)))


@pytest.mark.asyncio
async def test_receipt_bad_echo_refuses_success() -> None:
    server, hub = Server(), Hub()
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(
            frame(),
            token="hf_synthetic",
            client=client,
            hub=cast("HFHubClient", cast("object", hub)),
        ) as policy,
        policy.publication(requested_repo_id="example/alias", spec_sha256=DIGEST) as fence,
    ):
        server.updates = {"commit_sha": "f" * 40}
        with pytest.raises(EvaluationError, match="publication_receipt_mismatch"):
            await policy.record_receipt(
                fence, CommitRef(sha=COMMIT, url="https://example.invalid/commit")
            )


@pytest.mark.asyncio
async def test_publication_without_hf_credential_never_requests_plan() -> None:
    server = Server()
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(server)) as client,
        EvaluationPolicyRPC(frame(), token=False, client=client) as policy,
    ):
        with pytest.raises(EvaluationError, match="output_credential_required"):
            async with policy.publication(requested_repo_id="example/alias", spec_sha256=DIGEST):
                pytest.fail("no credential")
    assert not server.requests


@pytest.mark.asyncio
async def test_portable_wire_fixture_matches_actual_client_calls() -> None:
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/evaluation_policy_rpc.json").read_text()
    )
    exchanges = iter(fixture["exchanges"])
    visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        expected = next(exchanges)
        assert request.url.path == expected["path"]
        assert json.loads(request.content) == expected["request"]
        visited.append(request.url.path)
        return httpx.Response(200, json=expected["response"])

    hub = Hub()
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        EvaluationPolicyRPC(
            EvaluationAuthority.model_validate(fixture["authority"]),
            token="hf_synthetic",
            client=client,
            hub=cast("HFHubClient", cast("object", hub)),
        ) as policy,
    ):
        await policy.decide(REF, repo_type="dataset", spec_sha256=DIGEST)
        async with policy.publication(
            requested_repo_id="example/alias", spec_sha256=DIGEST
        ) as fence:
            await policy.record_receipt(
                fence, CommitRef(sha=COMMIT, url="https://example.invalid/commit")
            )
    assert len(visited) == 4
    assert next(exchanges, None) is None
