"""Pinned local consumption shares renderer bytes without granting remote/native authority."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from strata_forge.prompts.composition_errors import PromptCompositionError
from strata_forge.prompts.consumption import (
    ConsumedBuild,
    LocalBuildReference,
    LocalBuildSource,
    consume_build,
    render_consumed,
)
from strata_forge.prompts.portable import canonical_bytes

ROOT = Path(__file__).parents[3]
CORPUS = json.loads((ROOT / "tests/fixtures/prompts/composition-renders-v1.json").read_text())
SERVER = ROOT / "packages/prompt-sdk/test/fixtures/stored-build-http-v1.json"
SERVER_SHA = "5efc2497eac64c41fcdca329f085875d396be6d9ffe9f4bfff48e0e65aa83cb7"


def source(*, partition: str = "fixture:alice", reference: bool = False) -> LocalBuildSource:
    return LocalBuildSource(
        kind="caller_bytes",
        partition=partition,
        reference=LocalBuildReference(
            project_id="a" * 32,
            surface_resource_id="b" * 32,
            resource_id="e" * 32,
            version_id="f" * 32,
            manifest_sha256="1" * 64,
        )
        if reference
        else None,
    )


def consume(raw: bytes, **changes: object) -> ConsumedBuild:
    arguments: dict[str, object] = {
        "expected_build_id": hashlib.sha256(raw).hexdigest(),
        "expected_size_bytes": len(raw),
        "source": source(),
    }
    arguments.update(changes)
    return consume_build(raw, **arguments)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["name"])
async def test_existing_python_node_render_corpus_is_unchanged(case: dict[str, object]) -> None:
    raw = bytes.fromhex(cast("str", case["build_canonical_utf8_hex"]))
    build = consume(raw)
    assert build.build_id == case["build_id"]
    assert build.canonical == raw
    assert build.size_bytes == len(raw)
    rendered = await render_consumed(build, case["values"])
    assert rendered.render_id == case["render_id"]
    assert rendered.canonical.hex() == case["render_canonical_utf8_hex"]


async def test_actual_server_golden_bytes_and_descriptive_locator() -> None:
    assert hashlib.sha256(SERVER.read_bytes()).hexdigest() == SERVER_SHA
    fixture = json.loads(SERVER.read_text())
    raw = fixture["body_utf8"].encode("utf-8")
    build = consume(raw, source=source(reference=True))
    assert build.build_id == fixture["body_sha256"]
    assert build.size_bytes == fixture["body_size_bytes"] == 1511
    assert build.source.reference is not None
    assert build.source.reference.resource_id == fixture["compile_receipt"]["ref"]["resource_id"]
    first = await render_consumed(build, {"question": "{{literal}} 😀é\r\n"})
    other = await render_consumed(
        consume(raw, source=source(partition="fixture:bob")), {"question": "{{literal}} 😀é\r\n"}
    )
    assert first.canonical == other.canonical
    assert json.dumps("{{literal}} 😀é\r\n", ensure_ascii=False)[1:-1] in json.dumps(
        first.wire_payload()["output"], ensure_ascii=False
    )
    assert "authorized" not in build.source.model_dump()


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_build_id": "0" * 64},
        {"expected_build_id": "f" * 64 + "\n"},
        {"expected_size_bytes": 0},
        {"expected_size_bytes": True},
        {"expected_size_bytes": 8 * 1024 * 1024 + 1},
        {"expected_size_bytes": 1},
    ],
)
def test_pin_refusals_are_fixed_and_content_free(changes: dict[str, object]) -> None:
    raw = bytes.fromhex(CORPUS["cases"][0]["build_canonical_utf8_hex"])
    with pytest.raises(PromptCompositionError) as caught:
        consume(raw, **changes)
    assert str(caught.value) in {"consumption_invalid", "consumption_pin_mismatch"}
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "raw", [b"", b"{}", b"\xff", b'{"schema_version":1,"schema_version":1}', b"\xef\xbb\xbf{}"]
)
def test_invalid_bytes_with_matching_hash_still_refuse(raw: bytes) -> None:
    with pytest.raises(PromptCompositionError):
        consume(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "server_authorized"},
        {"partition": "private-canary\n"},
        {"reference": {"resource_id": "private-canary"}},
    ],
)
def test_forged_source_metadata_never_claims_remote_authority(changes: dict[str, object]) -> None:
    raw = bytes.fromhex(CORPUS["cases"][0]["build_canonical_utf8_hex"])
    fields = source().model_dump()
    fields.update(changes)
    supplied = LocalBuildSource.model_construct(**fields)
    with pytest.raises(PromptCompositionError, match="consumption_source_invalid") as caught:
        consume(raw, source=supplied)
    assert caught.value.__context__ is None
    assert "private-canary" not in str(caught.value)


async def test_forged_consumed_snapshot_is_rechecked_before_render() -> None:
    raw = bytes.fromhex(CORPUS["cases"][0]["build_canonical_utf8_hex"])
    build = consume(raw)
    forged = build.model_copy(update={"build_id": "0" * 64})
    with pytest.raises(PromptCompositionError, match="consumption_pin_mismatch"):
        await render_consumed(forged, {})
    assert "canonical=" not in repr(build)
    with pytest.raises(ValidationError, match="frozen_instance"):
        build.source.partition = "changed"


@pytest.mark.parametrize(
    "values", [{}, {"question": True}, {"question": 1}, {"question": "ok", "extra": "no"}]
)
async def test_existing_renderer_rejects_missing_extra_or_incompatible_bindings(
    values: object,
) -> None:
    raw = json.loads(SERVER.read_text())["body_utf8"].encode()
    with pytest.raises(PromptCompositionError):
        await render_consumed(consume(raw), values)


def test_exact_eight_mib_schema_valid_local_template_and_honest_cap() -> None:
    fixture = json.loads(SERVER.read_text())
    wire = json.loads(fixture["body_utf8"])
    # The synthetic local snapshot has no retained duplicate-literal advisory.
    wire["diagnostics"] = []
    leaf = next(item for item in wire["outputs"][0]["leaves"] if item["kind"] == "text")
    leaf["text"] = ""
    leaf["text"] = "x" * (8 * 1024 * 1024 - len(canonical_bytes(wire)))
    raw = canonical_bytes(wire)
    assert len(raw) == 8 * 1024 * 1024
    assert consume(raw).canonical == raw
    with pytest.raises(PromptCompositionError, match="consumption_invalid"):
        consume(raw + b"x")


def test_import_and_local_render_have_no_network_trace_ml_or_tokenizer_asset_work() -> None:
    program = r"""
import asyncio, importlib.abc, json, pathlib, socket, sys
fixture = json.loads(pathlib.Path(sys.argv[1]).read_text())
raw = fixture["body_utf8"].encode()
attempts = []
def denied(*args, **kwargs):
    attempts.append("network")
    raise AssertionError("network_forbidden")
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch","transformers","trl","peft","numpy","langfuse","pyarrow","skypilot"}:
            attempts.append(fullname)
            raise AssertionError("optional_import_forbidden")
sys.meta_path.insert(0, Guard())
import tiktoken
tiktoken.get_encoding = denied
tiktoken.encoding_for_model = denied
from strata_forge.prompts import LocalBuildSource, consume_build, render_consumed
source = LocalBuildSource(kind="caller_bytes", partition="fixture:local", reference=None)
build = consume_build(raw, expected_build_id=fixture["body_sha256"], expected_size_bytes=len(raw), source=source)
result = asyncio.run(render_consumed(build, {"question":"literal {{value}}"}))
assert result.wire_payload()["build_id"] == build.build_id
assert attempts == [], attempts
print("local-only")
"""
    result = subprocess.run(  # noqa: S603 - fixed local program and a synthetic fixture path
        [sys.executable, "-c", program, str(SERVER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "local-only"
