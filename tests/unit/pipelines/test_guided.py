"""Guided JSON: the schema's rules, validation of replies, the one repair turn and its record."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from strata_forge.compute.batch import BatchInferenceResult
from strata_forge.core.redact import Redactor
from strata_forge.llm import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute
from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import _guided as gd
from strata_forge.pipelines import inference_runner as ir

if TYPE_CHECKING:
    from pydantic import JsonValue

    from strata_forge.llm import LLMClient

_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["entailed", "contradicted", "unknown"]},
        "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
    },
    "required": ["verdict"],
    "additionalProperties": False,
    "$defs": {"step": {"type": "string"}},
}


def _spec(**hyperparams: Any) -> ir.RunSpec:
    return ir.RunSpec.model_validate(
        {
            "engine_version": SPEC_VERSION,
            "model_id": "org/model",
            "dataset_id": "org/ds",
            "split": "train",
            "column_mapping": {"q": "question"},
            "template": "Answer: {q}",
            "hyperparams": {"guided_json": _SCHEMA} | hyperparams,
        }
    )


# ------------------------------ the schema ------------------------------


def test_a_schema_with_local_refs_is_accepted() -> None:
    assert gd.schema_problem(_SCHEMA) is None
    assert ir.Hyperparams.model_validate({"guided_json": _SCHEMA}).guided_json == _SCHEMA


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        ({}, "admits anything"),
        ({"type": 7}, "not a valid JSON Schema"),
        ({"$ref": "https://example.com/s.json"}, "only its own definitions"),
        ({"properties": {"a": {"$ref": "other.json#/x"}}}, "only its own definitions"),
        ({"description": "x" * gd.MAX_SCHEMA_JSON_BYTES}, "bytes of JSON"),
    ],
)
def test_an_unusable_schema_is_refused(schema: dict[str, JsonValue], match: str) -> None:
    problem = gd.schema_problem(schema)
    assert problem is not None
    assert match in problem
    with pytest.raises(ValueError, match=match):
        ir.Hyperparams.model_validate({"guided_json": schema})


def test_the_draft_the_schema_names_is_the_one_checked() -> None:
    draft7 = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}
    assert gd.OutputValidator(draft7).draft == "Draft7Validator"
    assert gd.OutputValidator(_SCHEMA).draft == "Draft202012Validator"


def test_guided_json_is_refused_on_a_scoring_engine() -> None:
    with pytest.raises(ValueError, match="generates no text"):
        ir.Hyperparams.model_validate(
            {"engine": "hf_classify", "pair_template": "{h}", "guided_json": _SCHEMA}
        )


def test_the_schema_digest_ignores_key_order() -> None:
    reordered = json.loads(json.dumps(_SCHEMA, sort_keys=True))
    assert gd.schema_digest(reordered) == gd.schema_digest(_SCHEMA)
    canonical = json.dumps(_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
    assert gd.schema_digest(_SCHEMA) == hashlib.sha256(canonical).hexdigest()


def test_the_schema_rides_as_the_servers_structured_output_constraint() -> None:
    extras = ir._sampling_extras(_spec().hyperparams)  # pyright: ignore[reportPrivateUsage]
    assert extras == {"openai_compat": {"extra_body": {"structured_outputs": {"json": _SCHEMA}}}}


# ------------------------------ validating a reply ------------------------------


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('{"verdict": "entailed", "steps": ["a"]}', None),
        ('{"verdict": "maybe"}', "'maybe' is not one of"),
        ('{"verdict": "unknown", "extra": 1}', "Additional properties"),
        ('{"verdict": "unknown", "steps": [1]}', "at /steps/0"),
        ('{"verdict": "entailed"', "not valid JSON"),
        ('{"verdict": NaN}', "not valid JSON"),
        (None, "no text"),
    ],
)
def test_a_reply_is_checked_against_the_schema(text: str | None, error: str | None) -> None:
    found = gd.OutputValidator(_SCHEMA).error(text)
    if error is None:
        assert found is None
    else:
        assert found is not None
        assert error in found


def test_a_reference_outside_the_schema_is_never_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    # schema_problem refuses it at spec load; the validator itself must not fetch it either.
    def _no_network(*_a: Any, **_kw: Any) -> Any:
        pytest.fail("a schema reference was fetched")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    validator = gd.OutputValidator({"$ref": "https://example.com/s.json"})
    with pytest.raises(Exception, match=r"example\.com"):
        validator.error("{}")


def test_the_repair_turn_names_the_error() -> None:
    message = gd.repair_message("'maybe' is not one of ['a']")
    assert message.startswith("Your previous reply does not satisfy the required JSON schema: ")
    assert "'maybe' is not one of ['a']" in message
    assert gd.repair_prompt_digest() == hashlib.sha256(gd.REPAIR_PROMPT.encode()).hexdigest()


# ------------------------------ the repair turn in a batch ------------------------------


def _response(text: str, *, input_tokens: int = 5, output_tokens: int = 10) -> BatchInferenceResult:
    return BatchInferenceResult(
        response=LLMResponse(
            text=text,
            finish_reason="stop",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            cost_usd=0.0,
            route=ModelRoute(model="m", provider="openai_compat", provider_model_id="m"),
            latency_ms=100.0,
        )
    )


class _ScriptedRunner:
    """Answers each conversation from a script keyed by its prompt, and whether it is a repair."""

    script: ClassVar[dict[str, BatchInferenceResult]] = {}
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, client: Any, *, concurrency: int, on_error: str) -> None:
        del client, concurrency, on_error

    async def run(self, prompts: Any, **kwargs: Any) -> tuple[BatchInferenceResult, ...]:
        conversations = [list(p) for p in prompts]
        _ScriptedRunner.calls.append({"conversations": conversations, **kwargs})
        out: list[BatchInferenceResult] = []
        for conversation in conversations:
            first = str(conversation[0].content)
            out.append(
                _ScriptedRunner.script[first if len(conversation) == 1 else f"repair:{first}"]
            )
        return tuple(out)


async def test_each_invalid_reply_gets_exactly_one_repair_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ScriptedRunner.calls = []
    _ScriptedRunner.script = {
        "Answer: valid": _response('{"verdict": "entailed"}'),
        "Answer: fixed": _response('{"verdict": "maybe"}'),
        "repair:Answer: fixed": _response(
            '{"verdict": "unknown"}', input_tokens=7, output_tokens=3
        ),
        "Answer: still": _response("not json"),
        "repair:Answer: still": _response('{"verdict": 1}'),
        "Answer: lost": _response('{"verdict": "maybe"}'),
        "repair:Answer: lost": BatchInferenceResult(error=RuntimeError("server gone")),
        "Answer: failed": BatchInferenceResult(error=RuntimeError("boom")),
    }
    monkeypatch.setattr(ir, "BatchInferenceRunner", _ScriptedRunner)
    spec = _spec(max_tokens=64)
    rows = [{"question": q} for q in ("valid", "fixed", "still", "lost", "failed")]
    prompts, ids = ir._build_requests(spec, rows)  # pyright: ignore[reportPrivateUsage]

    out = await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
        spec,
        client=cast("LLMClient", object()),
        prompts=prompts,
        custom_ids=ids,
        writer=None,
        redactor=Redactor(),
    )

    by_id = {row["custom_id"]: row for row in out}
    assert (by_id["row-0"]["json_valid"], by_id["row-0"]["json_retried"]) == (True, False)
    assert by_id["row-0"]["json_error"] is None
    # Repaired: the final reply is the output, and the row is accounted as both attempts.
    fixed = by_id["row-1"]
    assert (fixed["output"], fixed["json_valid"], fixed["json_retried"]) == (
        '{"verdict": "unknown"}',
        True,
        True,
    )
    assert (fixed["input_tokens"], fixed["output_tokens"]) == (5 + 7, 10 + 3)
    still = by_id["row-2"]
    assert (still["json_valid"], still["json_retried"]) == (False, True)
    assert "is not one of" in still["json_error"] or "is not of type" in still["json_error"]
    # The repair never answered: the first reply stays, and the record says why it is invalid.
    lost = by_id["row-3"]
    assert lost["output"] == '{"verdict": "maybe"}'
    assert lost["json_valid"] is False
    assert "the repair request failed" in lost["json_error"]
    failed = by_id["row-4"]
    assert failed["error"] is not None
    assert (failed["json_valid"], failed["json_retried"]) == (None, False)

    first, repair = _ScriptedRunner.calls
    assert len(first["conversations"]) == 5
    # One repair call, for the three rows whose first reply failed validation.
    assert [c[0].content for c in repair["conversations"]] == [
        "Answer: fixed",
        "Answer: still",
        "Answer: lost",
    ]
    turn = repair["conversations"][0]
    assert [m.role for m in turn] == ["user", "assistant", "user"]
    assert turn[1].content == '{"verdict": "maybe"}'
    assert turn[2].content == gd.repair_message(
        str(gd.OutputValidator(_SCHEMA).error('{"verdict": "maybe"}'))
    )
    # The repair turn runs under the same constraint and sampling settings.
    assert repair["provider_extras"] == first["provider_extras"]
    assert repair["max_tokens"] == 64


def test_the_manifest_records_the_schema_and_the_counts() -> None:
    spec = _spec()
    base = {
        "output": "x",
        "reasoning": None,
        "finish_reason": "stop",
        "input_tokens": 1,
        "output_tokens": 1,
    }
    rows = [
        base | {"error": None, "json_valid": True, "json_retried": False},
        base | {"error": None, "json_valid": True, "json_retried": True},
        base | {"error": None, "json_valid": False, "json_retried": True},
        base | {"error": "boom", "json_valid": None, "json_retried": False},
    ]
    manifest = ir._build_manifest(spec, rows, None)  # pyright: ignore[reportPrivateUsage]
    assert manifest["guided"] == {
        "kind": "json",
        "schema_sha256": gd.schema_digest(_SCHEMA),
        "constraint": "structured_outputs.json",
        "validator": "Draft202012Validator",
        "max_repairs": 1,
        "repair_prompt_sha256": gd.repair_prompt_digest(),
    }
    counts = manifest["counts"]
    assert (counts["json_valid"], counts["json_invalid"]) == (2, 1)
    assert (counts["json_retried"], counts["json_repaired"]) == (2, 1)


def test_an_unguided_run_records_no_guided_block() -> None:
    spec = ir.RunSpec.model_validate(_spec().model_dump() | {"hyperparams": {}})
    row = {"output": "x", "reasoning": None, "error": None, "finish_reason": "stop"}
    manifest = ir._build_manifest(spec, [row | {"input_tokens": 1, "output_tokens": 1}], None)  # pyright: ignore[reportPrivateUsage]
    assert manifest["guided"] is None
    assert "json_valid" not in manifest["counts"]


async def test_the_constraint_reaches_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real client and batch runner pass ``structured_outputs`` to the model server's body."""
    from pydantic import SecretStr

    from strata_forge.compute.batch import BatchInferenceRunner
    from strata_forge.llm import LLMClient as RealClient
    from strata_forge.llm.providers.config import OpenAICompatConfig
    from strata_forge.llm.providers.openai_compat import (
        UNAUTHENTICATED_API_KEY,
        OpenAICompatProvider,
    )

    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    monkeypatch.setattr(ir, "BatchInferenceRunner", BatchInferenceRunner)
    provider = OpenAICompatProvider(
        OpenAICompatConfig(
            base_url="http://127.0.0.1:8000/v1", api_key=SecretStr(UNAUTHENTICATED_API_KEY)
        )
    )
    client = RealClient(
        model="org/model", provider="openai_compat", provider_clients={"openai_compat": provider}
    )
    spec = _spec(seed=3)
    prompts, ids = ir._build_requests(spec, [{"question": "a"}])  # pyright: ignore[reportPrivateUsage]
    await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
        spec, client, prompts, ids, None, redactor=Redactor()
    )
    assert captured["extra_body"] == {"structured_outputs": {"json": _SCHEMA}}
    assert captured["seed"] == 3
