"""The VM-side batch-inference runner.

The security-critical surface is the INERT config + template handling (no code execution)
and the guarantee that the HF write token never reaches a progress event. Pure functions
are tested directly; the orchestration is tested with the heavy externals (datasets, vLLM
serving, the batch runner, the Hub client) mocked — but the real LLMClient + openai_compat
provider are kept, so the provider_clients= seam is validated at runtime.
"""

from __future__ import annotations

import contextlib
import json
import types
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from strata_forge.compute.batch import BatchInferenceResult
from strata_forge.pipelines import inference_runner as ir

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from strata_forge.llm import LLMClient, LLMResponse

_TOKEN = "hf_secretwritetoken1234567890"


def _ok(text: str) -> BatchInferenceResult:
    return BatchInferenceResult(response=cast("LLMResponse", types.SimpleNamespace(text=text)))


def _fail(exc: Exception) -> BatchInferenceResult:
    return BatchInferenceResult(error=exc)


def _spec_json(**overrides: Any) -> str:
    base: dict[str, Any] = {
        "model_id": "org/model",
        "dataset_id": "org/ds",
        "split": "train",
        "column_mapping": {"q": "question"},
        "template": "Answer: {q}",
        "output_repo_id": "org/out",
        "progress_path": "progress.jsonl",
    }
    base.update(overrides)
    return json.dumps(base)


# ------------------------------ render_template -----------------------------


def test_render_template_substitutes_mapped_columns() -> None:
    out = ir.render_template("Q: {q} / {missing}", {"question": "hi"}, {"q": "question"})
    assert out == "Q: hi / {missing}"  # mapped replaced; unmapped placeholder left literal


def test_render_template_is_injection_safe() -> None:
    # str.format injection vectors — positional, attribute access, conversion, format spec —
    # are NOT bare {name} tokens, so the regex never matches them: they stay exactly literal
    # and the column value cannot leak through a crafted placeholder. ({x} is unmapped.)
    row = {"question": "hi"}
    mapping = {"q": "question"}
    for hostile in ("{0}", "{q.__class__}", "{q!r}", "{q:>9999999}", "{x}"):
        rendered = ir.render_template(hostile, row, mapping)
        assert "hi" not in rendered  # the column value never leaks via a crafted placeholder
        assert rendered == hostile  # left exactly literal — no execution, no substitution


def test_render_template_has_no_brace_escaping() -> None:
    # Documented nuance (not a vuln): there's no `{{`-escaping; the inner {q} still substitutes.
    assert ir.render_template("{{q}}", {"question": "hi"}, {"q": "question"}) == "{hi}"


def test_render_template_caps_length() -> None:
    cap = ir._MAX_RENDERED_CHARS  # pyright: ignore[reportPrivateUsage]
    out = ir.render_template("{v}", {"c": "x" * (cap + 1000)}, {"v": "c"})
    assert len(out) == cap


# -------------------------------- load_spec ---------------------------------


def test_load_spec_rejects_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
    with pytest.raises(ir.RunError, match="not set"):
        ir.load_spec()


def test_load_spec_rejects_extra_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(surprise="x"))
    with pytest.raises(ir.RunError, match="invalid STRATA_RUN_CONFIG"):
        ir.load_spec()


@pytest.mark.parametrize(
    "bad",
    ["../evil", "https://x/y", "no-slash", "a b/c", "org/../escape", "org/model\n"],
)
def test_load_spec_rejects_bad_ids(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id=bad))
    with pytest.raises(ir.RunError, match="invalid model id"):
        ir.load_spec()


def test_load_spec_accepts_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    spec = ir.load_spec()
    assert spec.model_id == "org/model"
    assert spec.output_repo_id == "org/out"


# ------------------------------ sanitize / ids ------------------------------


def test_sanitize_strips_token_and_token_shapes() -> None:
    msg = f"boom token={_TOKEN} and Bearer abc.def-123 done"
    out = ir._sanitize(msg, _TOKEN)  # pyright: ignore[reportPrivateUsage]
    assert _TOKEN not in out
    assert "Bearer abc.def-123" not in out
    assert "boom" in out
    assert "done" in out


class _NeverEndingDataset:
    """A split that yields forever — islice MUST stop it, or _load_rows would hang/OOM."""

    column_names: ClassVar[list[str]] = ["question"]

    def __init__(self, counter: dict[str, int]) -> None:
        self._counter = counter

    def __iter__(self) -> Any:
        i = 0
        while True:
            self._counter["consumed"] += 1
            yield {"question": f"q{i}"}
            i += 1


def test_load_rows_caps_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    counter = {"consumed": 0}

    def _load_dataset(*_a: Any, **_k: Any) -> _NeverEndingDataset:
        return _NeverEndingDataset(counter)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=_load_dataset))
    spec = ir.RunSpec.model_validate_json(_spec_json(hyperparams={"row_limit": 3}))
    rows = ir._load_rows(spec, None)  # pyright: ignore[reportPrivateUsage]
    assert len(rows) == 3
    assert counter["consumed"] == 3  # islice stopped at the cap; the split was NOT materialized


def test_build_requests_index_aligned() -> None:
    spec = ir.RunSpec.model_validate_json(_spec_json())
    rows = [{"question": "a"}, {"question": "b"}]
    prompts, ids = ir._build_requests(spec, rows)  # pyright: ignore[reportPrivateUsage]
    assert ids == ["row-0", "row-1"]
    assert [m[0].content for m in prompts] == ["Answer: a", "Answer: b"]


# ------------------------- _run_batches (mocked runner) ---------------------


class _FakeRunner:
    """Stand-in for BatchInferenceRunner: returns one canned result per prompt."""

    scripted: ClassVar[list[BatchInferenceResult]] = []

    def __init__(self, client: Any, *, concurrency: int, on_error: str) -> None:
        del client, concurrency
        assert on_error == "collect"  # load-bearing: never cancel the batch on first failure

    async def run(self, prompts: Any, **_: Any) -> tuple[BatchInferenceResult, ...]:
        return tuple(_FakeRunner.scripted[: len(list(prompts))])


async def test_run_batches_reconciles_and_emits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FakeRunner.scripted = [_ok("OUT0"), _fail(RuntimeError("boom"))]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)
    spec = ir.RunSpec.model_validate_json(_spec_json())
    prompts, ids = ir._build_requests(spec, [{"question": "a"}, {"question": "b"}])  # pyright: ignore[reportPrivateUsage]

    progress = tmp_path / "progress.jsonl"
    with ir.JsonlProgressWriter(str(progress)) as writer:
        out = await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
            spec, client=cast("LLMClient", object()), prompts=prompts, custom_ids=ids, writer=writer
        )

    assert out == [
        {"custom_id": "row-0", "output": "OUT0", "error": None},
        {"custom_id": "row-1", "output": None, "error": "RuntimeError('boom')"},
    ]
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    step = [e for e in events if e["kind"] == "step"]
    assert step
    assert step[-1]["metrics"] == {"succeeded": 1.0, "failed": 1.0}


# ----------------------- main: happy path + no token leak -------------------


@contextlib.asynccontextmanager
async def _fake_serving(*_a: Any, **_kw: Any) -> AsyncGenerator[Any]:
    yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1")


async def test_main_happy_path_pushes_and_never_leaks_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        del spec, token
        return [{"question": "a"}, {"question": "b"}]

    def _write(rows: Any, workdir: Any) -> Path:
        del rows, workdir
        return tmp_path / "results.parquet"

    pushed: dict[str, Any] = {}

    async def _fake_push(spec: Any, results_path: Any, token: str) -> str:
        del results_path
        pushed["token"] = token  # the runner must pass the EXPLICIT write token
        return cast("str", spec.output_repo_id)

    _FakeRunner.scripted = [_ok("A"), _ok("B")]
    monkeypatch.setattr(ir, "_load_rows", _rows)
    monkeypatch.setattr(ir, "serving_endpoint", _fake_serving)
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)
    monkeypatch.setattr(ir, "_write_results", _write)
    monkeypatch.setattr(ir, "_push_results", _fake_push)

    code = await ir.main()
    assert code == 0
    assert pushed["token"] == _TOKEN  # token reached the push (the only place it's used)

    text = progress.read_text()
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "end"
    assert events[-1]["message"] == "org/out"  # the result location (a repo id, not a secret)
    assert _TOKEN not in text  # the token NEVER appears in any emitted event


async def test_main_error_path_scrubs_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)

    def _boom(spec: Any, token: Any) -> list[dict[str, Any]]:
        del spec, token
        msg = f"dataset load failed with creds {_TOKEN}"
        raise ir.RunError(msg)

    monkeypatch.setattr(ir, "_load_rows", _boom)

    code = await ir.main()
    assert code == 1
    text = progress.read_text()
    assert _TOKEN not in text  # even an exception message carrying the token is scrubbed
    assert "***" in text


# ----------------------- main: optional push (results-on-VM) ----------------


async def test_main_no_token_keeps_results_on_vm_and_skips_push(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    # No write token + no output repo -> keep results on the VM, never push.
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG",
        _spec_json(progress_path=str(progress), output_repo_id=None, run_id="run123"),
    )
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        del spec, token
        return [{"question": "a"}]

    written: dict[str, str] = {}

    def _write(rows: Any, outdir: Any) -> str:
        del rows
        written["outdir"] = str(outdir)
        return f"{outdir}/results.parquet"

    pushed = {"called": False}

    async def _fake_push(*_a: Any, **_kw: Any) -> str:
        pushed["called"] = True
        return "unreachable"

    _FakeRunner.scripted = [_ok("A")]
    monkeypatch.setattr(ir, "_load_rows", _rows)
    monkeypatch.setattr(ir, "serving_endpoint", _fake_serving)
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)
    monkeypatch.setattr(ir, "_write_results", _write)
    monkeypatch.setattr(ir, "_push_results", _fake_push)

    code = await ir.main()
    assert code == 0
    assert pushed["called"] is False  # no token + no repo -> never pushes
    # Results land in the cleanup-surviving dir named by the run id (outside the workdir), not the cwd.
    assert written["outdir"].endswith("strata-inference-results/run123")
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert events[-1]["kind"] == "end"
    assert events[-1]["message"].endswith("results.parquet")  # a VM path, not a repo id


def test_local_results_dir_validates_run_id() -> None:
    safe = ir._local_results_dir("abc-123_DEF")  # pyright: ignore[reportPrivateUsage]
    assert str(safe).endswith("strata-inference-results/abc-123_DEF")
    # Traversal / unsafe / empty / missing names fall back to the cwd — never an escaping path.
    for bad in ("../../etc", "a/b", "", None):
        assert ir._local_results_dir(bad) == ir.Path.cwd()  # pyright: ignore[reportPrivateUsage]
