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
from strata_forge.compute.batch import BatchInferenceRunner as _RealBatchRunner
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
    # Phases precede `start` (the dataset download runs before it), so the milestone contract
    # is about the countable kinds: the first of those is still `start`.
    assert next(k for k in kinds if k != "phase") == "start"
    assert kinds[-1] == "end"
    assert events[-1]["message"] == "org/out"  # the result location (a repo id, not a secret)
    assert _TOKEN not in text  # the token NEVER appears in any emitted event


# --------------------------- main: provisioning phases ----------------------


def _same_dir(left: Any, right: Any) -> bool:
    return ir.Path(left).resolve() == ir.Path(right).resolve()


def _recording_serving(record: dict[str, Any], *, drive: Any = None) -> Any:
    """A `serving_endpoint` stand-in that records its arguments and can drive the phase hook."""

    @contextlib.asynccontextmanager
    async def _serving(backend: Any, task: Any, **kwargs: Any) -> AsyncGenerator[Any]:
        record["backend"] = backend
        record["task"] = task
        record.update(kwargs)
        if drive is not None:
            drive(record)
        yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1")

    return _serving


def _mock_main_deps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    serving: Any,
    *,
    rows: list[dict[str, Any]] | None = None,
    scripted: list[BatchInferenceResult] | None = None,
    written: list[list[dict[str, Any]]] | None = None,
) -> None:
    """Stub the heavy externals so `main` runs its own orchestration end to end.

    `rows`/`scripted` drive the input split and the per-row outcomes; `written`, when given,
    collects what `_write_results` was handed, so a test can assert the evidence a failed run
    leaves behind.
    """
    the_rows = [{"question": "a"}] if rows is None else rows

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        del spec, token
        return the_rows

    def _write(rows: Any, outdir: Any) -> Path:
        del outdir
        if written is not None:
            written.append(list(cast("list[dict[str, Any]]", rows)))
        return tmp_path / "results.parquet"

    async def _push(spec: Any, results_path: Any, token: str) -> str:
        del results_path, token
        return cast("str", spec.output_repo_id)

    _FakeRunner.scripted = [_ok("A")] if scripted is None else scripted
    monkeypatch.setattr(ir, "_load_rows", _rows)
    monkeypatch.setattr(ir, "serving_endpoint", serving)
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)
    monkeypatch.setattr(ir, "_write_results", _write)
    monkeypatch.setattr(ir, "_push_results", _push)


async def test_main_reports_a_phase_for_every_silent_stretch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Each of these covers a stretch with nothing to count. Without them a run is one
    # indeterminate wait — which is where a model server that never comes up spends its
    # entire timeout, leaving the user staring at a spinner.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)
    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)

    assert await ir.main() == 0
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    kinds = [e["kind"] for e in events]
    assert [e["message"] for e in events if e["kind"] == "phase"] == [
        "Loading the dataset",
        "Generating responses",
        "Writing results",
        "Uploading results to the Hub",
    ]
    # The split download runs before the first countable milestone, so its phase must too.
    assert kinds.index("phase") < kinds.index("start")
    phase_events = [e for e in events if e["kind"] == "phase"]
    assert all(e["step"] is None and e["total_steps"] is None for e in phase_events)


async def test_serving_hook_phrases_are_scrubbed_and_capped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `serving_endpoint`'s on_phase is public API, so the phrase reaching the progress file is
    # not necessarily one this module wrote. It gets the same treatment as an error message.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)

    def _drive(rec: dict[str, Any]) -> None:
        rec["on_phase"](f"pulling weights with {_TOKEN} " + "x" * 500)

    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record, drive=_drive))

    assert await ir.main() == 0
    text = progress.read_text()
    assert _TOKEN not in text
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    driven = [e for e in events if e["kind"] == "phase" and "pulling weights" in e["message"]]
    assert len(driven) == 1
    assert "***" in driven[0]["message"]
    assert len(driven[0]["message"]) <= 200


async def test_serving_gets_the_phase_hook_a_log_dir_and_unbuffered_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))
    monkeypatch.chdir(tmp_path)  # on the VM this is the per-run job workdir

    assert await ir.main() == 0
    # The hook has to reach the readiness wait — that is where the whole blackout happens.
    assert record["on_phase"] is not None
    # The served process's streams are teed to disk, because when the runner dies its buffers
    # die with it and the file is the only thing left that explains why the server never came up.
    log_dir = record["backend"]._log_dir  # pyright: ignore[reportPrivateUsage]
    assert log_dir is not None
    assert _same_dir(log_dir, tmp_path)
    # Unbuffered, or a hung server's output sits in its own 8 KiB block buffer and the file
    # stays empty for exactly the failure it exists to explain.
    assert record["task"].env["PYTHONUNBUFFERED"] == "1"
    # No runtime kernel compilation. vLLM's default sampler is FlashInfer's, which JIT-builds its
    # kernels during warmup by shelling out to ninja — absent on a GPU image that ships the driver
    # and runtime but no build tools, and the run dies there having already loaded the weights,
    # compiled the graph and allocated the KV cache. We do not provision the user's box, so the
    # engine must not require a compiler on it.
    assert record["task"].env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


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


# --------------------- the verdict: did the run produce anything? -----------


async def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]],
    scripted: list[BatchInferenceResult],
    written: list[list[dict[str, Any]]] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Drive `main` over a scripted split; return its exit code and the progress events."""
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)
    _mock_main_deps(
        monkeypatch, tmp_path, _fake_serving, rows=rows, scripted=scripted, written=written
    )
    code = await ir.main()
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    return code, events


async def test_a_run_whose_every_row_failed_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verdict has to describe the outcome.

    Row errors are collected, not fatal — but the control plane reads this process's exit code
    as the run's verdict, so collecting them all and exiting 0 tells the user their results are
    ready when the file holds nothing but errors. A staging run reported `succeeded` over 2098
    failed rows and zero generations, and nothing anywhere on the run contradicted it.
    """
    code, events = await _run_main(
        monkeypatch,
        tmp_path,
        rows=[{"question": "a"}, {"question": "b"}],
        scripted=[
            _fail(RuntimeError("provider exploded")),
            _fail(RuntimeError("provider exploded")),
        ],
    )

    assert code == 1
    # The counts and the destination are still recorded: for this failure they ARE the diagnosis.
    end = [e for e in events if e["kind"] == "end"]
    assert end, "the counts must survive the failure"
    assert end[-1]["metrics"] == {"succeeded": 0.0, "failed": 2.0}
    error = [e for e in events if e["kind"] == "error"]
    assert error, "the run must say why it failed, not just that it did"
    # A representative row error, so the reason is legible without downloading the parquet.
    assert "all 2 rows failed" in error[-1]["message"]
    assert "provider exploded" in error[-1]["message"]
    # Reported on stderr too: that is where the control plane reads a failed run's reason from,
    # and a caught exception prints no traceback of its own.
    assert "provider exploded" in capsys.readouterr().err


async def test_a_partially_failed_run_still_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Deliberate: it produced usable output, and the counts describe the rest. Only "nothing at
    # all" is a failure, because only that has no reading under which the run did its job.
    code, events = await _run_main(
        monkeypatch,
        tmp_path,
        rows=[{"question": "a"}, {"question": "b"}],
        scripted=[_ok("A"), _fail(RuntimeError("just this one"))],
    )

    assert code == 0
    assert events[-1]["kind"] == "end"
    assert events[-1]["metrics"] == {"succeeded": 1.0, "failed": 1.0}


async def test_a_run_over_an_empty_split_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same bug class, different cause: zero rows also means zero output, and an empty parquet
    # reported as success is the same lie with no error column to explain it.
    code, events = await _run_main(monkeypatch, tmp_path, rows=[], scripted=[])

    assert code == 1
    error = [e for e in events if e["kind"] == "error"]
    assert error, "an empty split must be reported, not silently accepted"
    assert "no rows" in error[-1]["message"]


async def test_a_failed_run_still_writes_its_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The per-row `error` column is the record of what went wrong. Failing the run BEFORE
    # writing it would throw away the only evidence of why every row failed.
    written: list[list[dict[str, Any]]] = []
    code, _ = await _run_main(
        monkeypatch,
        tmp_path,
        rows=[{"question": "a"}],
        scripted=[_fail(RuntimeError("kaboom"))],
        written=written,
    )

    assert code == 1
    assert written, "the results file must be written before the run is failed"
    assert written[-1] == [
        {"custom_id": "row-0", "output": None, "error": "RuntimeError('kaboom')"}
    ]


async def test_the_stderr_failure_reason_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # stderr is a NEW egress for an error message, and the run's console log is captured and
    # stored. A row error carrying the write token must be scrubbed on the way out, like the
    # progress file already was.
    code, _ = await _run_main(
        monkeypatch,
        tmp_path,
        rows=[{"question": "a"}],
        scripted=[_fail(RuntimeError(f"upstream rejected {_TOKEN}"))],
    )

    assert code == 1
    err = capsys.readouterr().err
    assert _TOKEN not in err
    assert "***" in err


async def test_the_local_endpoint_is_called_with_an_explicit_placeholder_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner names its own credential rather than inheriting the VM's.

    The vLLM server it just launched is on loopback and takes no credential. Leaving the key
    unset does NOT mean "send none": the OpenAI client refuses to build a request without one,
    which failed every row of a 2098-row run before any of them reached the server. Saying
    "unauthenticated" explicitly also keeps an OPENAI_API_KEY that happens to be exported on the
    VM from being sent to a local server that never asked for one.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-the-users-real-key")
    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise RuntimeError("stop here — the kwargs are the assertion")

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)
    # The REAL BatchInferenceRunner + LLMClient, so the provider wiring is exercised end to end.
    # Restored from its own import: by this point `ir.BatchInferenceRunner` is the stub.
    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RealBatchRunner)

    await ir.main()

    assert captured["api_base"] == "http://127.0.0.1:8000/v1"
    assert captured["api_key"] == "EMPTY"  # not omitted, and not the ambient key
    assert captured["api_key"] != "sk-the-users-real-key"
