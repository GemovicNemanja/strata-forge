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
import sys
import types
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from strata_forge.compute.batch import BatchInferenceResult
from strata_forge.compute.batch import BatchInferenceRunner as _RealBatchRunner
from strata_forge.core.redact import Redactor
from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._common import (
    LEGACY_TOKEN_ENV,
    LEGACY_TOKEN_MESSAGE,
    REQUIRE_ENGINE_VERSION_ENV,
    UNCHECKED_ENGINE_MESSAGE,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from strata_forge.llm import LLMClient, LLMResponse

_TOKEN = "hf_secretwritetoken1234567890"
# The real preflight, kept before the autouse fixture below replaces it for every other test.
_REAL_CHECK_OUTPUT_REPO = ir._check_output_repo  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _output_repo_is_private(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    # The output-repo preflight asks the Hub; tests that exercise it call the real one directly.
    async def _ok(spec: Any, hf_token: Any) -> None:
        del spec, hf_token

    monkeypatch.setattr(ir, "_check_output_repo", _ok)


@pytest.fixture(autouse=True)
def _no_ambient_secrets(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    # Neither delivery channel may leak in from the shell running the tests.
    monkeypatch.delenv("FORGE_SECRETS_FILE", raising=False)
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)


@pytest.fixture(autouse=True)
def hub_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Stand in for the Hub download the runner makes before serving; record each call.

    Patched on the real ``huggingface_hub`` module, so the runner's own ``HFHubClient`` call path
    (token resolution included) is what is exercised. The snapshot holds one safetensors file.
    """
    calls: list[dict[str, Any]] = []
    snapshot = tmp_path / "hub-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"")

    def _snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    return calls


def _deliver_token(monkeypatch: pytest.MonkeyPatch, directory: Path) -> Path:
    """Hand the runner the write token the way a backend does: a 0600 file it is pointed at."""
    path = directory / ".secrets.json"
    path.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    path.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(path))
    return path


def _ok(
    text: str,
    *,
    latency_ms: float = 100.0,
    output_tokens: int = 10,
    input_tokens: int = 5,
    finish_reason: str = "stop",
    reasoning: str | None = None,
) -> BatchInferenceResult:
    """A successful result shaped like a real ``LLMResponse``.

    ``latency_ms`` and ``usage`` are not decoration: the runner reads both to report per-row
    latency and tokens/s, so a double without them is a double that cannot exercise the path.
    """
    return BatchInferenceResult(
        response=cast(
            "LLMResponse",
            types.SimpleNamespace(
                text=text,
                reasoning=reasoning,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            ),
        )
    )


def _fail(exc: Exception) -> BatchInferenceResult:
    return BatchInferenceResult(error=exc)


def _spec_json(**overrides: Any) -> str:
    # Stamped the way a real launch is: a spec with no claim is accepted but recorded as an
    # unchecked launch, and that record is its own test, not noise in every other one.
    base: dict[str, Any] = {
        "engine_version": SPEC_VERSION,
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


def test_load_spec_refuses_a_spec_validated_by_another_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The handshake is applied to the real spec, not only to the shared loader.
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(engine_version="0.0.1"))
    with pytest.raises(ir.RunError, match="engine version mismatch"):
        ir.load_spec()


def test_load_spec_accepts_a_spec_validated_by_this_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(engine_version=SPEC_VERSION))
    assert ir.load_spec().engine_version == SPEC_VERSION


@pytest.mark.parametrize(
    "bad",
    # "no-slash" is deliberately absent: a bare canonical id (gpt2, t5-small) is VALID, and
    # requiring the slash is what rejected every one of them after a VM had already been spun up.
    ["../evil", "https://x/y", "a b/c", "org/../escape", "org/model\n"],
)
def test_load_spec_rejects_bad_ids(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id=bad))
    with pytest.raises(ir.RunError, match="invalid model id"):
        ir.load_spec()


@pytest.mark.parametrize("canonical", ["gpt2", "t5-small", "distilgpt2", "bert-base-uncased"])
def test_load_spec_accepts_a_bare_canonical_id(
    monkeypatch: pytest.MonkeyPatch, canonical: str
) -> None:
    """A canonical Hugging Face id has no owner, and this used to reject every one of them.

    The control plane's own validator accepts them — with a comment saying the two agree — so a
    run over `gpt2` was accepted at submit, given a VM, bootstrapped, and only THEN rejected here
    as an "invalid model id".
    """
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id=canonical))
    assert ir.load_spec().model_id == canonical


@pytest.mark.parametrize(
    "bad",
    ["/leading", "trailing/", "a//b", ".hidden", "-dash", "a/b/c", "org/model "],
)
def test_load_spec_still_rejects_malformed_ids(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    # Accepting the canonical shape must not have opened the door on anything else: this is the
    # VM's own defence-in-depth check, and the id reaches a Hub client on a box holding a token.
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id=bad))
    with pytest.raises(ir.RunError, match="invalid model id"):
        ir.load_spec()


def test_load_spec_accepts_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    spec = ir.load_spec()
    assert spec.model_id == "org/model"
    assert spec.output_repo_id == "org/out"


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

    counter = {"consumed": 0}

    def _load_dataset(*_a: Any, **_k: Any) -> _NeverEndingDataset:
        return _NeverEndingDataset(counter)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=_load_dataset))
    spec = ir.RunSpec.model_validate_json(_spec_json(hyperparams={"row_limit": 3}))
    rows = ir._load_rows(spec, False)  # pyright: ignore[reportPrivateUsage]
    assert len(rows) == 3
    assert counter["consumed"] == 3  # islice stopped at the cap; the split was NOT materialized


@pytest.mark.parametrize("token", [_TOKEN, False])
def test_load_rows_passes_the_credential_explicitly(
    monkeypatch: pytest.MonkeyPatch, token: str | bool
) -> None:
    # `None` would let `datasets` reach for the VM's own login; the runner never passes it.
    seen: dict[str, Any] = {}

    def _load_dataset(*_a: Any, **kwargs: Any) -> _NeverEndingDataset:
        seen.update(kwargs)
        return _NeverEndingDataset({"consumed": 0})

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=_load_dataset))
    spec = ir.RunSpec.model_validate_json(_spec_json(hyperparams={"row_limit": 1}))
    ir._load_rows(spec, token)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert seen["token"] is token


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
            spec,
            client=cast("LLMClient", object()),
            prompts=prompts,
            custom_ids=ids,
            writer=writer,
            redactor=Redactor(),
        )

    assert out == [
        {
            "custom_id": "row-0",
            "output": "OUT0",
            "reasoning": None,
            "error": None,
            "finish_reason": "stop",
            "input_tokens": 5,
            "output_tokens": 10,
        },
        {
            "custom_id": "row-1",
            "output": None,
            "reasoning": None,
            "error": "RuntimeError('boom')",
            "finish_reason": None,
            "input_tokens": None,
            "output_tokens": None,
        },
    ]
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    step = [e for e in events if e["kind"] == "step"]
    assert step
    metrics = step[-1]["metrics"]
    # Subset rather than equality: the same event also carries throughput and (on a real GPU box)
    # hardware counters, and pinning the whole dict would make every future gauge break this test
    # for a reason that has nothing to do with reconciliation.
    assert metrics["succeeded"] == 1.0
    assert metrics["failed"] == 1.0
    assert step[-1]["stage"] == "run"

    # Only the successful row contributes to latency and tokens — a failure has neither.
    assert metrics["latency_p50_ms"] == 100.0
    assert metrics["rows_per_s"] > 0
    assert metrics["tokens_per_s"] > 0


# ----------------------- main: happy path + no token leak -------------------


async def _always_alive() -> bool:
    """The serving process is up. Tests that kill it mid-batch supply their own probe."""
    return True


@contextlib.asynccontextmanager
async def _fake_serving(*_a: Any, **_kw: Any) -> AsyncGenerator[Any]:
    yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1", is_alive=_always_alive)


async def test_main_happy_path_pushes_and_never_leaks_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    _deliver_token(monkeypatch, tmp_path)

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
        yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1", is_alive=_always_alive)

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

    async def _no_served_len(base_url: str, model_id: str) -> int | None:
        del base_url, model_id
        return None

    _FakeRunner.scripted = [_ok("A")] if scripted is None else scripted
    # The real probe would reach for a model server on loopback; nothing listens there in a test.
    monkeypatch.setattr(ir, "_served_max_model_len", _no_served_len)
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
    _deliver_token(monkeypatch, tmp_path)
    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)

    assert await ir.main() == 0
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    kinds = [e["kind"] for e in events]
    assert [e["message"] for e in events if e["kind"] == "phase"] == [
        "Checking the output repo",
        "Loading the dataset",
        "Downloading the model",
        "Generating responses",
        "Writing results",
        "Uploading results to the Hub",
    ]
    # The split download runs before the first countable milestone, so its phase must too.
    assert kinds.index("phase") < kinds.index("start")
    phase_events = [e for e in events if e["kind"] == "phase"]
    assert all(e["step"] is None and e["total_steps"] is None for e in phase_events)


async def test_main_records_an_unchecked_engine_when_the_spec_makes_no_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A spec from before the handshake still runs, but the run's own record says the engine was
    # never checked: the first event, before any work, and the stderr line the control plane
    # reads. Without it a run that misbehaved on a stale machine is indistinguishable from one
    # that was checked.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress), engine_version=None)
    )
    _deliver_token(monkeypatch, tmp_path)
    monkeypatch.delenv(REQUIRE_ENGINE_VERSION_ENV, raising=False)
    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)

    assert await ir.main() == 0
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert events[0]["kind"] == "phase"
    assert events[0]["message"] == UNCHECKED_ENGINE_MESSAGE
    assert f"warning: {UNCHECKED_ENGINE_MESSAGE}" in capsys.readouterr().err


async def test_main_refuses_a_spec_with_no_claim_when_the_machine_requires_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The orchestrator that stamps every spec sets the switch on its machines, and the
    # transition ends there: a null from a rolled-back control plane is a mismatch, not a launch.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress), engine_version=None)
    )
    _deliver_token(monkeypatch, tmp_path)
    monkeypatch.setenv(REQUIRE_ENGINE_VERSION_ENV, "1")
    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)

    assert await ir.main() == 1
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert [e["kind"] for e in events] == ["error"]
    assert "engine version mismatch" in events[0]["message"]
    assert REQUIRE_ENGINE_VERSION_ENV in events[0]["message"]


async def test_serving_hook_phrases_are_scrubbed_and_capped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `serving_endpoint`'s on_phase is public API, so the phrase reaching the progress file is
    # not necessarily one this module wrote. It gets the same treatment as an error message.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    _deliver_token(monkeypatch, tmp_path)

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
    _deliver_token(monkeypatch, tmp_path)
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


@pytest.mark.parametrize("delivery", ["file", "environment"])
async def test_the_model_server_environment_holds_no_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, delivery: str
) -> None:
    # vLLM is third-party code serving a model a user chose. It gets an allow-listed environment
    # and does not inherit the runner's, so neither the token (however it was delivered), nor
    # the secrets file's path, nor the spec, nor whatever credentials the VM carries reach it.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    if delivery == "file":
        _deliver_token(monkeypatch, tmp_path)
    else:
        monkeypatch.setenv(LEGACY_TOKEN_ENV, _TOKEN)
    monkeypatch.setenv("HF_TOKEN", "hf_ambientvmtoken0123456789")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient0123456789")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))
    monkeypatch.chdir(tmp_path)

    assert await ir.main() == 0
    backend = record["backend"]
    assert backend._env_inherit is False  # pyright: ignore[reportPrivateUsage]
    env = record["task"].env
    assert not record["task"].secrets
    rendered = json.dumps(env)
    for leaked in (_TOKEN, "hf_ambientvmtoken0123456789", "sk-ambient0123456789"):
        assert leaked not in rendered
    for name in ("HF_TOKEN", LEGACY_TOKEN_ENV, "FORGE_SECRETS_FILE", "STRATA_RUN_CONFIG"):
        assert name not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "PATH" in env


# --------------------------- main: the model download ------------------------


async def test_the_model_is_fetched_with_the_delivered_token_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    # The runner downloads the model itself, with the token it was handed, and only the servable
    # files. Forge settings and the VM's own HF_TOKEN hold a different credential that must not be
    # the one used: the constructor argument is the whole of the run's authority.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    _deliver_token(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_ambientvmtoken0123456789")
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 0
    assert len(hub_snapshots) == 1
    call = hub_snapshots[0]
    assert call["repo_id"] == "org/model"
    assert call["token"] == _TOKEN
    assert call["allow_patterns"] == list(ir.SNAPSHOT_PATTERNS)
    assert call["ignore_patterns"] == list(ir.SNAPSHOT_IGNORED)


def test_the_snapshot_never_admits_pickle_checkpoints_or_code() -> None:
    # Through the Hub client's own filter, not a re-implementation of it: what matters is what
    # `snapshot_download` admits, case sensitivity and path handling included.
    # The function `snapshot_download` itself calls; re-exported by `utils` without an `__all__`.
    from huggingface_hub.utils import (
        filter_repo_objects,  # pyright: ignore[reportPrivateImportUsage]
    )

    kept = (
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "spiece.model",
        "qwen.tiktoken",
        "chat_template.jinja",
        "merges.txt",
        "vocab.txt",
    )
    refused = (
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "model.pt",
        "consolidated.00.pth",
        "modeling_custom.py",
        "tokenizer.py",
        "tokenization_custom.py",
        "original/consolidated.00.pth",
        "original/params.json",
        # What a bare `tokenizer*` prefix would have let in beside the weights.
        "tokenizer.so",
        "tokenizer.pyc",
        "tokenizer.sh",
        "tokenizer.joblib",
        "tokenizer.npz",
        "tokenizer.msgpack",
        "tokenizer.pkl.gz",
        "tokenizer.PY",
        "tokenizer.BIN",
        "tokenizer.model.pkl",
        "tokenizer.model.so",
    )
    admitted = set(
        filter_repo_objects(
            [*kept, *refused],
            allow_patterns=list(ir.SNAPSHOT_PATTERNS),
            ignore_patterns=list(ir.SNAPSHOT_IGNORED),
        )
    )
    assert admitted == set(kept)


async def test_a_run_with_no_token_downloads_anonymously(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    # No token delivered means no token used — not the VM's ambient one, not forge's settings.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(output_repo_id=None))
    monkeypatch.setenv("HF_TOKEN", "hf_ambientvmtoken0123456789")
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving({}))

    assert await ir.main() == 0
    assert hub_snapshots[0]["token"] is False


async def test_the_model_server_loads_the_local_snapshot_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    # vLLM is pointed at the directory the runner downloaded, answers to the Hub id the client
    # uses, and runs with the Hub switched off: it never holds the token and fetches nothing.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    _deliver_token(monkeypatch, tmp_path)
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 0
    del hub_snapshots
    task = record["task"]
    snapshot = str(tmp_path / "hub-snapshot")
    assert f"--model {snapshot}" in task.run
    assert "--served-model-name org/model" in task.run
    # Safetensors by name, not vLLM's `auto`, which falls back to a pickle checkpoint that an
    # earlier unfiltered download may have left in the shared cache directory.
    assert "--load-format safetensors" in task.run
    assert task.env["HF_HUB_OFFLINE"] == "1"
    assert not task.secrets
    assert _TOKEN not in json.dumps(task.env)
    assert _TOKEN not in task.run


async def test_a_model_without_safetensors_weights_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    # A repo that ships only pickle weights downloads nothing loadable. Say so by name, before
    # the server starts, rather than as a model-server failure minutes later.
    del hub_snapshots
    (tmp_path / "hub-snapshot" / "model.safetensors").unlink()
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    _deliver_token(monkeypatch, tmp_path)
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 1
    assert "task" not in record  # the server never started
    assert "no safetensors weights" in progress.read_text()


async def test_safetensors_only_in_a_subfolder_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    # The model server reads the directory root. Safetensors in a subfolder beside a root pickle
    # checkpoint (one an earlier unfiltered download left in the shared cache folder) would pass a
    # recursive check while the server loaded the pickle.
    del hub_snapshots
    snapshot = tmp_path / "hub-snapshot"
    (snapshot / "model.safetensors").unlink()
    (snapshot / "pytorch_model.bin").write_bytes(b"pickle")
    (snapshot / "sub").mkdir()
    (snapshot / "sub" / "model.safetensors").write_bytes(b"weights")
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    _deliver_token(monkeypatch, tmp_path)
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 1
    assert "task" not in record  # the server never started
    assert "no safetensors weights" in progress.read_text()


async def test_a_failed_model_download_names_the_model_and_scrubs_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    _deliver_token(monkeypatch, tmp_path)
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    def _gated(**kwargs: Any) -> str:
        msg = f"403 Forbidden: access to this repo is restricted (token {kwargs['token']})"
        raise OSError(msg)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _gated)

    assert await ir.main() == 1
    text = progress.read_text()
    assert "could not download the model org/model" in text
    assert _TOKEN not in text
    assert "task" not in record


async def test_main_accepts_the_deprecated_env_token_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    monkeypatch.setenv(LEGACY_TOKEN_ENV, _TOKEN)
    pushed: list[str] = []

    async def _push(spec: Any, results_path: Any, token: str) -> str:
        del results_path
        pushed.append(token)
        return cast("str", spec.output_repo_id)

    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)
    monkeypatch.setattr(ir, "_push_results", _push)

    assert await ir.main() == 0
    assert pushed == [_TOKEN]
    text = progress.read_text()
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert events[0]["message"] == LEGACY_TOKEN_MESSAGE
    assert _TOKEN not in text


async def test_main_error_path_scrubs_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress)))
    _deliver_token(monkeypatch, tmp_path)

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
    _deliver_token(monkeypatch, tmp_path)
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
        {
            "custom_id": "row-0",
            "output": None,
            "reasoning": None,
            "error": "RuntimeError('kaboom')",
            "finish_reason": None,
            "input_tokens": None,
            "output_tokens": None,
        }
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


async def test_the_error_column_pushed_with_the_results_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The per-row `error` column is written into the results file and pushed to the Hub with
    # it, so an exception quoting the write token must not carry it there. Encoded forms too:
    # an HTTP error usually quotes the request URL.
    quoted = _TOKEN.replace("_", "%5F")
    written: list[list[dict[str, Any]]] = []
    code, _ = await _run_main(
        monkeypatch,
        tmp_path,
        rows=[{"question": "a"}, {"question": "b"}, {"question": "c"}],
        scripted=[
            _ok("A"),
            _fail(RuntimeError(f"upstream rejected {_TOKEN} for org")),
            _fail(RuntimeError(f"401 for https://host/x?token={quoted}")),
        ],
        written=written,
    )

    assert code == 0
    rows = written[-1]
    assert rows[1]["error"] == "RuntimeError('upstream rejected *** for org')"
    assert rows[2]["error"] == "RuntimeError('401 for https://host/x?token=***')"
    assert not any(_TOKEN in str(row) or quoted in str(row) for row in rows)


async def test_an_error_row_loses_credential_shapes_no_token_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A redactor holding no value still removes the credential shapes: a key this process was
    # never given is caught in the column too.
    foreign = "sk-ant-api03-AbCdEfGhIjKlMnOpQrSt"
    spec = ir.RunSpec.model_validate_json(_spec_json())
    prompts, ids = ir._build_requests(spec, [{"question": "a"}])  # pyright: ignore[reportPrivateUsage]
    _FakeRunner.scripted = [_fail(RuntimeError(f"bad key {foreign}"))]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)

    out = await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
        spec,
        client=cast("LLMClient", object()),
        prompts=prompts,
        custom_ids=ids,
        writer=None,
        redactor=Redactor(),
    )
    assert out[0]["error"] == "RuntimeError('bad key ***')"


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
    _deliver_token(monkeypatch, tmp_path)
    # The REAL BatchInferenceRunner + LLMClient, so the provider wiring is exercised end to end.
    # Restored from its own import: by this point `ir.BatchInferenceRunner` is the stub.
    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RealBatchRunner)

    await ir.main()

    assert captured["api_base"] == "http://127.0.0.1:8000/v1"
    assert captured["api_key"] == "EMPTY"  # not omitted, and not the ambient key
    assert captured["api_key"] != "sk-the-users-real-key"


# ------------------- the template must actually use the dataset ---------------


class TestTemplateCoverage:
    """A placeholder with no mapping renders literally, which silently ruins a whole run.

    Every row then gets the byte-identical, row-independent prompt; the model answers it N times;
    every row succeeds; and the run reports `succeeded` with N copies of an answer to the literal
    text. Nothing downstream can notice — the results file holds {custom_id, output, error}, so
    neither the rendered prompt nor the source row is in it.
    """

    def test_a_braced_mapping_key_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The exact trap: the launch form's key field hinted "{placeholder}", so the mapping was
        # written with braces. `_PLACEHOLDER_RE` captures the name WITHOUT them, so the key can
        # never match -- while the old validation passed, because it only checked that the mapping's
        # VALUE ("question") was a real column.
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG",
            _spec_json(template="Q: {question}", column_mapping={"{question}": "question"}),
        )
        with pytest.raises(ir.RunError, match="no column_mapping entry"):
            ir.load_spec()

    def test_an_empty_mapping_with_a_placeholder_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The launch form permits submitting with no mapping rows filled in at all.
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG", _spec_json(template="Q: {question}", column_mapping={})
        )
        with pytest.raises(ir.RunError, match="no column_mapping entry"):
            ir.load_spec()

    def test_the_error_names_what_is_missing_and_what_is_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # It fails before the dataset download and before the GPU, so the message is the whole
        # diagnosis -- it has to say enough to fix the run without a second launch.
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG",
            _spec_json(template="{a} and {b}", column_mapping={"a": "col_a"}),
        )
        with pytest.raises(ir.RunError) as info:
            ir.load_spec()
        assert "'b'" in str(info.value)  # the unmapped one
        assert "'a'" in str(info.value)  # what IS mapped
        assert "without braces" in str(info.value).lower()

    def test_a_fully_mapped_template_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG",
            _spec_json(template="Q: {question}", column_mapping={"question": "question"}),
        )
        assert ir.load_spec().template == "Q: {question}"

    def test_a_template_with_no_placeholders_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Row-independent by INTENT is a different thing from row-independent by accident, and
        # only the accident is worth refusing.
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG", _spec_json(template="Say hello.", column_mapping={})
        )
        assert ir.load_spec().template == "Say hello."

    def test_brace_shapes_that_are_not_placeholders_do_not_trip_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `_PLACEHOLDER_RE` only matches a bare {name}, so JSON-ish instructions in a prompt are
        # not placeholders and must not be demanded of the mapping.
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG",
            _spec_json(
                template='Reply as {"answer": str} for {q}. Not {a b} nor {}.',
                column_mapping={"q": "question"},
            ),
        )
        assert ir.load_spec() is not None


# --------------- results must survive the push that was meant to move them ---


async def test_results_are_written_outside_the_workdir_even_when_pushing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The orchestrator deletes the per-run workdir on EVERY terminal state.

    Writing the parquet there and pushing from it means a failed upload — an expired token, a
    rate limit, a network blip — takes the entire run's output with it: hours of generation gone
    at the last step, with nothing left to retry from.
    """
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress), run_id="run123")
    )
    _deliver_token(monkeypatch, tmp_path)
    outdirs: list[Path] = []

    def _write(rows: Any, outdir: Path) -> Path:
        del rows
        outdirs.append(outdir)
        return outdir / "results.parquet"

    async def _push(spec: Any, results_path: Any, token: str) -> str:
        del results_path, token
        return cast("str", spec.output_repo_id)

    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)
    monkeypatch.setattr(ir, "_write_results", _write)
    monkeypatch.setattr(ir, "_push_results", _push)
    monkeypatch.chdir(tmp_path)  # on the VM this is the workdir that gets deleted

    assert await ir.main() == 0
    assert outdirs, "results were never written"
    # The cleanup-surviving dir named by the run id — NOT the cwd the orchestrator wipes.
    assert str(outdirs[-1]).endswith("strata-inference-results/run123")
    assert outdirs[-1] != tmp_path


async def test_a_failed_push_says_where_the_results_are(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The run still fails — they were asked for on the Hub and are not there — but a failure at
    # the last step of a long run is exactly when it matters that the output was not lost.
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG", _spec_json(progress_path=str(progress), run_id="run123")
    )
    _deliver_token(monkeypatch, tmp_path)

    async def _push_boom(spec: Any, results_path: Any, token: str) -> str:
        del spec, results_path, token
        msg = "429 rate limited"
        raise RuntimeError(msg)

    _mock_main_deps(monkeypatch, tmp_path, _fake_serving)
    monkeypatch.setattr(ir, "_push_results", _push_boom)

    assert await ir.main() == 1
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    error = [e for e in events if e["kind"] == "error"]
    assert error
    # WHERE (the path _write_results returned — the stub's, here) and WHY, in one sentence.
    assert "results are on the VM at" in error[-1]["message"]
    assert "results.parquet" in error[-1]["message"]
    assert "429 rate limited" in error[-1]["message"]


# ------------- a model server that dies AFTER it came up ---------------------


async def test_a_dead_server_stops_the_batch_instead_of_timing_out_every_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Readiness is checked once, and then nothing watched the server again.

    It can die at any point after that — an OOM on a long prompt, a CUDA fault. Every row from
    then on fails against a socket nobody is listening on, and the client RETRIES each one, so a
    dead server became a long expensive silence instead of an error: the rest of the run spent
    timing out one row at a time, and the failure that eventually surfaced described a connection
    rather than the crash behind it.
    """
    spec = ir.RunSpec.model_validate_json(_spec_json(hyperparams={"progress_chunk": 2}))
    prompts, ids = ir._build_requests(  # pyright: ignore[reportPrivateUsage]
        spec, [{"question": c} for c in "abcdef"]
    )
    _FakeRunner.scripted = [_ok("A"), _ok("B")]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)

    calls = {"n": 0}

    async def _dies_after_the_first_chunk() -> bool:
        calls["n"] += 1
        return False

    progress = tmp_path / "progress.jsonl"
    with (
        ir.JsonlProgressWriter(str(progress)) as writer,
        pytest.raises(ir.RunError, match="model server died"),
    ):
        await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
            spec,
            client=cast("LLMClient", object()),
            prompts=prompts,
            custom_ids=ids,
            writer=writer,
            is_alive=_dies_after_the_first_chunk,
            redactor=Redactor(),
        )

    # Stopped at the SECOND chunk: the first one ran before anything could be known about it.
    assert calls["n"] == 1


async def test_a_live_server_runs_every_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The guard must not cost a healthy run anything. A status the backend cannot report counts as
    # alive upstream, so a flaky probe can never kill a run that is working.
    spec = ir.RunSpec.model_validate_json(_spec_json(hyperparams={"progress_chunk": 2}))
    prompts, ids = ir._build_requests(  # pyright: ignore[reportPrivateUsage]
        spec, [{"question": c} for c in "abcd"]
    )
    _FakeRunner.scripted = [_ok("A"), _ok("B")]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)

    progress = tmp_path / "progress.jsonl"
    with ir.JsonlProgressWriter(str(progress)) as writer:
        out = await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
            spec,
            client=cast("LLMClient", object()),
            prompts=prompts,
            custom_ids=ids,
            writer=writer,
            is_alive=_always_alive,
            redactor=Redactor(),
        )
    assert len(out) == 4


async def test_the_batch_runs_without_a_liveness_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `is_alive` is optional: a caller serving its own endpoint elsewhere still gets a batch.
    spec = ir.RunSpec.model_validate_json(_spec_json(hyperparams={"progress_chunk": 2}))
    prompts, ids = ir._build_requests(  # pyright: ignore[reportPrivateUsage]
        spec, [{"question": c} for c in "abcd"]
    )
    _FakeRunner.scripted = [_ok("A"), _ok("B")]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)

    progress = tmp_path / "progress.jsonl"
    with ir.JsonlProgressWriter(str(progress)) as writer:
        out = await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
            spec,
            client=cast("LLMClient", object()),
            prompts=prompts,
            custom_ids=ids,
            writer=writer,
            redactor=Redactor(),
        )
    assert len(out) == 4


# ----------------------------- pinned revisions + manifest ------------------

_SHA = "9216db5781bf21249d130ec9da846c4624c16137"


@pytest.mark.parametrize("bad", ["main", "v1.0", _SHA[:12], _SHA.upper(), _SHA + "0", ""])
def test_a_model_revision_that_is_not_a_full_commit_is_refused(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # A branch or tag moves and an abbreviated id is ambiguous: a run pinned by either is not
    # reproducible, so the spec is refused before anything is downloaded.
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_revision=bad))
    with pytest.raises(ir.RunError, match="full 40-character"):
        ir.load_spec()


async def test_a_pinned_model_revision_is_the_one_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_revision=_SHA))
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving({}))

    assert await ir.main() == 0
    assert hub_snapshots[0]["revision"] == _SHA


async def test_an_unpinned_run_downloads_the_default_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_snapshots: list[dict[str, Any]]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving({}))

    assert await ir.main() == 0
    assert "revision" not in hub_snapshots[0]


def test_seed_and_top_k_reach_the_model_server_as_request_fields() -> None:
    # top_k is not a chat-completions parameter, so it must travel in the body itself.
    hp = ir.Hyperparams(seed=20261104, top_k=20)
    assert ir._sampling_extras(hp) == {  # pyright: ignore[reportPrivateUsage]
        "openai_compat": {"seed": 20261104, "extra_body": {"top_k": 20}}
    }
    assert ir._sampling_extras(ir.Hyperparams()) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("field", [{"seed": -1}, {"top_k": 0}, {"top_n": 3}])
def test_out_of_range_or_unknown_sampling_keys_are_refused(field: dict[str, int]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - Pydantic's own error type
        ir.Hyperparams.model_validate(field)


async def test_sampling_extras_are_sent_with_every_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[dict[str, Any]] = []

    class _Recording(_FakeRunner):
        async def run(self, prompts: Any, **kwargs: Any) -> tuple[BatchInferenceResult, ...]:
            seen.append(kwargs)
            return await super().run(prompts)

    _FakeRunner.scripted = [_ok("A")]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _Recording)
    spec = ir.RunSpec.model_validate_json(
        _spec_json(hyperparams={"seed": 7, "top_k": 20, "temperature": 0.7, "top_p": 0.8})
    )
    prompts, ids = ir._build_requests(spec, [{"question": "a"}])  # pyright: ignore[reportPrivateUsage]
    await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
        spec,
        client=cast("LLMClient", object()),
        prompts=prompts,
        custom_ids=ids,
        writer=None,
        redactor=Redactor(),
    )
    assert seen[0]["temperature"] == 0.7
    assert seen[0]["top_p"] == 0.8
    assert seen[0]["provider_extras"] == {"openai_compat": {"seed": 7, "extra_body": {"top_k": 20}}}


def test_the_snapshot_commit_is_read_from_the_cache_directory_name(tmp_path: Path) -> None:
    cached = tmp_path / "models--org--model" / "snapshots" / _SHA
    cached.mkdir(parents=True)
    assert ir.snapshot_commit(cached) == _SHA  # pyright: ignore[reportPrivateUsage]
    # A layout that does not name a commit records none rather than a guess.
    assert ir.snapshot_commit(tmp_path / "hub-snapshot") is None  # pyright: ignore[reportPrivateUsage]


async def test_the_manifest_records_pins_versions_and_counts_but_no_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    template = "SECRET-PROMPT-TEXT {q}"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG",
        _spec_json(
            model_revision=_SHA,
            dataset_commit_sha="8978b4c1c8a030706e837a239dd870a8fe8de8a4",
            template=template,
            hyperparams={"seed": 1, "top_k": 20, "max_tokens": 4096},
        ),
    )
    _mock_main_deps(
        monkeypatch,
        tmp_path,
        _recording_serving({}),
        rows=[{"question": "a"}, {"question": "b"}, {"question": "c"}],
        scripted=[
            _ok("ROW-TEXT-A", input_tokens=3, output_tokens=7),
            _ok("ROW-TEXT-B", input_tokens=4, output_tokens=9, finish_reason="length"),
            _fail(RuntimeError("x")),
        ],
    )

    assert await ir.main() == 0
    raw = (tmp_path / "manifest.json").read_text()
    manifest = json.loads(raw)
    assert manifest["schema"] == "strata.run.manifest/v1"
    assert manifest["kind"] == "inference"
    assert set(manifest["serving"]) == {"max_model_len", "dtype", "flags", "harmony_date"}
    assert "--load-format" in manifest["serving"]["flags"]
    assert manifest["serving"]["harmony_date"] is None  # not a harmony model
    assert manifest["model"] == {"id": "org/model", "requested_revision": _SHA, "commit": _SHA}
    assert manifest["dataset"]["requested_revision"] == "8978b4c1c8a030706e837a239dd870a8fe8de8a4"
    assert manifest["engine"]["version"] == SPEC_VERSION
    assert set(manifest["engine"]["packages"]) == {
        "vllm",
        "transformers",
        "torch",
        "datasets",
        "huggingface-hub",
    }
    assert manifest["hyperparams"]["top_k"] == 20
    assert manifest["counts"] == {
        "rows": 3,
        "succeeded": 2,
        "failed": 1,
        "finish_reasons": {"length": 1, "stop": 1},
        "input_tokens": 7,
        "output_tokens": 16,
    }
    # Digest, never text: neither the template nor a row's output rides out in the manifest.
    assert manifest["template_sha256"] == ir.hashlib.sha256(template.encode()).hexdigest()
    assert "SECRET-PROMPT-TEXT" not in raw
    assert "ROW-TEXT" not in raw


class _Hub:
    """A Hub client double: records uploads and answers visibility from ``public``."""

    public: ClassVar[set[str]] = set()
    calls: ClassVar[list[tuple[str, Any]]] = []

    def __init__(self, *, token: str) -> None:
        assert token == _TOKEN

    async def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("create_repo", (repo_id, kwargs)))

    async def is_private(self, repo_id: str, **_kw: Any) -> bool:
        return repo_id not in _Hub.public

    async def upload_files(self, files: Any, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("upload_files", (dict(files), repo_id, kwargs)))

    async def upload_file(self, *_a: Any, **_kw: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("results and manifest go up in one commit, never file by file")


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> type[_Hub]:
    _Hub.public = set()
    _Hub.calls = []
    monkeypatch.setattr(ir, "HFHubClient", _Hub)
    return _Hub


async def test_results_and_manifest_go_up_in_one_commit(tmp_path: Path, hub: type[_Hub]) -> None:
    results = tmp_path / "results.parquet"
    results.write_bytes(b"")
    (tmp_path / "manifest.json").write_text("{}")
    spec = ir.RunSpec.model_validate_json(_spec_json())

    assert await ir._push_results(spec, results, _TOKEN) == "org/out"  # pyright: ignore[reportPrivateUsage]
    uploads = [c for c in hub.calls if c[0] == "upload_files"]
    assert len(uploads) == 1
    files, repo_id, kwargs = uploads[0][1]
    # Exactly this run's two files, nothing else from a possibly shared directory.
    assert files == {"results.parquet": results, "manifest.json": tmp_path / "manifest.json"}
    assert repo_id == "org/out"
    assert kwargs["repo_type"] == "dataset"
    # Created private when missing.
    assert hub.calls[0] == (
        "create_repo",
        ("org/out", {"repo_type": "dataset", "private": True, "exist_ok": True}),
    )


async def test_a_public_output_repo_is_refused_at_push(tmp_path: Path, hub: type[_Hub]) -> None:
    hub.public = {"org/out"}
    results = tmp_path / "results.parquet"
    results.write_bytes(b"")
    spec = ir.RunSpec.model_validate_json(_spec_json())

    with pytest.raises(ir.RunError, match="is public"):
        await ir._push_results(spec, results, _TOKEN)  # pyright: ignore[reportPrivateUsage]
    assert not [c for c in hub.calls if c[0] == "upload_files"]


async def test_a_public_output_repo_is_refused_before_any_gpu_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: type[_Hub]
) -> None:
    hub.public = {"org/out"}
    monkeypatch.setattr(ir, "_check_output_repo", _REAL_CHECK_OUTPUT_REPO)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    _deliver_token(monkeypatch, tmp_path)
    loaded: list[bool] = []

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        del spec, token
        loaded.append(True)
        return [{"question": "a"}]

    _mock_main_deps(monkeypatch, tmp_path, _recording_serving({}))
    monkeypatch.setattr(ir, "_load_rows", _rows)

    assert await ir.main() == 1
    assert not loaded  # refused before the dataset download, the model download or the server


async def test_a_run_that_does_not_push_skips_the_output_preflight(hub: type[_Hub]) -> None:
    spec = ir.RunSpec.model_validate_json(_spec_json())
    await _REAL_CHECK_OUTPUT_REPO(spec, None)
    assert hub.calls == []


async def test_an_unpinned_run_records_the_commit_it_served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sha = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    cached = tmp_path / "models--org--model" / "snapshots" / sha
    cached.mkdir(parents=True)
    (cached / "model.safetensors").write_bytes(b"")

    def _download(**kwargs: Any) -> str:
        del kwargs
        return str(cached)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving({}))

    assert await ir.main() == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["model"] == {"id": "org/model", "requested_revision": None, "commit": sha}


@pytest.mark.parametrize("bad", [{"seed": 2**63}, {"seed": True}, {"top_k": True}, {"seed": "1"}])
def test_seed_and_top_k_are_strict_bounded_integers(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - Pydantic's own error type
        ir.Hyperparams.model_validate(bad)


async def test_seed_and_top_k_reach_the_wire_through_the_real_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole client stack down to the transport call: what the model server would receive.
    import litellm

    from strata_forge.llm import LLMClient as RealClient
    from strata_forge.llm.providers.config import OpenAICompatConfig

    sent: dict[str, Any] = {}

    async def _acompletion(**kwargs: Any) -> Any:
        sent.update(kwargs)
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    provider = ir.OpenAICompatProvider(
        OpenAICompatConfig(base_url="http://127.0.0.1:8000/v1", api_key=ir.SecretStr("x"))
    )
    client = RealClient(
        model="org/model", provider="openai_compat", provider_clients={"openai_compat": provider}
    )
    hp = ir.Hyperparams(seed=7, top_k=20, temperature=0.7, top_p=0.8)
    await client.complete(
        messages=[ir.UserMessage(content="hi")],
        temperature=hp.temperature,
        top_p=hp.top_p,
        provider_extras=ir._sampling_extras(hp),  # pyright: ignore[reportPrivateUsage]
    )
    assert sent["seed"] == 7
    assert sent["extra_body"] == {"top_k": 20}
    assert sent["temperature"] == 0.7
    assert sent["top_p"] == 0.8


async def test_a_hub_failure_preparing_the_output_repo_names_the_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Down:
        def __init__(self, *, token: str) -> None:
            del token

        async def create_repo(self, *_a: Any, **_kw: Any) -> None:
            raise RuntimeError("503 from the Hub")

    monkeypatch.setattr(ir, "HFHubClient", _Down)
    spec = ir.RunSpec.model_validate_json(_spec_json())
    with pytest.raises(ir.RunError, match="could not prepare the output repo org/out: 503"):
        await _REAL_CHECK_OUTPUT_REPO(spec, _TOKEN)


# ------------------------- reasoning models (gpt-oss) -------------------------


def _gpt_oss_snapshot(tmp_path: Path) -> None:
    """Make the stubbed Hub snapshot a gpt-oss checkpoint."""
    (tmp_path / "hub-snapshot" / "config.json").write_text(json.dumps({"model_type": "gpt_oss"}))


@pytest.mark.parametrize(
    "bad",
    [
        {"reasoning_effort": "high"},  # no parser: nothing reads the effort
        {"reasoning_effort": "high", "reasoning_parser": "qwen3"},  # qwen3 has no effort levels
        {"reasoning_effort": "max", "reasoning_parser": "openai_gptoss"},
        {"reasoning_parser": "deepseek_r1"},  # not on the allow-list
        {"reasoning_parser": "openai_gptoss --trust-remote-code"},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 1.5},
        {"max_num_batched_tokens": 0},
        {"max_num_batched_tokens": True},
        {"harmony_date": "2026-11-18"},  # no harmony parser
        {"harmony_date": "2026-11-18", "reasoning_parser": "qwen3"},
        *(
            {"harmony_date": bad, "reasoning_parser": "openai_gptoss"}
            for bad in ("2026-02-30", "2026-1-05", "20261118", "2026-11-18T00:00", "2026-W47-3", "")
        ),
    ],
)
def test_reasoning_and_memory_knobs_are_validated(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - Pydantic's own error type
        ir.Hyperparams.model_validate(bad)


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_rides_in_the_request_body(effort: str) -> None:
    hp = ir.Hyperparams.model_validate(
        {"reasoning_parser": "openai_gptoss", "reasoning_effort": effort, "top_k": 5}
    )
    assert ir._sampling_extras(hp) == {  # pyright: ignore[reportPrivateUsage]
        "openai_compat": {"extra_body": {"top_k": 5, "reasoning_effort": effort}}
    }


def test_serve_args_carry_the_parser_and_memory_flags() -> None:
    hp = ir.Hyperparams.model_validate(
        {
            "reasoning_parser": "openai_gptoss",
            "gpu_memory_utilization": 0.95,
            "max_num_batched_tokens": 1024,
        }
    )
    assert ir._serve_args(hp) == (  # pyright: ignore[reportPrivateUsage]
        "--load-format",
        "safetensors",
        "--reasoning-parser",
        "openai_gptoss",
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-batched-tokens",
        "1024",
    )
    # Nothing declared, nothing added: an existing run's command line is unchanged.
    assert ir._serve_args(ir.Hyperparams()) == ("--load-format", "safetensors")  # pyright: ignore[reportPrivateUsage]


async def test_rows_keep_reasoning_out_of_the_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeRunner.scripted = [_ok("42", reasoning="The user asks for 6*7."), _ok("plain")]
    monkeypatch.setattr(ir, "BatchInferenceRunner", _FakeRunner)
    spec = ir.RunSpec.model_validate_json(_spec_json())
    prompts, ids = ir._build_requests(spec, [{"question": "a"}, {"question": "b"}])  # pyright: ignore[reportPrivateUsage]
    out = await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
        spec,
        client=cast("LLMClient", object()),
        prompts=prompts,
        custom_ids=ids,
        writer=None,
        redactor=Redactor(),
    )
    assert [(r["output"], r["reasoning"]) for r in out] == [
        ("42", "The user asks for 6*7."),
        ("plain", None),
    ]


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_the_servers_reasoning_field_reaches_the_reasoning_column_through_the_real_client(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    # vLLM names the field `reasoning` (`reasoning_content` before it renamed it); the whole client
    # stack down to the transport must keep it apart from the final answer, and send the effort.
    import litellm

    from strata_forge.llm import LLMClient as RealClient
    from strata_forge.llm.providers.config import OpenAICompatConfig

    sent: dict[str, Any] = {}

    async def _acompletion(**kwargs: Any) -> Any:
        sent.update(kwargs)
        message = {"role": "assistant", "content": "Paris.", field: "France's capital is Paris."}
        return litellm.ModelResponse(
            choices=[{"message": message, "finish_reason": "stop"}],
            usage={"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
        )

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    provider = ir.OpenAICompatProvider(
        OpenAICompatConfig(base_url="http://127.0.0.1:8000/v1", api_key=ir.SecretStr("x"))
    )
    client = RealClient(
        model="openai/gpt-oss-20b",
        provider="openai_compat",
        provider_clients={"openai_compat": provider},
    )
    hp = ir.Hyperparams.model_validate(
        {"reasoning_parser": "openai_gptoss", "reasoning_effort": "low"}
    )
    response = await client.complete(
        messages=[ir.UserMessage(content="Capital of France?")],
        provider_extras=ir._sampling_extras(hp),  # pyright: ignore[reportPrivateUsage]
    )
    assert response.text == "Paris."
    assert response.reasoning == "France's capital is Paris."
    assert sent["extra_body"] == {"reasoning_effort": "low"}


async def test_a_gpt_oss_run_serves_with_the_harmony_parser_and_records_its_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _gpt_oss_snapshot(tmp_path)
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG",
        _spec_json(
            model_id="openai/gpt-oss-120b",
            model_revision=_SHA,
            hyperparams={
                "reasoning_parser": "openai_gptoss",
                "reasoning_effort": "high",
                "gpu_memory_utilization": 0.95,
                "max_num_batched_tokens": 1024,
                "harmony_date": "2026-11-18",
            },
        ),
    )
    record: dict[str, Any] = {}
    written: list[list[dict[str, Any]]] = []
    _mock_main_deps(
        monkeypatch,
        tmp_path,
        _recording_serving(record),
        scripted=[_ok("FINAL", reasoning="ANALYSIS")],
        written=written,
    )

    assert await ir.main() == 0
    run = record["task"].run
    assert "--reasoning-parser openai_gptoss" in run
    assert "--gpu-memory-utilization 0.95" in run
    assert "--max-num-batched-tokens 1024" in run
    assert written[0][0]["output"] == "FINAL"
    assert written[0][0]["reasoning"] == "ANALYSIS"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    flags = manifest["serving"]["flags"]
    assert flags[0] == "--host"
    assert flags[flags.index("--reasoning-parser") + 1] == "openai_gptoss"
    assert str(tmp_path / "hub-snapshot") not in flags  # a machine path, not a run fact
    assert manifest["hyperparams"]["reasoning_effort"] == "high"
    # The requested date pins the system message, and the manifest says which date that was.
    assert record["task"].env["VLLM_SYSTEM_START_DATE"] == "2026-11-18"
    assert manifest["serving"]["harmony_date"] == "2026-11-18"
    assert manifest["model"]["commit"] == _SHA
    # Row text stays out of the manifest, the reasoning included.
    assert "ANALYSIS" not in json.dumps(manifest)


async def test_a_gpt_oss_model_without_the_harmony_parser_is_refused_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without the parser every row's output would carry the analysis channel.
    monkeypatch.chdir(tmp_path)
    _gpt_oss_snapshot(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id="openai/gpt-oss-20b"))
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 1
    assert "task" not in record  # the model server never started
    assert "reasoning_parser='openai_gptoss'" in capsys.readouterr().err


async def test_the_harmony_parser_on_another_model_is_refused_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hub-snapshot" / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG", _spec_json(hyperparams={"reasoning_parser": "openai_gptoss"})
    )
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 1
    assert "task" not in record
    assert "model_type='qwen3'" in capsys.readouterr().err


async def test_an_undated_gpt_oss_run_pins_and_records_the_machines_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Left to the model server, the date is read per request: a run crossing midnight would state
    # two dates. The runner takes it once, pins it, and records it.
    monkeypatch.chdir(tmp_path)
    _gpt_oss_snapshot(tmp_path)

    monkeypatch.setattr(ir, "_today", lambda: "2026-11-19")
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG",
        _spec_json(
            model_id="openai/gpt-oss-20b", hyperparams={"reasoning_parser": "openai_gptoss"}
        ),
    )
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 0
    assert record["task"].env["VLLM_SYSTEM_START_DATE"] == "2026-11-19"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["serving"]["harmony_date"] == "2026-11-19"


async def test_a_model_without_harmony_gets_no_date_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
    record: dict[str, Any] = {}
    _mock_main_deps(monkeypatch, tmp_path, _recording_serving(record))

    assert await ir.main() == 0
    assert "VLLM_SYSTEM_START_DATE" not in record["task"].env
