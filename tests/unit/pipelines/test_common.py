"""The plumbing every VM-side runner shares.

These behaviours were tested through the batch-inference runner while it was the only one.
They belong here now: each is a property of running ANY pipeline on someone else's machine —
scrubbing a write token out of every message, re-validating ids at the trust boundary, keeping
a long phase from looking hung, unwinding on SIGTERM, and reporting an outcome exactly once.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, ConfigDict

from strata_forge.pipelines import _common
from strata_forge.pipelines._common import (
    RunError,
    load_config,
    phase_sink,
    results_dir,
    runner_main,
    sanitize,
    ticking_phase,
    validate_repo_id,
)
from strata_forge.training.progress import JsonlProgressWriter

if TYPE_CHECKING:
    from pathlib import Path

_TOKEN = "hf_secretwritetoken1234567890"


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int = 1


# ------------------------------ scrubbing ------------------------------------


class TestSanitize:
    def test_strips_the_known_token_and_token_shapes(self) -> None:
        out = sanitize(f"boom token={_TOKEN} and Bearer abc.def-123 done", _TOKEN)
        assert _TOKEN not in out
        assert "Bearer abc.def-123" not in out
        assert "boom" in out
        assert "done" in out

    def test_strips_a_token_shape_even_with_no_known_token(self) -> None:
        # The pattern is the defense that survives a token this process never saw.
        out = sanitize("leaked hf_abcdefghijklmnop here", None)
        assert "hf_abcdefghijklmnop" not in out

    def test_leaves_ordinary_text_alone(self) -> None:
        assert sanitize("dataset org/name split train", _TOKEN) == "dataset org/name split train"

    def test_the_phase_sink_scrubs_and_caps(self, tmp_path: Path) -> None:
        # The sink is handed to library code whose phase hook is public API, so a phrase from
        # outside the runner gets the same treatment as an error — and cannot grow the file the
        # orchestrator tails without bound.
        writer = JsonlProgressWriter(tmp_path / "p.jsonl")
        phase_sink(writer, _TOKEN)(f"pushing with {_TOKEN} " + "x" * 500)
        writer.close()
        raw = (tmp_path / "p.jsonl").read_text()
        assert _TOKEN not in raw
        message = json.loads(raw)["message"]
        assert len(message) == _common.MAX_PHASE_CHARS

    def test_the_phase_sink_tolerates_no_writer(self) -> None:
        # Progress is optional: a runner launched without a progress path must still run.
        phase_sink(None, _TOKEN)("still going")


# ------------------------------ repo ids -------------------------------------


class TestValidateRepoId:
    @pytest.mark.parametrize("good", ["org/name", "gpt2", "t5-small", "bert-base-uncased", "a/b.c"])
    def test_accepts_owned_and_bare_canonical_ids(self, good: str) -> None:
        assert validate_repo_id(good, "model") == good

    @pytest.mark.parametrize(
        "bad",
        [
            "../etc/passwd",
            "org/../x",
            "org/name\n",
            "org name",
            "https://hf.co/org/name",
            "/org/name",
            "org/name;rm -rf /",
            "",
        ],
    )
    def test_refuses_traversal_schemes_and_metacharacters(self, bad: str) -> None:
        with pytest.raises(RunError, match="invalid model id"):
            validate_repo_id(bad, "model")


# ------------------------------ the config ------------------------------------


class TestLoadConfig:
    def test_parses_into_the_given_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", '{"name": "x", "count": 3}')
        assert load_config(_Spec) == _Spec(name="x", count=3)

    def test_missing_env_is_a_runner_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
        with pytest.raises(RunError, match="STRATA_RUN_CONFIG is not set"):
            load_config(_Spec)

    def test_an_unrecognised_key_is_refused_rather_than_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # extra="forbid" is the reason a spec cannot smuggle an instruction past the runner.
        monkeypatch.setenv("STRATA_RUN_CONFIG", '{"name": "x", "surprise": 1}')
        with pytest.raises(RunError, match="invalid STRATA_RUN_CONFIG"):
            load_config(_Spec)

    def test_malformed_json_is_a_runner_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", "{not json")
        with pytest.raises(RunError, match="invalid STRATA_RUN_CONFIG"):
            load_config(_Spec)


class TestProgressPath:
    def test_prefers_the_spec_over_the_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", '{"progress_path": "from-spec.jsonl"}')
        monkeypatch.setenv("FORGE_PROGRESS_PATH", "from-env.jsonl")
        assert _common.progress_path() == "from-spec.jsonl"

    def test_falls_back_to_the_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", "{}")
        monkeypatch.setenv("FORGE_PROGRESS_PATH", "from-env.jsonl")
        assert _common.progress_path() == "from-env.jsonl"

    def test_an_unparseable_spec_still_resolves_the_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Read defensively: this runs BEFORE validation, so the spec may be anything at all —
        # and a runner with no progress file cannot report why it rejected the spec.
        monkeypatch.setenv("STRATA_RUN_CONFIG", "[1, 2, 3]")
        monkeypatch.setenv("FORGE_PROGRESS_PATH", "from-env.jsonl")
        assert _common.progress_path() == "from-env.jsonl"

    def test_nothing_configured_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
        monkeypatch.delenv("FORGE_PROGRESS_PATH", raising=False)
        assert _common.progress_path() is None


# ------------------------------ output location -------------------------------


class TestResultsDir:
    def test_a_safe_run_id_names_a_directory_under_home(self) -> None:
        out = results_dir("abc-123_DEF", name="strata-inference-results")
        assert str(out).endswith("strata-inference-results/abc-123_DEF")

    @pytest.mark.parametrize("bad", ["../../etc", "a/b", "", None])
    def test_traversal_or_missing_ids_fall_back_to_the_cwd(self, bad: str | None) -> None:
        from pathlib import Path

        assert results_dir(bad, name="strata-inference-results") == Path.cwd()

    def test_the_directory_name_is_the_caller_s(self) -> None:
        out = results_dir("r1", name="strata-finetune-output")
        assert str(out).endswith("strata-finetune-output/r1")


# --------------- a long phase keeps saying it is still going -----------------


class TestTickingPhase:
    """A one-shot phase says a step BEGAN and never that it is still going.

    "Loading the dataset" then sits unchanged for minutes, indistinguishable from a run that has
    hung — which is the question anyone watching is actually asking.
    """

    async def test_it_reports_immediately_without_an_elapsed(self) -> None:
        # Zero is noise; the bare phrase marks the start.
        seen: list[str] = []
        async with ticking_phase(seen.append, "Loading the dataset", interval_s=10):
            pass
        assert seen == ["Loading the dataset"]

    async def test_it_re_stamps_the_phase_while_the_block_runs(self) -> None:
        seen: list[str] = []
        async with ticking_phase(seen.append, "Writing results", interval_s=0.01):
            await asyncio.sleep(0.05)
        assert len(seen) > 1, "a long step must re-report itself"
        assert seen[0] == "Writing results"
        assert all(m.startswith("Writing results (") for m in seen[1:])

    async def test_it_stops_when_the_block_ends(self) -> None:
        # A caption still ticking after its step finished would describe work that is not running.
        seen: list[str] = []
        async with ticking_phase(seen.append, "Loading the dataset", interval_s=0.01):
            await asyncio.sleep(0.03)
        settled = len(seen)
        await asyncio.sleep(0.05)
        assert len(seen) == settled

    async def test_it_stops_when_the_block_raises(self) -> None:
        # Otherwise a failed step leaves a caption ticking forever underneath the error.
        seen: list[str] = []
        with contextlib.suppress(RuntimeError):
            async with ticking_phase(seen.append, "Uploading results", interval_s=0.01):
                await asyncio.sleep(0.03)
                raise RuntimeError("push failed")
        settled = len(seen)
        await asyncio.sleep(0.05)
        assert len(seen) == settled

    async def test_it_ticks_through_a_blocking_step_handed_to_a_thread(self) -> None:
        """The reason a runner uses `to_thread` for its blocking work.

        The ticker is an asyncio task, so a step that blocks the event loop stops the very caption
        that says it is still running — the exact stretch where it is needed most.
        """
        seen: list[str] = []

        def _blocking() -> None:
            time.sleep(0.05)

        async with ticking_phase(seen.append, "Loading the dataset", interval_s=0.01):
            await asyncio.to_thread(_blocking)
        assert len(seen) > 1, "a blocking step must still tick when handed to a thread"


# ------------------------------ the entry point -------------------------------


class TestRunnerMain:
    async def test_success_is_exit_zero_and_closes_the_writer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("FORGE_PROGRESS_PATH", str(tmp_path / "p.jsonl"))
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
        seen: list[Any] = []

        async def _execute(writer: JsonlProgressWriter | None, token: str | None) -> str:
            seen.append((writer, token))
            return "done"

        assert await runner_main(_execute) == 0
        writer, _ = seen[0]
        assert writer is not None
        assert writer._fh.closed  # pyright: ignore[reportPrivateUsage] - lifecycle is the assertion

    async def test_the_write_token_reaches_execute_from_its_own_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_PROGRESS_PATH", raising=False)
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
        monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)
        seen: list[str | None] = []

        async def _execute(writer: JsonlProgressWriter | None, token: str | None) -> None:
            del writer
            seen.append(token)

        await runner_main(_execute)
        assert seen == [_TOKEN]

    async def test_a_failure_is_exit_one_and_the_reason_is_scrubbed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        progress = tmp_path / "p.jsonl"
        monkeypatch.setenv("FORGE_PROGRESS_PATH", str(progress))
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
        monkeypatch.setenv("HF_WRITE_TOKEN", _TOKEN)

        async def _execute(writer: JsonlProgressWriter | None, token: str | None) -> None:
            del writer, token
            msg = f"upload rejected using {_TOKEN}"
            raise RunError(msg)

        assert await runner_main(_execute) == 1
        text = progress.read_text()
        assert '"kind":"error"' in text.replace(" ", "")
        assert _TOKEN not in text
        # Also on stderr, because that is where the control plane reads a failed run's reason.
        assert _TOKEN not in capsys.readouterr().err

    async def test_a_synchronous_raise_inside_execute_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Spec loading happens inside the callable, so it must not escape as an unhandled crash.
        monkeypatch.delenv("FORGE_PROGRESS_PATH", raising=False)
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)

        def _execute(writer: JsonlProgressWriter | None, token: str | None) -> Any:
            del writer, token
            msg = "bad spec"
            raise RunError(msg)

        assert await runner_main(_execute) == 1

    async def test_cancellation_reports_itself_as_cancelled_not_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        progress = tmp_path / "p.jsonl"
        monkeypatch.setenv("FORGE_PROGRESS_PATH", str(progress))
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)

        async def _execute(writer: JsonlProgressWriter | None, token: str | None) -> None:
            del writer, token
            raise asyncio.CancelledError

        assert await runner_main(_execute) == 1
        assert "run cancelled" in progress.read_text()


class TestTerminationTeardown:
    """A cancelled run must shut down whatever it started on the way out.

    Cancelling signals the job's process group, which contains the runner. A model server or a
    training subprocess does NOT run in that group — the backend puts it in a session of its own
    so that killing its tree cannot signal the orchestrator — so the group signal never reaches
    it. The only thing that stops it is the teardown in this process's `finally` blocks, and
    Python's default SIGTERM handling terminates the interpreter where it stands, without
    unwinding. That leaves the GPU held by the very process the cancel existed to stop.

    Driven in a SUBPROCESS on purpose: the failure mode of the mechanism is "SIGTERM kills the
    interpreter", which in-process would take the whole test run with it.
    """

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
    def test_sigterm_unwinds_the_stack_so_finally_blocks_run(self, tmp_path: Path) -> None:
        marker = tmp_path / "torn-down"
        script = f"""
import asyncio, os, signal, sys
from strata_forge.pipelines._common import install_termination_handlers

async def main():
    install_termination_handlers()
    try:
        await asyncio.sleep(60)          # stands in for the work, with the server up
    finally:
        open({str(marker)!r}, "w").write("torn down")   # stands in for serving teardown

async def driver():
    task = asyncio.create_task(main())
    await asyncio.sleep(0.5)             # let the handler install and the sleep begin
    os.kill(os.getpid(), signal.SIGTERM)
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(driver())
"""
        completed = subprocess.run(  # noqa: S603 — fixed interpreter, generated script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert marker.is_file(), (
            "SIGTERM killed the runner outright — the teardown that stops the model server "
            f"never ran.\nstdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
        assert marker.read_text() == "torn down"
