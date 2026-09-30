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
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import BaseModel, ConfigDict

import strata_forge
from strata_forge.pipelines import SPEC_VERSION, _common
from strata_forge.pipelines._common import (
    REQUIRE_ENGINE_VERSION_ENV,
    UNCHECKED_ENGINE_MESSAGE,
    RunError,
    check_engine_version,
    installed_engine_commit,
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
    # Shaped like a real runner spec: every one of them declares engine_version.
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int = 1
    engine_version: str | None = None


class _BareSpec(BaseModel):
    # What a runner spec that forgot the handshake looks like.
    model_config = ConfigDict(extra="forbid")

    name: str


@pytest.fixture(autouse=True)
def _no_required_engine_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's shell may carry the switch; the tests that exercise it set it themselves.
    monkeypatch.delenv(REQUIRE_ENGINE_VERSION_ENV, raising=False)


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

    def test_the_phase_sink_records_the_stage_when_given_one(self, tmp_path: Path) -> None:
        # `stage` is what lets a consumer render an ordered stepper without pattern-matching the
        # English in `message`, which changes constantly and is not an API.
        writer = JsonlProgressWriter(tmp_path / "p.jsonl")
        phase_sink(writer, None)("Starting the model server", stage="load_model")
        writer.close()
        assert json.loads((tmp_path / "p.jsonl").read_text())["stage"] == "load_model"

    def test_the_phase_sink_leaves_the_stage_unset_by_default(self, tmp_path: Path) -> None:
        # Keeps the plain one-argument call valid for callers outside the runner, whose hooks
        # know nothing about run stages.
        writer = JsonlProgressWriter(tmp_path / "p.jsonl")
        phase_sink(writer, None)("Uploading")
        writer.close()
        assert json.loads((tmp_path / "p.jsonl").read_text())["stage"] is None

    def test_the_phase_sink_carries_gpu_counters(self, tmp_path: Path) -> None:
        # This is where hardware telemetry matters most: loading a model or uploading results
        # can take minutes during which nothing is countable and the gauges are all that moves.
        class _Sampler:
            def sample(self) -> dict[str, float]:
                return {"gpu_util_pct": 94.0}

        writer = JsonlProgressWriter(tmp_path / "p.jsonl")
        phase_sink(writer, None, gpu=cast("Any", _Sampler()))(
            "Loading the model", stage="load_model"
        )
        writer.close()
        assert json.loads((tmp_path / "p.jsonl").read_text())["metrics"] == {"gpu_util_pct": 94.0}

    @pytest.mark.asyncio
    async def test_a_staged_ticking_phase_stamps_every_tick(self, tmp_path: Path) -> None:
        # A consumer that misses one event still learns the stage from the next.
        writer = JsonlProgressWriter(tmp_path / "p.jsonl")
        async with ticking_phase(
            phase_sink(writer, None), "Generating responses", 0.01, stage="run"
        ):
            await asyncio.sleep(0.035)
        writer.close()
        events = [
            json.loads(line)
            for line in (tmp_path / "p.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(events) > 1, "the ticker should have re-stamped at least once"
        assert {e["stage"] for e in events} == {"run"}

    @pytest.mark.asyncio
    async def test_an_unstaged_ticking_phase_accepts_a_one_argument_sink(self) -> None:
        # `serving.py`'s public `on_phase` hook is a plain Callable[[str], None]; forcing a
        # `stage=` keyword onto it would break every caller for a value they never asked for.
        seen: list[str] = []
        async with ticking_phase(seen.append, "Loading", 0.01):
            await asyncio.sleep(0.02)
        assert seen
        assert seen[0] == "Loading"


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

    @pytest.mark.parametrize("raw", ["[]", '"x"', "1", "null"])
    def test_a_non_object_is_a_runner_error(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", raw)
        with pytest.raises(RunError, match="must be a JSON object"):
            load_config(_Spec)


# ------------------------------ the engine version handshake ------------------------------


_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def _run_config(monkeypatch: pytest.MonkeyPatch, engine_version: str | None) -> None:
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG", json.dumps({"name": "x", "engine_version": engine_version})
    )


def _installed_commit(monkeypatch: pytest.MonkeyPatch, commit: str | None) -> None:
    monkeypatch.setattr(_common, "installed_engine_commit", lambda: commit)


class TestEngineVersionHandshake:
    def test_the_spec_version_is_the_package_version(self) -> None:
        # What an orchestrator writes into ``engine_version`` is the version of the package the
        # spec models live in: the field set ships with the version, so the version names it.
        assert strata_forge.__version__ == SPEC_VERSION

    def test_no_claim_is_accepted_and_recorded_as_unchecked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A spec from before the handshake carries no version, and must still run -- but the run
        # record must say the engine was never checked, so a launch a rolled-back control plane
        # sent as null is not mistaken for a checked one.
        monkeypatch.delenv(REQUIRE_ENGINE_VERSION_ENV, raising=False)
        _run_config(monkeypatch, None)
        assert check_engine_version(None) == UNCHECKED_ENGINE_MESSAGE
        path = tmp_path / "progress.jsonl"
        writer = JsonlProgressWriter(str(path))
        try:
            assert load_config(_Spec, writer=writer).engine_version is None
        finally:
            writer.close()
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert [(e["kind"], e["message"]) for e in events] == [("phase", UNCHECKED_ENGINE_MESSAGE)]
        assert f"warning: {UNCHECKED_ENGINE_MESSAGE}" in capsys.readouterr().err

    def test_a_missing_claim_is_accepted_without_a_writer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(REQUIRE_ENGINE_VERSION_ENV, raising=False)
        _run_config(monkeypatch, None)
        assert load_config(_Spec).engine_version is None

    def test_a_matching_claim_is_not_a_warning(self) -> None:
        assert check_engine_version(SPEC_VERSION) is None

    @pytest.mark.parametrize("flag", ["1", "true", "yes", " 1 "])
    def test_a_missing_claim_is_refused_when_the_machine_requires_one(
        self, monkeypatch: pytest.MonkeyPatch, flag: str
    ) -> None:
        # The switch an orchestrator that stamps every spec sets on its machines: from then on
        # a null is the skew the handshake exists to close, not a transition to tolerate.
        monkeypatch.setenv(REQUIRE_ENGINE_VERSION_ENV, flag)
        _run_config(monkeypatch, None)
        with pytest.raises(RunError) as excinfo:
            load_config(_Spec)
        assert "engine version mismatch" in str(excinfo.value)
        assert REQUIRE_ENGINE_VERSION_ENV in str(excinfo.value)

    @pytest.mark.parametrize("flag", ["", "0", "false", "FALSE"])
    def test_the_switch_is_off_for_its_off_values(
        self, monkeypatch: pytest.MonkeyPatch, flag: str
    ) -> None:
        monkeypatch.setenv(REQUIRE_ENGINE_VERSION_ENV, flag)
        _run_config(monkeypatch, None)
        assert load_config(_Spec).engine_version is None

    def test_the_switch_does_not_soften_a_present_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Requiring a claim is about null only; a stale claim is refused with or without it.
        monkeypatch.setenv(REQUIRE_ENGINE_VERSION_ENV, "1")
        _run_config(monkeypatch, SPEC_VERSION)
        assert load_config(_Spec).engine_version == SPEC_VERSION
        _run_config(monkeypatch, "0.0.1")
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_a_spec_class_without_the_field_is_a_bug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A runner spec cannot opt out of the handshake by forgetting to declare the field: that
        # would run unchecked, and a control plane that did stamp the version would be refused
        # for an unknown key, so the omission would surface on a VM instead of here.
        monkeypatch.setenv("STRATA_RUN_CONFIG", '{"name": "x"}')
        with pytest.raises(TypeError, match="_BareSpec does not declare engine_version"):
            load_config(_BareSpec)

    def test_every_runner_spec_declares_the_field(self) -> None:
        # The registry-style guard for the next runner: the spec each runner's load_spec returns
        # must declare engine_version. load_config would refuse the omission anyway, but it
        # belongs in this suite, not on a VM.
        import importlib
        import inspect
        import pkgutil

        import strata_forge.pipelines as pipelines

        runners: dict[str, type[BaseModel]] = {}
        for module_info in pkgutil.iter_modules(pipelines.__path__):
            module = importlib.import_module(f"{pipelines.__name__}.{module_info.name}")
            load_spec = getattr(module, "load_spec", None)
            if load_spec is None:
                continue
            # The return annotation is a string (postponed evaluation) naming a class of the
            # runner's own module; resolved by name rather than get_type_hints, whose evaluation
            # of the parameters would trip on a TYPE_CHECKING-only import.
            returns = inspect.signature(load_spec).return_annotation
            runners[module_info.name] = getattr(module, returns)
        assert set(runners) >= {"inference_runner", "finetune_runner"}
        for name, spec_cls in runners.items():
            assert "engine_version" in spec_cls.model_fields, name
            field = spec_cls.model_fields["engine_version"]
            assert field.annotation == (str | None), name
            assert field.default is None, name

    def test_an_engine_version_that_is_not_a_string_is_refused_before_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", '{"name": "x", "engine_version": 1}')
        with pytest.raises(RunError, match="engine_version must be a string"):
            load_config(_Spec)

    def test_a_stale_version_is_reported_before_an_unknown_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The realistic skew: a newer control plane sends a field this engine does not know AND
        # a version it does not match. The run must blame the stale machine, not the spec.
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG", '{"name": "x", "engine_version": "0.0.1", "new_field": 1}'
        )
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_the_installed_version_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _run_config(monkeypatch, SPEC_VERSION)
        assert load_config(_Spec).engine_version == SPEC_VERSION

    @pytest.mark.parametrize("stale", ["0.0.1", "99.0.0", "", "+" + _COMMIT])
    def test_another_version_is_refused(self, monkeypatch: pytest.MonkeyPatch, stale: str) -> None:
        # The case the handshake exists for: a warm machine still running the engine it
        # installed for an earlier run, handed a spec a newer engine validated (or the reverse).
        _run_config(monkeypatch, stale)
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    @pytest.mark.parametrize(
        "spelling",
        [
            "v{v}",
            "{v}.post0",
            "{v}.0",
            "{v}rc0",
            " {v}",
            "{v} ",
            "{v}\n",
        ],
    )
    def test_a_pep_440_equivalent_spelling_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        # The compare is string equality on purpose. A normalising compare ("0.3" == "0.3.0",
        # "v0.3.0", a post-release) would let a version the control plane never validated
        # against pass, and would do so silently the day someone reached for packaging.version.
        _run_config(monkeypatch, spelling.format(v=SPEC_VERSION))
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_a_short_form_of_the_version_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # "0.3" for "0.3.0": the same PEP 440 version, not the same string.
        short = SPEC_VERSION.rsplit(".", 1)[0]
        assert short != SPEC_VERSION
        _run_config(monkeypatch, short)
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_a_long_claim_is_truncated_in_the_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The claim is echoed for diagnosis; a control plane's is short, and the record should
        # not carry an arbitrarily long one twice (event + stderr).
        claim = "9" * 500
        _run_config(monkeypatch, claim)
        with pytest.raises(RunError) as excinfo:
            load_config(_Spec)
        message = str(excinfo.value)
        assert claim not in message
        assert "9" * 100 + "..." in message
        assert len(message) < 300

    def test_a_commit_claim_is_accepted_when_the_installed_commit_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _installed_commit(monkeypatch, _COMMIT)
        _run_config(monkeypatch, f"{SPEC_VERSION}+{_COMMIT}")
        assert load_config(_Spec).engine_version == f"{SPEC_VERSION}+{_COMMIT}"

    def test_a_commit_claim_is_refused_on_a_different_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every commit of a development branch shares one __version__, so this is the skew a
        # version-only comparison cannot see.
        _installed_commit(monkeypatch, _OTHER_COMMIT)
        _run_config(monkeypatch, f"{SPEC_VERSION}+{_COMMIT}")
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_a_commit_claim_is_refused_on_a_release_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An index install records no commit. It cannot satisfy a commit claim, because nothing
        # says which commit the release was cut from.
        _installed_commit(monkeypatch, None)
        _run_config(monkeypatch, f"{SPEC_VERSION}+{_COMMIT}")
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    @pytest.mark.parametrize(
        "malformed",
        [
            _COMMIT[:7],
            _COMMIT[:12],
            _COMMIT.upper(),
            "",
            "+" + _COMMIT,
            " " + _COMMIT,
            _COMMIT + " ",
        ],
    )
    def test_a_commit_claim_must_be_a_full_lowercase_id(
        self, monkeypatch: pytest.MonkeyPatch, malformed: str
    ) -> None:
        # A prefix would match more than one commit; the handshake compares whole ids only. And
        # an empty claim after the "+" (a SHA build-arg that came through blank) is a malformed
        # claim, never a version-only one: reading it as "no commit" would run whatever the warm
        # machine has, which is the exact failure the commit half exists to catch.
        _installed_commit(monkeypatch, _COMMIT)
        _run_config(monkeypatch, f"{SPEC_VERSION}+{malformed}")
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_an_empty_commit_claim_is_refused_even_on_a_release_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _installed_commit(monkeypatch, None)
        _run_config(monkeypatch, f"{SPEC_VERSION}+")
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_a_drift_between_the_installed_and_executing_versions_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pin resolves against the distribution's metadata; the code that runs reports
        # __version__. A shadowed import or a pyproject bump that missed __init__ makes them
        # disagree, and then the spec matching one of them proves nothing.
        monkeypatch.setattr(_common, "installed_engine_version", lambda: "0.0.1")
        _run_config(monkeypatch, SPEC_VERSION)
        with pytest.raises(RunError) as excinfo:
            load_config(_Spec)
        assert "engine version mismatch" in str(excinfo.value)
        assert f"{SPEC_VERSION} (installed as 0.0.1)" in str(excinfo.value)

    def test_no_installed_distribution_falls_back_to_the_executing_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A source tree on PYTHONPATH with nothing installed has no metadata to disagree with.
        monkeypatch.setattr(_common, "installed_engine_version", lambda: None)
        _run_config(monkeypatch, SPEC_VERSION)
        assert load_config(_Spec).engine_version == SPEC_VERSION

    def test_a_commit_claim_with_a_stale_version_is_refused_before_the_commit_is_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom() -> str | None:
            raise AssertionError("the commit must not be consulted for a stale version")

        monkeypatch.setattr(_common, "installed_engine_commit", _boom)
        _run_config(monkeypatch, f"0.0.1+{_COMMIT}")
        with pytest.raises(RunError, match="engine version mismatch"):
            load_config(_Spec)

    def test_the_mismatch_message_names_both_sides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The reason has to be diagnosable from the failed run's record alone.
        _installed_commit(monkeypatch, _OTHER_COMMIT)
        _run_config(monkeypatch, f"{SPEC_VERSION}+{_COMMIT}")
        with pytest.raises(RunError) as excinfo:
            load_config(_Spec)
        message = str(excinfo.value)
        assert f"{SPEC_VERSION}+{_COMMIT}" in message
        assert f"{SPEC_VERSION}+{_OTHER_COMMIT}" in message


class _FakeDistribution:
    def __init__(self, direct_url: str | None) -> None:
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._direct_url


class TestInstalledEngineCommit:
    """The PEP 610 read behind the commit half of the handshake."""

    def _install(self, monkeypatch: pytest.MonkeyPatch, direct_url: str | None) -> None:
        def _distribution(_name: str) -> _FakeDistribution:
            return _FakeDistribution(direct_url)

        monkeypatch.setattr(_common.metadata, "distribution", _distribution)

    def test_a_vcs_install_reports_its_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(
            monkeypatch,
            json.dumps(
                {
                    "url": "https://github.com/example/strata-forge",
                    "vcs_info": {"vcs": "git", "commit_id": _COMMIT, "requested_revision": "dev"},
                }
            ),
        )
        assert installed_engine_commit() == _COMMIT

    def test_an_index_install_has_no_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # pip writes no direct_url.json for a release installed from an index.
        self._install(monkeypatch, None)
        assert installed_engine_commit() is None

    def test_an_editable_checkout_has_no_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A dir_info URL is a local path, not a commit: the checkout may be dirty or unpushed.
        self._install(
            monkeypatch, json.dumps({"url": "file:///src/forge", "dir_info": {"editable": True}})
        )
        assert installed_engine_commit() is None

    @pytest.mark.parametrize("raw", ["{not json", "[]", '{"vcs_info": "git"}', '{"vcs_info": {}}'])
    def test_unreadable_metadata_is_no_commit(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        self._install(monkeypatch, raw)
        assert installed_engine_commit() is None

    @pytest.mark.parametrize(
        "error",
        [
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            OSError("input/output error"),
        ],
    )
    def test_an_unreadable_file_is_no_commit(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        # read_text swallows only a missing file. A corrupt or unreadable one must still be
        # "no commit", so the run reports the named mismatch rather than a raw exception.
        class _Broken:
            def read_text(self, _filename: str) -> str | None:
                raise error

        def _distribution(_name: str) -> _Broken:
            return _Broken()

        monkeypatch.setattr(_common.metadata, "distribution", _distribution)
        assert installed_engine_commit() is None

    def test_a_missing_distribution_is_no_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _missing(_name: str) -> _FakeDistribution:
            raise _common.metadata.PackageNotFoundError(_name)

        monkeypatch.setattr(_common.metadata, "distribution", _missing)
        assert installed_engine_commit() is None


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

    async def test_a_stale_engine_is_exit_one_with_the_mismatch_as_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The whole handshake, end to end: a spec another engine validated reaches a runner
        # through STRATA_RUN_CONFIG, the run exits 1, and the reason the control plane reads
        # from stderr and the progress file is the mismatch, not whatever stale code did next.
        progress = tmp_path / "p.jsonl"
        monkeypatch.setenv("FORGE_PROGRESS_PATH", str(progress))
        _run_config(monkeypatch, "0.0.1")
        entered: list[bool] = []

        async def _execute(writer: JsonlProgressWriter | None, token: str | None) -> None:
            del writer, token
            load_config(_Spec)
            entered.append(True)

        assert await runner_main(_execute) == 1
        assert entered == []
        assert "engine version mismatch" in progress.read_text()
        assert "run failed: engine version mismatch" in capsys.readouterr().err

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
