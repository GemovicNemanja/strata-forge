"""Metadata presence is an observation, never an installed execution capability."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from strata_forge.core import runtime_environment as runtime


@pytest.fixture
def observation(monkeypatch):
    def version(name):
        if name == "strata-forge":
            return "0.3.0"
        raise runtime.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.metadata, "version", version)
    return runtime.capture_runtime_environment()


def refusal(value):
    with pytest.raises(ValueError, match=r"^runtime_environment_invalid$") as caught:
        runtime.canonical_environment_bytes(value)
    assert caught.value.args == ("runtime_environment_invalid",)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_fixed_complete_profile_and_observed_interpreter(observation):
    assert tuple(row.name for row in observation.distributions) == runtime.DISTRIBUTIONS
    assert observation.interpreter.python_version == ".".join(map(str, sys.version_info[:3]))
    assert observation.interpreter.platform == sys.platform
    present = [row for row in observation.distributions if row.status == "present"]
    assert [(row.name, row.version) for row in present] == [("strata-forge", "0.3.0")]
    absent = [row for row in observation.distributions if row.status == "absent"]
    assert len(absent) == 18
    assert all(row.version is None and row.reason == "metadata_not_found" for row in absent)
    assert (
        observation.importability
        == observation.hardware
        == observation.source_revision
        == "not_observed"
    )
    assert observation.image_digest == observation.lock_digest == "not_observed"
    assert observation.execution == "not_assessed"


@pytest.mark.parametrize(
    "bad", [None, True, "", "1" * 129, "1.0\n", "1.0/secret", "https://example.test"]
)
def test_malformed_metadata_is_unavailable_without_echo(monkeypatch, bad):
    monkeypatch.setattr(runtime.metadata, "version", lambda _name: bad)
    result = runtime.capture_runtime_environment()
    assert all(
        row.status == "unavailable" and row.version is None and row.reason == "metadata_invalid"
        for row in result.distributions
    )
    assert b"secret" not in runtime.canonical_environment_bytes(result)


def test_reader_errors_are_unavailable_and_independent(monkeypatch):
    calls = []

    def version(name):
        calls.append(name)
        if name == "torch":
            raise OSError("FICTIONAL_PRIVATE_METADATA_DETAIL")
        if name == "numpy":
            raise runtime.metadata.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(runtime.metadata, "version", version)
    result = runtime.capture_runtime_environment()
    rows = {row.name: row for row in result.distributions}
    assert calls == list(runtime.DISTRIBUTIONS)
    assert rows["torch"].reason == "metadata_unavailable"
    assert rows["numpy"].reason == "metadata_not_found"
    assert rows["vllm"].status == "present"
    assert b"FICTIONAL_PRIVATE_METADATA_DETAIL" not in runtime.canonical_environment_bytes(result)


def test_process_control_is_not_hidden_as_missing_metadata(monkeypatch):
    def stop(_name):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime.metadata, "version", stop)
    with pytest.raises(KeyboardInterrupt):
        runtime.capture_runtime_environment()


def test_canonical_bytes_and_digest_bind_time_and_exact_metadata(observation):
    fixed = observation.model_copy(update={"observed_at": "2026-11-10T12:00:00.000000Z"})
    raw = runtime.canonical_environment_bytes(fixed)
    assert len(raw) <= runtime.MAX_OBSERVATION_BYTES
    assert (
        raw
        == json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )
    assert runtime.environment_observation_sha256(fixed) == hashlib.sha256(raw).hexdigest()
    later = fixed.model_copy(update={"observed_at": "2026-11-10T12:00:00.000001Z"})
    assert runtime.environment_observation_sha256(later) != runtime.environment_observation_sha256(
        fixed
    )
    assert b"cwd" not in raw
    assert b"direct_url" not in raw
    assert b"executable" not in raw
    with pytest.raises(ValidationError, match="frozen_instance"):
        fixed.distributions[0].version = "2.0"


def test_synthetic_portable_golden_and_structural_schema_are_exact():
    fixtures = Path(__file__).with_name("fixtures")
    raw = (fixtures / "runtime_environment_v1.canonical.json").read_bytes()
    value = runtime.RuntimeEnvironmentObservation.model_validate_json(raw, strict=True)
    assert runtime.canonical_environment_bytes(value) == raw
    assert (
        runtime.environment_observation_sha256(value)
        == (fixtures / "runtime_environment_v1.sha256").read_text().strip()
    )
    assert value.model_json_schema() == json.loads(
        (fixtures / "runtime_environment_v1.schema.json").read_bytes()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"profile": "other"},
        {"hardware": "available"},
        {"source_revision": "a" * 40},
        {"extra": "private"},
        {"observed_at": "2026-02-30T00:00:00.000000Z"},
    ],
)
def test_copied_root_values_are_revalidated(observation, change):
    refusal(observation.model_copy(update=change))


@pytest.mark.parametrize(
    "change",
    [
        {"status": "present"},
        {"reason": "observed"},
        {"name": "custom-package"},
        {"version": "1.0/private"},
        {"version": "1.0", "status": "present", "reason": "metadata_not_found"},
    ],
)
def test_copied_nested_observations_are_revalidated(observation, change):
    first = observation.distributions[0].model_copy(update=change)
    refusal(
        observation.model_copy(update={"distributions": (first, *observation.distributions[1:])})
    )


def test_omitted_duplicate_and_reordered_distribution_profiles_are_refused(observation):
    rows = observation.distributions
    for changed in (rows[:-1], (rows[0], *rows[:-1]), tuple(reversed(rows))):
        refusal(observation.model_copy(update={"distributions": changed}))
    interpreter = observation.interpreter.model_copy(update={"pointer_bits": 64.0})
    refusal(observation.model_copy(update={"interpreter": interpreter}))


def test_dictionary_subclass_and_constructed_root_are_not_trusted(observation):
    refusal(observation.model_dump())
    refusal(runtime.RuntimeEnvironmentObservation.model_construct(schema_version=1))

    class Other(runtime.RuntimeEnvironmentObservation):
        pass

    refusal(Other.model_validate(observation.model_dump()))


def test_flag_free_cold_capture_needs_no_optional_import_network_or_process():
    source = Path(__file__).parents[3] / "src"
    program = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
counts={"socket":0,"process":0,"optional_import":0}
blocked={"torch","numpy","transformers","trl","peft","datasets","pyarrow","vllm","litellm","openai","anthropic"}
def audit(event,args):
    kind="socket" if event.startswith("socket.") else "process" if event in {"subprocess.Popen","os.system","os.fork","os.posix_spawn"} else "optional_import" if event=="import" and args[0].split(".")[0] in blocked else None
    if kind:
        counts[kind]+=1
        raise RuntimeError("runtime_observation_unexpected_effect")
sys.addaudithook(audit)
from strata_forge.core.runtime_environment import capture_runtime_environment, canonical_environment_bytes
value=capture_runtime_environment()
raw=canonical_environment_bytes(value)
assert len(value.distributions)==19 and len(raw)<=16384 and not any(counts.values())
print(json.dumps({"counters":counts,"distribution_count":19,"canonical_bytes":len(raw)}))
"""
    completed = subprocess.run(  # noqa: S603 — fixed inline program and the current test interpreter, no shell
        [sys.executable, "-I", "-B", "-c", program, str(source)],
        env={},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["counters"] == {"socket": 0, "process": 0, "optional_import": 0}
