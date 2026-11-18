"""Admission's pure half: a request checked against the declared interface, nothing executed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strata_forge.plugins.preflight import InputBinding, PluginRequest, preflight

from .conftest import SHA

if TYPE_CHECKING:
    from strata_forge.plugins.manifest import PluginManifest
    from strata_forge.plugins.pins import AllowedPlugin

STEP_COLUMNS = ("item_id", "step_index", "context", "step")


def _request(**changes: Any) -> PluginRequest:
    data: dict[str, Any] = {
        "entry_point": "prm",
        "inputs": [
            InputBinding(
                role="steps",
                repo_id="acme/steps",
                revision=SHA,
                split="train",
                columns=STEP_COLUMNS,
            )
        ],
        "output_repo_id": "acme/scores",
        "config": {"mode": "fast"},
    }
    data.update(changes)
    return PluginRequest.model_validate(data)


def _status(report: Any) -> dict[str, str]:
    return {c.key: c.status for c in report.checks}


def test_a_good_request_passes_with_defaults_filled(
    manifest: PluginManifest, allowed: AllowedPlugin
) -> None:
    report = preflight(manifest, allowed, _request())
    assert report.ok, report.failures()
    assert report.config == {"mode": "fast", "scale": 1.0}


def test_unknown_or_disallowed_entry_points_fail(
    manifest: PluginManifest, allowed: AllowedPlugin
) -> None:
    assert (
        _status(preflight(manifest, allowed, _request(entry_point="nope")))["entry_point"] == "fail"
    )
    narrowed = allowed.model_copy(update={"entry_points": ("job",)})
    report = preflight(manifest, narrowed, _request())
    assert _status(report)["allowed"] == "fail"
    assert report.config is None


def test_bindings_are_checked(manifest: PluginManifest, allowed: AllowedPlugin) -> None:
    missing = preflight(manifest, allowed, _request(inputs=[]))
    assert "required input 'steps'" in next(c.detail for c in missing.checks if c.key == "inputs")
    extra = _request(
        inputs=[
            InputBinding(
                role="steps",
                repo_id="acme/steps",
                revision=SHA,
                split="train",
                columns=STEP_COLUMNS,
            ),
            InputBinding(role="gold", repo_id="acme/gold", revision=SHA, split="test"),
        ]
    )
    assert _status(preflight(manifest, allowed, extra))["inputs"] == "fail"
    model_with_split = _request(
        inputs=[
            InputBinding(
                role="steps",
                repo_id="acme/steps",
                revision=SHA,
                split="train",
                columns=STEP_COLUMNS,
            ),
            InputBinding(role="verifier", repo_id="acme/prm", revision=SHA, split="train"),
        ]
    )
    assert _status(preflight(manifest, allowed, model_with_split))["inputs"] == "fail"


def test_columns_fail_when_missing_and_are_unknown_when_unread(
    manifest: PluginManifest, allowed: AllowedPlugin
) -> None:
    lacking = _request(
        inputs=[
            InputBinding(
                role="steps", repo_id="acme/steps", revision=SHA, split="train", columns=("step",)
            )
        ]
    )
    assert _status(preflight(manifest, allowed, lacking))["columns"] == "fail"
    unread = _request(
        inputs=[InputBinding(role="steps", repo_id="acme/steps", revision=SHA, split="train")]
    )
    report = preflight(manifest, allowed, unread)
    assert _status(report)["columns"] == "unknown"
    assert report.ok


def test_configuration_is_closed_and_typed(
    manifest: PluginManifest, allowed: AllowedPlugin
) -> None:
    for config in ({"extra": 1}, {"scale": "1"}, {"scale": 99}, {"mode": "medium"}):
        report = preflight(manifest, allowed, _request(config=config))
        assert _status(report)["config"] == "fail", config
    job = preflight(
        manifest,
        allowed,
        _request(
            entry_point="job",
            config={},
            inputs=[
                InputBinding(
                    role="items",
                    repo_id="acme/items",
                    revision=SHA,
                    split="train",
                    columns=("text",),
                )
            ],
        ),
    )
    assert "'label' is required" in next(c.detail for c in job.checks if c.key == "config")


def test_an_output_repository_is_required(manifest: PluginManifest, allowed: AllowedPlugin) -> None:
    assert _status(preflight(manifest, allowed, _request(output_repo_id=None)))["output"] == "fail"


def test_the_manifest_must_be_the_pinned_build(
    manifest: PluginManifest, allowed: AllowedPlugin
) -> None:
    other = allowed.model_copy(update={"pin": allowed.pin.model_copy(update={"version": "9.9.9"})})
    assert _status(preflight(manifest, other, _request()))["manifest"] == "fail"
