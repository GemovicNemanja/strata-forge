"""Unit tests for `strata_forge.compute.task`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from strata_forge.compute.task import ResourceSpec, Task

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------


class TestResourceSpec:
    def test_all_fields_default_to_none(self) -> None:
        spec = ResourceSpec()
        assert spec.cpus is None
        assert spec.memory_gb is None
        assert spec.accelerators is None
        assert spec.cloud is None
        assert spec.region is None
        assert spec.disk_gb is None

    def test_negative_cpus_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResourceSpec(cpus=0)
        with pytest.raises(ValidationError):
            ResourceSpec(cpus=-1)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResourceSpec(unknown="x")  # type: ignore[call-arg]

    def test_is_frozen(self) -> None:
        spec = ResourceSpec(cpus=4)
        with pytest.raises(ValidationError, match="frozen"):
            spec.cpus = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task — construction
# ---------------------------------------------------------------------------


class TestTaskConstruction:
    def test_minimal(self) -> None:
        task = Task(name="x", run="echo hi")
        assert task.name == "x"
        assert task.run == "echo hi"
        assert task.setup == ""
        assert task.num_nodes == 1

    def test_full(self) -> None:
        task = Task(
            name="train",
            run="python train.py",
            setup="pip install -r requirements.txt",
            workdir=".",
            env={"FOO": "1"},
            file_mounts={"/remote/data": "/local/data"},
            resources=ResourceSpec(cpus=8, accelerators="A100:1"),
            num_nodes=2,
            metadata={"experiment": "exp-42"},
        )
        assert task.env == {"FOO": "1"}
        assert task.resources is not None
        assert task.resources.accelerators == "A100:1"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(name="", run="echo hi")

    def test_empty_run_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(name="x", run="")

    def test_zero_num_nodes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(name="x", run="hi", num_nodes=0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(name="x", run="hi", unknown="oops")  # type: ignore[call-arg]

    def test_is_frozen(self) -> None:
        task = Task(name="x", run="echo")
        with pytest.raises(ValidationError, match="frozen"):
            task.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task — YAML round trip
# ---------------------------------------------------------------------------


class TestTaskYaml:
    def test_from_yaml_str_minimal(self) -> None:
        text = """
        name: test
        run: echo hello
        """
        task = Task.from_yaml_str(text)
        assert task.name == "test"
        assert task.run == "echo hello"
        assert task.resources is None

    def test_from_yaml_str_with_resources(self) -> None:
        text = """
        name: training-job
        run: python train.py
        setup: pip install torch
        resources:
          accelerators: A100:1
          memory_gb: 64
        envs:
          WANDB_PROJECT: my-project
        """
        task = Task.from_yaml_str(text)
        assert task.name == "training-job"
        assert task.resources is not None
        assert task.resources.accelerators == "A100:1"
        assert task.resources.memory_gb == 64
        # ``envs:`` plural also works.
        assert task.env == {"WANDB_PROJECT": "my-project"}

    def test_from_yaml_str_env_singular_also_works(self) -> None:
        text = """
        name: x
        run: hi
        env:
          FOO: bar
        """
        task = Task.from_yaml_str(text)
        assert task.env == {"FOO": "bar"}

    def test_unknown_fields_rejected(self) -> None:
        text = """
        name: x
        run: hi
        unknown_field: oops
        """
        with pytest.raises(ValueError, match="unsupported fields"):
            Task.from_yaml_str(text)

    def test_non_mapping_rejected(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            Task.from_yaml_str("- a\n- b\n")

    def test_non_mapping_resources_rejected(self) -> None:
        text = """
        name: x
        run: hi
        resources: not-a-mapping
        """
        with pytest.raises(ValueError, match="resources block"):
            Task.from_yaml_str(text)

    def test_from_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "task.yaml"
        yaml_file.write_text("name: f\nrun: echo file\n")
        task = Task.from_yaml(yaml_file)
        assert task.name == "f"

    def test_to_yaml_round_trip(self) -> None:
        original = Task(
            name="rt",
            run="python x.py",
            setup="pip install foo",
            env={"K": "V"},
            resources=ResourceSpec(cpus=4, accelerators="T4:1"),
            num_nodes=2,
        )
        text = original.to_yaml()
        rebuilt = Task.from_yaml_str(text)
        assert rebuilt == original

    def test_to_yaml_omits_defaults(self) -> None:
        task = Task(name="x", run="echo")
        text = task.to_yaml()
        # Only name + run; no setup, no num_nodes, etc.
        assert "setup" not in text
        assert "num_nodes" not in text
        assert "resources" not in text
        assert "envs" not in text

    def test_empty_yaml_rejected_via_pydantic(self) -> None:
        # An empty mapping fails validation (missing required name/run).
        with pytest.raises(ValidationError):
            Task.from_yaml_str("")
