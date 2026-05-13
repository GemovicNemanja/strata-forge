"""Unit tests for `forge.core.repro`."""

from __future__ import annotations

import random
import sys

from forge.core.repro import content_hash, env_snapshot, set_seed


class TestSetSeed:
    def test_python_random_is_reproducible(self) -> None:
        set_seed(42)
        first = [random.random() for _ in range(5)]  # noqa: S311
        set_seed(42)
        second = [random.random() for _ in range(5)]  # noqa: S311
        assert first == second

    def test_different_seeds_differ(self) -> None:
        set_seed(42)
        a = [random.random() for _ in range(5)]  # noqa: S311
        set_seed(43)
        b = [random.random() for _ in range(5)]  # noqa: S311
        assert a != b

    def test_does_not_raise_without_heavy_deps(self) -> None:
        # The function must be a no-op for optional deps; never raise.
        set_seed(0)
        set_seed(2**31 - 1)


class TestContentHash:
    def test_returns_64_char_hex(self) -> None:
        h = content_hash({"a": 1})
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # parses as hex

    def test_stable_across_calls(self) -> None:
        assert content_hash({"a": 1, "b": 2}) == content_hash({"a": 1, "b": 2})

    def test_key_order_invariant(self) -> None:
        # Logically equivalent dicts must hash the same regardless of insertion order.
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    def test_value_change_changes_hash(self) -> None:
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_nested_structures(self) -> None:
        a = {"outer": {"inner": [1, 2, {"k": "v"}]}}
        b = {"outer": {"inner": [1, 2, {"k": "v"}]}}
        assert content_hash(a) == content_hash(b)

    def test_handles_none_bool_int_float_str(self) -> None:
        h = content_hash({"none": None, "true": True, "i": 1, "f": 1.5, "s": "x"})
        assert len(h) == 64


class TestEnvSnapshot:
    def test_contains_core_keys(self) -> None:
        snap = env_snapshot()
        assert snap["python"] == sys.version.split()[0]
        assert isinstance(snap["platform"], str)
        assert isinstance(snap["machine"], str)
        assert isinstance(snap["system"], str)
        assert isinstance(snap["cwd"], str)

    def test_records_installed_packages(self) -> None:
        snap = env_snapshot()
        # pydantic is a core runtime dep — must be present.
        assert snap.get("pkg.pydantic") is not None
        # litellm is also a core dep.
        assert snap.get("pkg.litellm") is not None
        # tenacity is a core dep.
        assert snap.get("pkg.tenacity") is not None

    def test_missing_optional_packages_are_none(self) -> None:
        snap = env_snapshot()
        # `vllm` is behind the `serving` extra — not installed in default env.
        assert "pkg.vllm" in snap
        assert snap["pkg.vllm"] is None

    def test_keys_are_stable(self) -> None:
        # Two consecutive calls return the same set of keys.
        a = set(env_snapshot().keys())
        b = set(env_snapshot().keys())
        assert a == b
