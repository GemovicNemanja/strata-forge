"""Unit tests for `strata_forge.core.repro`."""

from __future__ import annotations

import random
import sys

from strata_forge.core.repro import content_hash, env_snapshot, set_seed


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

    def test_seeds_numpy_when_available(self, monkeypatch: object) -> None:
        # sys.modules-injected fake numpy proves the branch fires when
        # the optional dep is present, without requiring numpy itself.
        import types

        seeded: dict[str, int | None] = {"value": None}

        def _seed(v: int) -> None:
            seeded["value"] = v

        fake_random = types.SimpleNamespace(seed=_seed)
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.random = fake_random  # type: ignore[attr-defined]
        monkeypatch.setitem(  # type: ignore[attr-defined]
            sys.modules, "numpy", fake_numpy
        )
        set_seed(7)
        assert seeded["value"] == 7

    def test_seeds_torch_when_available(self, monkeypatch: object) -> None:
        # Injects a fake torch with a CUDA branch exercised + a CUDA-less
        # branch via two separate calls.
        import types

        events: list[tuple[str, int]] = []

        class _Cuda:
            def __init__(self, available: bool) -> None:
                self._available = available

            def is_available(self) -> bool:
                return self._available

            def manual_seed_all(self, v: int) -> None:
                events.append(("cuda_seed", v))

        # First pass: CUDA available — exercises lines 72-73.
        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = lambda v: events.append(("torch_seed", v))  # type: ignore[attr-defined]
        fake_torch.cuda = _Cuda(available=True)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)  # type: ignore[attr-defined]
        set_seed(11)
        assert ("torch_seed", 11) in events
        assert ("cuda_seed", 11) in events

        # Second pass: CUDA absent — exercises the False branch of line 72.
        events.clear()
        fake_torch_cpu = types.ModuleType("torch")
        fake_torch_cpu.manual_seed = lambda v: events.append(("torch_seed", v))  # type: ignore[attr-defined]
        fake_torch_cpu.cuda = _Cuda(available=False)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch_cpu)  # type: ignore[attr-defined]
        set_seed(13)
        assert ("torch_seed", 13) in events
        assert not any(e[0] == "cuda_seed" for e in events)


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
