"""Unit tests for `strata_forge.prompts.stores.memory`."""

from __future__ import annotations

import pytest

from strata_forge.prompts.registry import PromptNotFoundError, PromptStore
from strata_forge.prompts.stores.memory import InMemoryPromptStore
from strata_forge.prompts.template import PromptTemplate


def _t(name: str, body: str = "hi") -> PromptTemplate:
    return PromptTemplate(name=name, stable_section=body)


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


class TestInterfaceConformance:
    def test_is_promptstore(self) -> None:
        assert isinstance(InMemoryPromptStore(), PromptStore)


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


class TestPut:
    async def test_first_put_returns_version_1(self) -> None:
        store = InMemoryPromptStore()
        version = await store.put(_t("greet"))
        assert version == "1"

    async def test_subsequent_puts_increment(self) -> None:
        store = InMemoryPromptStore()
        v1 = await store.put(_t("greet", "a"))
        v2 = await store.put(_t("greet", "b"))
        v3 = await store.put(_t("greet", "c"))
        assert (v1, v2, v3) == ("1", "2", "3")

    async def test_versions_are_per_name(self) -> None:
        store = InMemoryPromptStore()
        a1 = await store.put(_t("a"))
        b1 = await store.put(_t("b"))
        a2 = await store.put(_t("a"))
        # Each name has its own counter.
        assert (a1, b1, a2) == ("1", "1", "2")

    async def test_identical_bodies_get_distinct_versions(self) -> None:
        # The store doesn't dedupe by content — every put is a new version.
        store = InMemoryPromptStore()
        v1 = await store.put(_t("x", "same"))
        v2 = await store.put(_t("x", "same"))
        assert v1 != v2


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_latest_returns_most_recent(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x", "first"))
        await store.put(_t("x", "second"))
        await store.put(_t("x", "third"))
        latest = await store.get("x")
        assert latest.stable_section == "third"

    async def test_get_specific_version(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x", "first"))
        await store.put(_t("x", "second"))
        v1 = await store.get("x", "1")
        v2 = await store.get("x", "2")
        assert v1.stable_section == "first"
        assert v2.stable_section == "second"

    async def test_unknown_name_raises(self) -> None:
        store = InMemoryPromptStore()
        with pytest.raises(PromptNotFoundError) as info:
            await store.get("missing")
        assert info.value.name == "missing"

    async def test_unknown_version_raises(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x"))
        with pytest.raises(PromptNotFoundError) as info:
            await store.get("x", "999")
        assert info.value.name == "x"
        assert info.value.version == "999"


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------


class TestVersions:
    async def test_returns_newest_first(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x"))
        await store.put(_t("x"))
        await store.put(_t("x"))
        result = await store.versions("x")
        assert result == ["3", "2", "1"]

    async def test_single_version(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x"))
        assert await store.versions("x") == ["1"]

    async def test_unknown_name_raises(self) -> None:
        store = InMemoryPromptStore()
        with pytest.raises(PromptNotFoundError) as info:
            await store.versions("missing")
        assert info.value.name == "missing"


# ---------------------------------------------------------------------------
# list_names
# ---------------------------------------------------------------------------


class TestListNames:
    async def test_empty_store(self) -> None:
        store = InMemoryPromptStore()
        assert await store.list_names() == []

    async def test_returns_sorted_distinct(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("z"))
        await store.put(_t("a"))
        await store.put(_t("m"))
        await store.put(_t("a"))  # second version of 'a' shouldn't dupe
        assert await store.list_names() == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_all_versions(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x"))
        await store.put(_t("x"))
        await store.delete("x")
        with pytest.raises(PromptNotFoundError):
            await store.get("x")

    async def test_delete_specific_version(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x", "first"))
        await store.put(_t("x", "second"))
        await store.delete("x", "1")
        # "1" is gone; "2" still resolves.
        result = await store.get("x", "2")
        assert result.stable_section == "second"
        with pytest.raises(PromptNotFoundError):
            await store.get("x", "1")

    async def test_delete_only_remaining_version_removes_name(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x"))
        await store.delete("x", "1")
        # No versions left → the name itself disappears from list_names.
        assert await store.list_names() == []

    async def test_delete_unknown_name_is_noop(self) -> None:
        store = InMemoryPromptStore()
        # No raise.
        await store.delete("missing")
        await store.delete("missing", "1")  # also no-op

    async def test_delete_unknown_version_on_known_name_raises(self) -> None:
        store = InMemoryPromptStore()
        await store.put(_t("x"))
        with pytest.raises(PromptNotFoundError) as info:
            await store.delete("x", "999")
        assert info.value.name == "x"
        assert info.value.version == "999"

    async def test_version_counter_persists_across_delete(self) -> None:
        # When you delete a version then put again, the counter doesn't reset.
        # This is the intuitive behavior: "version 3" never points to anything
        # other than the third put, even if the second one was deleted in between.
        store = InMemoryPromptStore()
        await store.put(_t("x"))  # v1
        await store.put(_t("x"))  # v2
        await store.delete("x", "1")
        v3 = await store.put(_t("x"))
        assert v3 == "3"


# ---------------------------------------------------------------------------
# End-to-end shape
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_round_trip_preserves_template_identity(self) -> None:
        store = InMemoryPromptStore()
        original = PromptTemplate(
            name="greet",
            stable_section="You are helpful.",
            dynamic_section="Hi {{ name }}",
            dynamic_variables=("name",),
            description="A greeter",
            metadata={"owner": "team"},
        )
        version = await store.put(original)
        retrieved = await store.get("greet", version)
        # Pydantic frozen models compare by value.
        assert retrieved == original
