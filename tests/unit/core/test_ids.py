"""Unit tests for `strata_forge.core.ids`."""

from __future__ import annotations

import asyncio
import time
import uuid

from strata_forge.core.ids import (
    correlation_id_var,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
    uuid7,
)


class TestUuid7Layout:
    def test_returns_uuid_instance(self) -> None:
        assert isinstance(uuid7(), uuid.UUID)

    def test_version_is_seven(self) -> None:
        for _ in range(20):
            assert uuid7().version == 7

    def test_variant_is_rfc_4122(self) -> None:
        for _ in range(20):
            assert uuid7().variant == uuid.RFC_4122

    def test_hex_length(self) -> None:
        assert len(uuid7().hex) == 32

    def test_str_length(self) -> None:
        # Canonical form: 8-4-4-4-12 = 36 chars with hyphens
        assert len(str(uuid7())) == 36


class TestUuid7Uniqueness:
    def test_consecutive_ids_differ(self) -> None:
        ids = {uuid7() for _ in range(1000)}
        assert len(ids) == 1000

    def test_consecutive_ids_are_time_ordered(self) -> None:
        # Generated in the same tight loop, IDs across millisecond boundaries
        # must be ordered. Within a millisecond, rand_a determines order, so
        # spread the samples across ms by sleeping.
        first = uuid7()
        time.sleep(0.002)
        second = uuid7()
        time.sleep(0.002)
        third = uuid7()
        assert first.int < second.int < third.int


class TestUuid7Timestamp:
    def test_timestamp_matches_clock(self) -> None:
        before_ms = int(time.time() * 1000)
        u = uuid7()
        after_ms = int(time.time() * 1000)
        encoded_ms = u.int >> 80
        # The encoded timestamp must fall within the [before, after] window
        # taken around the call.
        assert before_ms <= encoded_ms <= after_ms


class TestNewCorrelationId:
    def test_returns_hex_string(self) -> None:
        cid = new_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 32
        # Hex-only
        int(cid, 16)

    def test_distinct_each_call(self) -> None:
        ids = {new_correlation_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_does_not_set_contextvar(self) -> None:
        token = set_correlation_id(None)
        try:
            new_correlation_id()
            assert get_correlation_id() is None
        finally:
            correlation_id_var.reset(token)


class TestCorrelationIdContextvar:
    def test_set_and_get(self) -> None:
        token = set_correlation_id("abc123")
        try:
            assert get_correlation_id() == "abc123"
        finally:
            correlation_id_var.reset(token)

    def test_reset_restores_previous(self) -> None:
        outer = set_correlation_id("outer")
        try:
            inner = set_correlation_id("inner")
            assert get_correlation_id() == "inner"
            correlation_id_var.reset(inner)
            assert get_correlation_id() == "outer"
        finally:
            correlation_id_var.reset(outer)

    def test_setting_to_none(self) -> None:
        token = set_correlation_id("something")
        try:
            none_token = set_correlation_id(None)
            assert get_correlation_id() is None
            correlation_id_var.reset(none_token)
            assert get_correlation_id() == "something"
        finally:
            correlation_id_var.reset(token)

    async def test_propagates_across_awaits(self) -> None:
        token = set_correlation_id("traced-task")
        try:
            seen: list[str | None] = []

            async def child() -> None:
                # No explicit passing — contextvar should carry through.
                seen.append(get_correlation_id())
                await asyncio.sleep(0)
                seen.append(get_correlation_id())

            await child()
            assert seen == ["traced-task", "traced-task"]
        finally:
            correlation_id_var.reset(token)

    async def test_isolated_per_task(self) -> None:
        """A child task sees its parent's contextvar but its mutations don't leak out."""
        token = set_correlation_id("parent")
        try:
            results: list[str | None] = []

            async def child(label: str) -> None:
                set_correlation_id(label)
                await asyncio.sleep(0)
                results.append(get_correlation_id())

            await asyncio.gather(child("child-a"), child("child-b"))
            # Each child saw its own value; the parent's binding is untouched.
            assert sorted(filter(None, results)) == ["child-a", "child-b"]
            assert get_correlation_id() == "parent"
        finally:
            correlation_id_var.reset(token)
