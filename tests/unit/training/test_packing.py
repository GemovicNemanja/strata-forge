"""Unit tests for `strata_forge.training.packing`."""

from __future__ import annotations

import pytest

from strata_forge.training.packing import PackedSequence, pack_sequences


class TestPackSequences:
    def test_single_short_fits_one_pack(self) -> None:
        packs = pack_sequences([[1, 2, 3]], max_length=10, eos_token_id=0, pad_token_id=9)
        assert len(packs) == 1
        pack = packs[0]
        # 3 source tokens + EOS, padded to length 10.
        assert pack.input_ids == (1, 2, 3, 0, 9, 9, 9, 9, 9, 9)
        assert pack.attention_mask == (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        assert pack.source_indices == (0,)

    def test_multiple_short_pack_together(self) -> None:
        # Three short sequences fit into a single max_length=12 pack.
        packs = pack_sequences(
            [[1, 2], [3, 4], [5, 6]],
            max_length=12,
            eos_token_id=0,
            pad_token_id=9,
        )
        assert len(packs) == 1
        # Three short sequences fit into a single pack with EOS between each.
        assert packs[0].input_ids[:9] == (1, 2, 0, 3, 4, 0, 5, 6, 0)
        assert packs[0].source_indices == (0, 1, 2)

    def test_overflow_spills_to_next_pack(self) -> None:
        packs = pack_sequences(
            [[1, 2, 3], [4, 5, 6]],
            max_length=5,
            eos_token_id=0,
            pad_token_id=9,
        )
        assert len(packs) == 2
        assert packs[0].input_ids == (1, 2, 3, 0, 9)
        assert packs[0].source_indices == (0,)
        assert packs[1].input_ids == (4, 5, 6, 0, 9)
        assert packs[1].source_indices == (1,)

    def test_oversize_sequence_truncated(self) -> None:
        # max_length=5 leaves 4 real tokens after reserving 1 for EOS.
        packs = pack_sequences(
            [[1, 2, 3, 4, 5, 6, 7]],
            max_length=5,
            eos_token_id=0,
            pad_token_id=9,
        )
        # Truncated to 4 tokens + EOS, exactly filling the pack.
        assert packs[0].input_ids == (1, 2, 3, 4, 0)
        assert all(m == 1 for m in packs[0].attention_mask)

    def test_exact_fit_starts_new_pack(self) -> None:
        # First sequence exactly fills a max_length=4 pack; next starts new.
        packs = pack_sequences(
            [[1, 2, 3], [4]],
            max_length=4,
            eos_token_id=0,
            pad_token_id=9,
        )
        assert len(packs) == 2
        assert packs[0].input_ids == (1, 2, 3, 0)
        assert packs[1].input_ids == (4, 0, 9, 9)

    def test_empty_input(self) -> None:
        assert pack_sequences([], max_length=10, eos_token_id=0, pad_token_id=9) == ()

    def test_max_length_too_small(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            pack_sequences([[1]], max_length=1, eos_token_id=0, pad_token_id=9)

    def test_pack_lengths_consistent(self) -> None:
        packs = pack_sequences(
            [[1], [2], [3], [4]],
            max_length=8,
            eos_token_id=0,
            pad_token_id=9,
        )
        for pack in packs:
            assert len(pack.input_ids) == 8
            assert len(pack.attention_mask) == 8

    def test_frozen_dataclass(self) -> None:
        pack = PackedSequence(
            input_ids=(1, 2, 3),
            attention_mask=(1, 1, 1),
            source_indices=(0,),
        )
        with pytest.raises(Exception, match="cannot assign"):
            pack.input_ids = (4, 5, 6)  # type: ignore[misc]
