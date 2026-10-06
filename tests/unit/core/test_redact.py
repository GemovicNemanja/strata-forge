"""Secret redaction, the one implementation every surfaced string goes through.

The properties that matter are stated as Hypothesis properties rather than examples: a secret
split at any offset across any number of pieces never survives, streaming never changes the
output, short values are refused, and the encoded forms a log is likely to carry are caught. The
examples below the properties pin each credential shape and the text that must stay readable.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import pickle
import re
import unicodedata
import urllib.parse
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import SecretStr

from strata_forge.core import Redactor as ExportedRedactor
from strata_forge.core.errors import ValidationError
from strata_forge.core.redact import (
    DEFAULT_PATTERNS,
    MIN_SECRET_CHARS,
    PLACEHOLDER,
    Redactor,
    TokenPattern,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_HF = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz01234567"
_PEM_BODY = (
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUA\n"
)

# A secret never contains the placeholder's character: otherwise "*** + the text after it" could
# spell the secret without the secret ever having been in the input there.
_CHARS = st.characters(exclude_characters="*", exclude_categories=("Cs",))
_SECRETS = st.text(alphabet=_CHARS, min_size=MIN_SECRET_CHARS, max_size=120).filter(
    lambda s: len(s.strip()) >= MIN_SECRET_CHARS
)
_NOISE = st.text(alphabet=_CHARS, max_size=300)
# Fragments that sit on the edges of every pattern, so random text exercises partial and
# overlapping matches, private-key blocks with and without footers, URL and bearer shapes, a
# word ending in "sk" before `sk-`, and whitespace that may join two redacted runs.
_SHAPES = st.lists(
    st.sampled_from(
        [
            "hf_",
            "sk-",
            "ant-",
            "ghp_",
            "github_pat_",
            "AKIA",
            "eyJ",
            ".",
            "Bearer ",
            "bearer\t",
            "://",
            ":",
            "@",
            "/",
            "-----BEGIN PRIVATE KEY-----",
            "-----BEGIN RSA PRIVATE KEY-----",
            "-----END PRIVATE KEY-----",
            "-----BEGIN",
            "-----",
            "a1B2c3D4",
            "fla",
            "Z",
            "0",
            " ",
            "\n",
            "\r\n",
            "    ",
            "_",
            "-",
        ]
    ),
    max_size=400,
).map("".join)


def _split(text: str, cuts: list[int]) -> list[str]:
    points = sorted({min(c, len(text)) for c in cuts})
    pieces: list[str] = []
    prev = 0
    for point in points:
        pieces.append(text[prev:point])
        prev = point
    pieces.append(text[prev:])
    return pieces


def _streamed(redactor: Redactor, pieces: list[str]) -> str:
    stream = redactor.stream()
    return "".join(stream.feed(piece) for piece in pieces) + stream.flush()


# ------------------------------ properties -----------------------------------


class TestStreamingProperties:
    @settings(max_examples=300, deadline=None)
    @given(
        secret=_SECRETS,
        before=_NOISE,
        after=_NOISE,
        cuts=st.lists(st.integers(min_value=0, max_value=800), max_size=10),
    )
    def test_a_secret_split_anywhere_never_survives(
        self, secret: str, before: str, after: str, cuts: list[int]
    ) -> None:
        redactor = Redactor([secret])
        text = before + secret + after
        out = _streamed(redactor, _split(text, cuts))
        assert secret not in out
        assert out == redactor.redact(text)

    @settings(max_examples=300, deadline=None)
    @given(text=_SHAPES, cuts=st.lists(st.integers(min_value=0, max_value=6000), max_size=12))
    def test_streaming_never_changes_the_output(self, text: str, cuts: list[int]) -> None:
        # The stream's whole claim: a piece boundary moves WHEN output is released, never WHAT.
        redactor = Redactor([_HF])
        assert _streamed(redactor, _split(text, cuts)) == redactor.redact(text)

    def test_a_token_cut_at_every_offset_never_survives(self) -> None:
        redactor = Redactor([_HF])
        text = f"pushing with {_HF} to the hub"
        for offset in range(len(text) + 1):
            out = _streamed(redactor, [text[:offset], text[offset:]])
            assert _HF not in out
            assert out == "pushing with *** to the hub"

    def test_a_span_found_early_does_not_cover_text_before_it(self) -> None:
        # A match is found once its start is final, but what it redacts can begin later than
        # that. What is carried to the next piece is the span itself, not "everything up to its
        # end", which would blank the readable text between.
        redactor = Redactor()
        text = "Bearer abcdefgh12 " + "x" * 600
        expected = "Bearer *** " + "x" * 600
        for cut in (1, 2, 7, redactor.max_len - 1, redactor.max_len, redactor.max_len + 1):
            assert _streamed(redactor, [text[:cut], text[cut:]]) == expected

    def test_a_redacted_run_joins_the_next_across_any_cut(self) -> None:
        # The whitespace after a run is held back until a later run either joins it or can no
        # longer, so where the pieces break never changes how many placeholders come out.
        key = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nQyNTUxOQAAACD0aW5n\n"
        redactor = Redactor([key])
        text = "line one\r\nb3BlbnNzaC1rZXktdjEAAAAA\r\n  QyNTUxOQAAACD0aW5n\r\nline two"
        expected = "line one\r\n***\r\nline two"
        assert redactor.redact(text) == expected
        for offset in range(len(text) + 1):
            assert _streamed(redactor, [text[:offset], text[offset:]]) == expected

    def test_a_word_ending_in_sk_is_read_across_any_cut(self) -> None:
        # The stream keeps one character before what it has not decided, so the left boundary
        # of `sk-` sees the same character whatever the piece boundaries.
        redactor = Redactor()
        # Long enough that the decided point passes every position, one character at a time.
        tail = " " + "z" * redactor.max_len
        text = "pip install flask-SQLAlchemy-Extension2 then sk-proj-Ab3dEf6hIj9kLmN0pQrSt done"
        expected = "pip install flask-SQLAlchemy-Extension2 then *** done"
        assert redactor.redact(text + tail) == expected + tail
        assert _streamed(redactor, list(text + tail)) == expected + tail

    def test_one_character_at_a_time(self) -> None:
        text = f"a\n-----BEGIN PRIVATE KEY-----\n{_PEM_BODY}-----END PRIVATE KEY-----\nb {_HF} c"
        redactor = Redactor([_HF])
        assert _streamed(redactor, list(text)) == "a\n***\nb *** c"


class TestValues:
    @given(value=st.text(max_size=MIN_SECRET_CHARS - 1))
    def test_short_values_are_refused_without_being_echoed(self, value: str) -> None:
        with pytest.raises(ValidationError) as caught:
            Redactor([value])
        with pytest.raises(ValidationError) as reference:
            Redactor(["x"])
        # The same message whatever the value: it cannot carry the value or its length.
        assert str(caught.value) == str(reference.value)

    def test_a_short_value_is_refused_among_long_ones(self) -> None:
        with pytest.raises(ValidationError):
            Redactor([_HF, "short"])

    @pytest.mark.parametrize("value", ["pw12\n\n\n\n\n", "   pw12   ", "**********", " ******** "])
    def test_padding_and_masks_do_not_pass_for_a_secret(self, value: str) -> None:
        # Padding does not make a short secret safe to match, and a mask (what `str()` of a
        # pydantic secret prints) would leave the real secret in place while redacting `*`s.
        with pytest.raises(ValidationError) as caught:
            Redactor([value])
        with pytest.raises(ValidationError) as reference:
            Redactor(["x"])
        assert str(caught.value) == str(reference.value)

    def test_a_pydantic_secret_is_unwrapped_not_stringified(self) -> None:
        quoted = _HF.replace("_", "%5F")
        out = Redactor([SecretStr(_HF)]).redact(f"GET /x?t={quoted}")
        assert out == "GET /x?t=***"
        with pytest.raises(ValidationError):
            Redactor([SecretStr(str(SecretStr(_HF)))])

    def test_a_value_that_is_not_text_is_refused(self) -> None:
        with pytest.raises(TypeError):
            Redactor([b"hf_bytesbytesbytes"])  # type: ignore[list-item]

    @settings(max_examples=200, deadline=None)
    @given(
        secret=_SECRETS,
        prefix=st.binary(max_size=7),
        suffix=st.binary(max_size=7),
        urlsafe=st.booleans(),
    )
    def test_base64_is_caught_at_any_alignment(
        self, secret: str, prefix: bytes, suffix: bytes, urlsafe: bool
    ) -> None:
        # As in `Basic base64(user:token)`: the value sits at an arbitrary byte offset of a larger
        # encoded blob, so only the characters its own bits determine are predictable. Those are
        # recomputed here from the bit arithmetic, independently of how the redactor derives them.
        raw = secret.encode()
        blob_bytes = prefix + raw + suffix
        blob = (base64.urlsafe_b64encode if urlsafe else base64.b64encode)(blob_bytes).decode()
        first = -(-8 * len(prefix) // 6)
        last = 8 * (len(prefix) + len(raw)) // 6
        determined = blob[first:last]
        out = Redactor([secret]).redact(f"Authorization: Basic {blob}")
        assert determined not in out

    @settings(max_examples=200, deadline=None)
    @given(secret=_SECRETS)
    def test_url_json_and_repr_encodings_are_caught(self, secret: str) -> None:
        redactor = Redactor([secret])
        quoted = urllib.parse.quote(secret, safe="")
        plus = urllib.parse.quote_plus(secret, safe="")
        lower = re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), quoted)
        escaped = json.dumps(secret)[1:-1]
        raw_json = json.dumps(secret, ensure_ascii=False)[1:-1]
        error = repr(RuntimeError(f"auth failed: {secret}"))
        # Every non-alphanumeric character escaped, which a URL may legally do even to `_`.
        every = "".join(
            c if c.isascii() and c.isalnum() else "".join(f"%{b:02X}" for b in c.encode())
            for c in secret
        )
        for form, text in [
            (every, f"GET /x?t={every}"),
            (quoted, f"GET https://example.test/x?t={quoted}&a=1"),
            (plus, f"POST body=t={plus}"),
            (lower, f"GET /x?t={lower}"),
            (escaped, json.dumps({"message": f"failed with {secret}"})),
            (raw_json, json.dumps({"message": f"failed with {secret}"}, ensure_ascii=False)),
            (repr(secret)[1:-1], error),
        ]:
            assert form not in redactor.redact(text)

    @pytest.mark.parametrize("value_form", ["NFC", "NFD", "NFKC", "NFKD"])
    @pytest.mark.parametrize("text_form", ["NFC", "NFD"])
    def test_a_value_matches_whichever_normal_form_is_printed(
        self, value_form: str, text_form: str
    ) -> None:
        secret = "pässwörd-émigré"
        redactor = Redactor([unicodedata.normalize(value_form, secret)])  # type: ignore[arg-type]
        printed = unicodedata.normalize(text_form, secret)  # type: ignore[arg-type]
        out = redactor.redact(f"key={printed} end")
        assert out == "key=*** end"

    def test_a_compatibility_form_of_the_value_is_matched(self) -> None:
        # The value holds a ligature; a program that normalised it prints plain letters.
        redactor = Redactor(["\ufb01le-pässwörd"])
        assert redactor.redact("key=file-pässwörd end") == "key=*** end"

    def test_a_multi_line_value_is_caught_line_by_line(self) -> None:
        key = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nQyNTUxOQAAACD0aW5n\n"
        known = "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n"
        redactor = Redactor([key, known])
        out = redactor.redact("line one\r\nb3BlbnNzaC1rZXktdjEAAAAA\r\nQyNTUxOQAAACD0aW5n\r\n")
        # One placeholder for the lines together: one per line would count them.
        assert out == "line one\r\n***\r\n"
        assert "AAAAC3Nz" not in redactor.redact(f"host key {known.strip()} pinned")

    @settings(max_examples=100, deadline=None)
    @given(
        lines=st.lists(
            st.text(alphabet=st.characters(categories=("L", "N")), min_size=8, max_size=40),
            min_size=2,
            max_size=8,
        ),
        stored=st.sampled_from(["\n", "\r\n"]),
        printed=st.sampled_from(["\n", "\r\n", "\r", "\n    "]),
    )
    def test_a_multi_line_value_prints_as_one_placeholder_whatever_its_line_endings(
        self, lines: list[str], stored: str, printed: str
    ) -> None:
        # How many placeholders come out must not say how many lines (about how large a key)
        # the value had, whichever line endings it was printed with.
        redactor = Redactor([stored.join(lines)], patterns=())
        assert redactor.redact(f"before {printed.join(lines)} after") == "before *** after"

    def test_a_long_value_is_covered_whole_and_in_part(self) -> None:
        secret = "".join(chr(ord("a") + (i * 7) % 26) + str(i % 10) for i in range(200))
        redactor = Redactor([secret], patterns=())
        assert redactor.redact(f"<{secret}>") == f"<{PLACEHOLDER}>"
        # A clipped print: every full 64-character window inside it is gone.
        partial = secret[37:300]
        out = redactor.redact(f"<{partial}>")
        assert all(secret[i : i + 64] not in out for i in range(len(secret) - 63))

    @settings(max_examples=100, deadline=None)
    @given(first=_SECRETS, second=_SECRETS)
    def test_the_output_does_not_encode_a_secrets_length(self, first: str, second: str) -> None:
        template = "before {} after"
        out_first = Redactor([first], patterns=()).redact(template.format(first))
        out_second = Redactor([second], patterns=()).redact(template.format(second))
        assert out_first == out_second == "before *** after"


# ------------------------------ token shapes ---------------------------------


class TestDefaultPatterns:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (f"token={_HF}", "token=***"),
            ("key sk-proj-Ab3dEf6hIj9kLmN0pQrStUvWx_yz-12 set", "key *** set"),
            ("x-api-key: sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv-wxyz_0123", "x-api-key: ***"),
            ("ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 ok", "*** ok"),
            ("gho_AbCdEfGhIjKlMnOpQrStUvWx ok", "*** ok"),
            ("github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz ok", "*** ok"),
            ("id AKIAIOSFODNN7EXAMPLE end", "id *** end"),
            ("id ASIAIOSFODNN7EXAMPLE end", "id *** end"),
            (
                "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9 x",
                "jwt *** x",
            ),
            (
                "clone https://user:pa%40ss@github.com/o/r.git",
                "clone https://***@github.com/o/r.git",
            ),
            ("Authorization: Bearer abc.def-123_456~7+8/9=", "Authorization: Bearer ***"),
            ("authorization: bearer\tAbCdEfGh12345678", "authorization: bearer\t***"),
            # A URL parser splits the authority at its LAST `@`, so a raw `@` in the password
            # is still password; and the user may be empty.
            ("https://alice:p@ssw0rdXYZ@host/x", "https://***@host/x"),
            ("redis://:hunter2hunter2@cache:6379", "redis://***@cache:6379"),
            ("q=a%3Dsk-proj-Ab3dEf6hIj9kLmN0pQrStUvWx end", "q=a%3D*** end"),
        ],
    )
    def test_each_credential_shape_is_redacted(self, text: str, expected: str) -> None:
        assert Redactor().redact(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Bearer AAAAAAAAAAAABearer TOKENTOKENTOKEN", "Bearer ***"),
            ("hf_AAAAAAAAhf_BBBBBBBBBBBB", "***"),
            ("sk-proj-Ab3dEf6hIj9kLmN0sk-ant-Zy9xWv8uTs7rQp6", "***"),
        ],
    )
    def test_a_credential_glued_onto_another_is_redacted_too(
        self, text: str, expected: str
    ) -> None:
        # A search that resumed at the end of the first match would start inside the second
        # one's prefix and miss it.
        assert Redactor().redact(text) == expected

    @pytest.mark.parametrize("kind", ["", "RSA ", "EC ", "DSA ", "OPENSSH ", "ENCRYPTED ", "PGP "])
    def test_private_key_blocks_are_redacted_whole(self, kind: str) -> None:
        block = "BLOCK" if kind == "PGP " else ""
        tail = f" {block}" if block else ""
        pem = f"-----BEGIN {kind}PRIVATE KEY{tail}-----\n{_PEM_BODY}-----END {kind}PRIVATE KEY{tail}-----"
        assert Redactor().redact(f"before\n{pem}\nafter") == "before\n***\nafter"

    def test_two_key_blocks_leave_the_text_between_them(self) -> None:
        # Each footer closes only the blocks opened before it.
        pem = f"-----BEGIN PRIVATE KEY-----\n{_PEM_BODY}-----END PRIVATE KEY-----"
        assert Redactor().redact(f"{pem}\nbetween\n{pem}\nafter") == "***\nbetween\n***\nafter"

    def test_a_key_block_whose_footer_was_clipped_is_redacted_to_the_end(self) -> None:
        out = Redactor().redact(f"key: -----BEGIN PRIVATE KEY-----\n{_PEM_BODY}[truncated")
        assert out == "key: ***"

    @pytest.mark.parametrize("footer", ["", "-----END PRIVATE KEY-----"])
    def test_a_key_block_stops_at_its_window(self, footer: str) -> None:
        # A header with no footer in reach must not blank the rest of a long log: past a window
        # larger than any real key, the text is released again.
        header = "-----BEGIN PRIVATE KEY-----"
        body = "A" * 20_000
        for redactor in (Redactor(), Redactor([_HF])):
            # Enough text after the footer that it is decided in the same pass as the window.
            after = "\n" + "visible " * 100
            text = f"{header}{body}{footer}{after}"
            out = redactor.redact(text)
            assert out == PLACEHOLDER + "A" * (20_000 - 16_384) + footer + after
            assert _streamed(redactor, _split(text, [5, 9_000, 16_400, 20_010])) == out

    @pytest.mark.parametrize(
        "text",
        [
            "dataset org/name split train",
            "pip install scikit-learn sk-learn",
            "hf_short",
            "a Bearer token",
            "http://127.0.0.1:8000/v1/models",
            "-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIU\n-----END CERTIFICATE-----",
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5 user@host",
            "ssh://git@github.com:22/org/repo",
            # Words ending in "sk" before a hyphen, and lower-case names: ids, not keys.
            "pip install flask-sqlalchemy-extension2 Flask-SQLAlchemy-Extension2",
            "facebook/mask-generation-pipeline-v2 task-specific-fine-tuning",
            "whisk-recipes-dataset-large and sk-learn-compatible-estimators",
        ],
    )
    def test_ordinary_text_is_left_alone(self, text: str) -> None:
        assert Redactor().redact(text) == text

    def test_patterns_can_be_turned_off(self) -> None:
        assert Redactor(patterns=()).redact(_HF) == _HF

    def test_every_default_pattern_respects_its_bound(self) -> None:
        # The stream trusts `max_chars`; a default that could exceed it would be a bypass.
        for pattern in DEFAULT_PATTERNS:
            assert pattern.max_chars <= Redactor().max_len
            hostile = "://" + "a" * 500 + ":" + "b" * 500 + "@ Bearer " + "c" * 900
            for match in pattern.regex.finditer(hostile * 2 + _HF * 20):
                assert match.end() - match.start() <= pattern.max_chars

    def test_a_custom_pattern_that_overruns_its_bound_is_refused(self) -> None:
        sloppy = TokenPattern("sloppy", re.compile(r"secret-[a-z]+"), 10)
        with pytest.raises(ValidationError, match="sloppy"):
            Redactor(patterns=[sloppy]).redact("secret-abcdefghijklmnop")


# ------------------------------ the stream -----------------------------------


class TestStream:
    def test_holds_back_exactly_max_len_minus_one(self) -> None:
        redactor = Redactor([_HF])
        text = "x" * 5_000
        stream = redactor.stream()
        released = stream.feed(text)
        assert released == "x" * (len(text) - (redactor.max_len - 1))
        assert stream.flush() == "x" * (redactor.max_len - 1)

    def test_flush_starts_a_new_stream(self) -> None:
        stream = Redactor([_HF]).stream()
        assert stream.feed(_HF[:10]) + stream.flush() == _HF[:10]
        assert stream.feed(_HF[10:]) + stream.flush() == _HF[10:]

    @settings(max_examples=200, deadline=None)
    @given(cut=st.integers(min_value=0, max_value=len(_HF)), lead=st.text(" .,\n", max_size=40))
    def test_a_gap_never_lets_either_half_of_a_cut_token_through(self, cut: int, lead: str) -> None:
        # A console read that dropped bytes: the token's two halves arrive on either side of a
        # hole and neither is recognisable alone.
        stream = Redactor([_HF]).stream()
        out = stream.feed(lead + _HF[:cut]) + stream.gap() + stream.feed(_HF[cut:] + lead)
        out += stream.flush()
        assert set(out) <= set(" .,\n*")

    def test_a_key_block_open_at_a_gap_stays_open(self) -> None:
        stream = Redactor().stream()
        out = stream.feed("start -----BEGIN PRIVATE KEY-----\nMIIEvQIBADAN") + stream.gap()
        out += stream.feed(
            "x" * 600 + "\nMIIEsecondhalfofthekeybody\n-----END PRIVATE KEY-----\nok"
        )
        out += stream.flush()
        # The held tail goes with the hole, so "start" is masked too; the body after the hole is
        # covered by the block, which only its footer closes.
        assert out == "***\nok"

    def test_text_after_a_gap_is_released_once_past_the_mask(self) -> None:
        redactor = Redactor()
        stream = redactor.stream()
        out = stream.feed("before") + stream.gap() + stream.feed("y" * 2_000) + stream.flush()
        assert out == PLACEHOLDER + "y" * (2_000 - (redactor.max_len - 1))


# ------------------------------ hygiene --------------------------------------


class TestHygiene:
    def test_nothing_it_holds_appears_in_a_repr(self) -> None:
        redactor = Redactor([_HF])
        stream = redactor.stream()
        stream.feed(f"partial {_HF[:20]}")
        assert _HF not in repr(redactor)
        assert _HF[:20] not in repr(stream)

    def test_it_never_logs(
        self, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        caplog.set_level(logging.DEBUG)
        redactor = Redactor([_HF])
        redactor.redact(f"a {_HF} b")
        stream = redactor.stream()
        stream.feed(_HF)
        stream.flush()
        assert caplog.records == []
        captured = capsys.readouterr()
        assert captured.out == captured.err == ""

    @pytest.mark.parametrize("serialise", [pickle.dumps, copy.copy, copy.deepcopy])
    def test_it_cannot_be_pickled_or_copied(self, serialise: Callable[[object], object]) -> None:
        # A pickled stream would carry its raw held-back text; a pickled redactor every form of
        # every value. Neither has a reason to leave the process.
        redactor = Redactor([_HF])
        stream = redactor.stream()
        stream.feed(f"partial {_HF[:20]}")
        for held in (redactor, stream):
            with pytest.raises(TypeError):
                serialise(held)

    def test_is_exported_from_core(self) -> None:
        assert ExportedRedactor is Redactor

    @pytest.mark.parametrize(
        "hostile",
        [
            "-----BEGIN" * 20_000,
            "-----BEGIN PRIVATE KEY-----" * 5_000,
            "hf_" * 50_000,
            "://a:" * 30_000,
            "Bearer " * 20_000,
            "eyJ" + "a" * 200_000,
            "sk-" + "-" * 200_000,
            "sk-" * 70_000,
            "flask-" * 40_000,
            "hf_AAAAAAAA" * 20_000,
            ("\n" * 7 + "hf_AbCdEfGhIjKl") * 10_000,
        ],
    )
    def test_pathological_input_completes(self, hostile: str) -> None:
        # Linear in the input: every quantifier is bounded, and every scan resumes where it left
        # off. A regression to quadratic time shows here as a test that does not finish.
        out = Redactor([_HF]).redact(hostile)
        assert isinstance(out, str)
