# strata_forge.core

Cross-cutting utilities every other module depends on: exception hierarchy (`ForgeError` and subclasses), `@retry` decorator, structlog setup with `trace_id` correlation via contextvars, `BudgetContext` for cost/token ceilings, reproducibility helpers (seeds, content hashing, environment snapshots), UUIDv7-based correlation IDs, secret redaction, and shared type aliases. Strictly upstream — never imports from other `strata_forge.*` modules.

## Secret redaction

`strata_forge.core.redact` is the one implementation every surfaced string goes through, so a process that holds credentials and a process that relays its output apply the same rules:

```python
from strata_forge.core import Redactor

redactor = Redactor([hf_token, ssh_private_key])  # each value at least 8 characters
redactor.redact(message)                          # one string

stream = redactor.stream()                        # text that arrives in pieces
for piece in pieces:
    relay(stream.feed(piece))
relay(stream.flush())
```

- **What it removes.** Each given value as written, line by line for a multi-line value, in all four Unicode normal forms, base64-encoded at any alignment inside a larger blob (standard and URL-safe alphabets), percent-encoded (including every character escaped, either hex case), and escaped as it appears inside a JSON string or a Python `repr`. Plus `DEFAULT_PATTERNS`, the credential shapes it removes whoever they belong to: Hugging Face (`hf_`), OpenAI and Anthropic (`sk-`, `sk-ant-`; not after a lower-case letter, as in `flask-`, and not an all-lower-case name), GitHub (`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`, `github_pat_`), AWS access key ids (`AKIA`/`ASIA`), JWTs, the userinfo of a URL (`scheme://user:password@`, with an empty user too, and up to the last `@` of the authority, where a URL parser splits it), and bearer credentials. Credentials glued together are each found. PEM private-key blocks are redacted from header to footer or, with no footer, for 16,384 characters past the header; they are not a pattern, so `patterns=()` turns the shapes off and leaves key blocks on.
- **What it refuses.** A value shorter than 8 characters without its surrounding whitespace raises `ValidationError`: it would match inside ordinary words. So does a value made only of `*`, which is a mask printed in place of a secret. A pydantic `SecretStr` is unwrapped, never stringified; any other non-`str` raises `TypeError`. The message never repeats the value. A caller with an optional secret drops it when empty, and treats the refusal as a reason not to relay, never as a reason to relay raw.
- **What the output looks like.** Every maximal redacted run becomes one fixed `***`, so the output does not encode a secret's length. Runs separated only by up to 8 characters of whitespace count as one, so a multi-line value printed with other line endings, or indented, is one `***` rather than one per line. A bearer credential keeps its scheme word and a URL keeps its scheme and host.
- **Streams.** `feed` releases what is final and holds back the last `max_len - 1` characters, plus up to 8 characters of whitespace right after a redacted run; `flush` releases the rest. The concatenated output is identical to `redact` of the concatenated input, whatever the piece boundaries. A reader that loses text between pieces calls `gap()` there, which masks the held tail and the first `max_len - 1` characters after the hole. The redactor never logs, and neither its `repr` nor a stream's shows what it holds. Neither can be pickled or copied, so a stray serialisation cannot write what they hold anywhere. A relay's own contract (one stream per console stream, kept in memory, `gap()` before the first `feed` of a stream resumed mid-transcript, `flush` only at the end) is in `docs/modules/compute.md`, "Redacting the console".
- **Limits.** Text is never Unicode-normalised (the output is the input with spans replaced). A token longer than its pattern's bound is redacted for its first `max_chars` characters, which makes it unusable. A value longer than 64 characters is matched as overlapping 64-character fragments, so a clipped print of it is redacted for all but up to 63 characters at each edge. Accepted residuals: base64 of a known value wrapped into lines (MIME, 76 columns) is missed where a line break falls inside it; `Basic <base64>` of credentials the redactor was not given has no shape to match; a token re-wrapped by a pretty-printer is split into pieces no rule recognises; and a non-ASCII value split by a byte-offset read is not matched.
