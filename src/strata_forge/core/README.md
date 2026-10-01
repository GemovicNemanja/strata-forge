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

- **What it removes.** Each given value as written, line by line for a multi-line value, in all four Unicode normal forms, base64-encoded at any alignment inside a larger blob (standard and URL-safe alphabets), percent-encoded (including every character escaped, either hex case), and escaped as it appears inside a JSON string or a Python `repr`. Plus `DEFAULT_PATTERNS`, the credential shapes it removes whoever they belong to: Hugging Face (`hf_`), OpenAI and Anthropic (`sk-`, `sk-ant-`), GitHub (`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`, `github_pat_`), AWS access key ids (`AKIA`/`ASIA`), JWTs, the userinfo of a URL (`scheme://user:password@`), bearer credentials, and PEM private-key blocks, which are redacted from header to footer or, with no footer, for 16,384 characters past the header.
- **What it refuses.** A value shorter than 8 characters raises `ValidationError`: it would match inside ordinary words. The message never repeats the value.
- **What the output looks like.** Every maximal redacted run becomes one fixed `***`, so the output does not encode a secret's length. A bearer credential keeps its scheme word and a URL keeps its scheme and host.
- **Streams.** `feed` releases what is final and holds back the last `max_len - 1` characters; `flush` releases the rest. The concatenated output is identical to `redact` of the concatenated input, whatever the piece boundaries. A reader that loses text between pieces calls `gap()` there, which masks the held tail and the first `max_len - 1` characters after the hole. The redactor never logs, and neither its `repr` nor a stream's shows what it holds.
- **Limits.** Text is never Unicode-normalised (the output is the input with spans replaced). A token longer than its pattern's bound is redacted for its first `max_chars` characters, which makes it unusable. A value longer than 64 characters is matched as overlapping 64-character fragments, so a clipped print of it is redacted for all but up to 63 characters at each edge.
