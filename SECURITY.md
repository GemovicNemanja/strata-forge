# Security policy

## Supported versions

`strata-forge` is pre-1.0. Only the most recent release on PyPI receives security fixes; there are
no backports to earlier versions and no long-term support branches. A fix ships as a new patch
release, and the advisory names the first version that contains it.

| Version | Supported |
|---|---|
| Latest release on [PyPI](https://pypi.org/project/strata-forge/) | Yes |
| Any earlier version | No — upgrade to the latest release |

## Reporting a vulnerability

**Please do not open a public issue, discussion, or pull request for a security vulnerability.**
Public reports expose users before a fix exists.

Report privately through GitHub's private vulnerability reporting:

- [Open a draft security advisory](https://github.com/GemovicNemanja/strata-forge/security/advisories/new)

If you cannot use GitHub advisories, or you get no response through them, email
**gemovic@strataml.com** instead. Encrypted mail is welcome; ask for a key in a first message that
contains no details.

A useful report includes:

- The affected version, Python version, and which optional extras are installed.
- A minimal reproduction — the smaller and more deterministic, the faster the fix.
- The impact you believe it has, and the assumptions behind that (who has to be able to do what).
- Any workaround you have found, so users can be told something in the advisory.

**Never include live credentials in a report.** If a report would only be reproducible with a real
provider key, describe the request shape instead, and rotate any key you believe was exposed.

## What to expect

- **Acknowledgement within 3 business days** that the report was received and is being looked at.
- **An initial assessment within 10 business days** — whether it is accepted as a vulnerability, its
  severity, and a rough remediation plan.
- Progress updates at least every 14 days while the issue is open.
- Coordinated disclosure: the advisory is published once a fixed release is available. This is a
  single-maintainer project, so please allow up to 90 days before disclosing publicly, and say so
  in the report if you intend to disclose sooner.
- Credit in the advisory by the name or handle you choose, unless you prefer to stay anonymous.
  There is no bug-bounty program.

## Security surface of this library

`strata-forge` is a library, not a hosted service. It has no server, no user accounts, and no data
store of its own — but it does handle provider credentials, write call transcripts, and run commands
on machines you provision. The areas below are the ones worth understanding before you deploy it,
and the ones where a report is most likely to be actionable.

### Provider credentials

Credentials are read from the process environment and from a `.env` file loaded by
`strata_forge.config`; the library never prompts for a key, ships one, or transmits one anywhere
except to the provider it is calling. Inside the LLM module every key is held as a Pydantic
`SecretStr`, so it renders as `**********` in reprs, tracebacks, and model dumps, and is unwrapped
via `.get_secret_value()` only at the LiteLLM call boundary. Structured log records carry no raw
credential material; prompts are logged at `DEBUG`, never at `INFO`.

If you find a code path that puts a raw key, an AWS SigV4 signature, or a GCP bearer token into a
log record, an exception message, a trace payload, or a file on disk, that is a vulnerability —
report it.

### The diagnostic dump

The NDJSON diagnostic dump is **off by default**. With `FORGE_DIAGNOSTIC_ENABLED=1` it appends one
record per LLM attempt to `FORGE_DIAGNOSTIC_PATH` (default `./forge-diagnostic.ndjson`), and those
records contain the full conversation history and the model's response text in plaintext. Treat the
file as being as sensitive as the data you send to the model: it is unencrypted, unrotated, and
readable by anything running as your user. The repo `.gitignore` covers `forge-diagnostic*.ndjson`,
but a dump written anywhere else is yours to protect and delete.

### Recorded HTTP cassettes

Provider-touching tests replay VCR cassettes under `tests/vcr/cassettes/`. Recording (`RECORD=1`)
requires live keys, and `tests/vcr/conftest.py` scrubs auth-bearing headers and query parameters via
`before_record_request` before anything is written to disk. Two rules follow:

- **Inspect a freshly recorded cassette before committing it.** The scrub list covers the headers we
  know about; a provider that starts returning a new token-bearing field will not be caught for you.
- **If you record without scrubbing, treat every key involved as leaked and rotate it immediately.**
  A committed cassette is public and permanent, exactly like a published sdist.

Cassettes are excluded from the published distributions, and the release workflow fails the upload
if any appear in the archives.

### What ships in the published distribution

The sdist build config is an allow-list, not a filter: `only-include = ["src/strata_forge"]`, with
explicit exclusions for `**/.env*` and the agent-instruction files. Hatchling adds only the readme,
licence, `NOTICE`, and `pyproject.toml`. Before every upload, `.github/workflows/release.yml`
re-reads the built sdist and wheel and fails the job if `.env`, coverage data, recorded cassettes,
or agent-instruction files appear inside them. If you add a build `include` or `force-include`,
re-check what lands in the archive — an sdist is public and permanent, and cannot be cleanly
unpublished.

### Remote compute backends execute code

`strata_forge.compute` submits a `Task` whose `run` field is a shell command, executed verbatim:

- `LocalBackend` runs it as a subprocess on the machine running your Python process.
- `SSHBackend` runs it over `asyncssh` on a host whose address and credentials you supply.
- `SkyPilotBackend` runs it on a cloud VM SkyPilot provisions in your own cloud account.

There is no sandbox and none is claimed. Anything that composes a `Task` from model output, user
input, or a config file you did not write is executing that input as shell on your infrastructure —
validate it yourself before submitting. The same applies to `strata_forge.pipelines`, whose runner
reads its entire run specification from an environment variable.

`Backend.read_file` is the one place a caller-supplied path reaches a backend, and it is guarded:
`safe_workdir_relpath` rejects absolute paths, `..` traversal, backslashes, and NUL bytes, confining
reads to the job's own working directory. That guard is lexical and does not resolve symlinks;
`LocalBackend` additionally resolves the real path and re-confines it. A way around either check is
a vulnerability.

### Agent tools

Built-in tools are opt-in — an `Agent` only gets the tools you pass it.

- `fs_read_tool` is a factory that requires an explicit `allowed_dirs` list and compares against the
  resolved (symlink-followed) target, so an escape attempt is refused rather than followed.
- `fetch_url` issues an HTTP GET with redirects followed and no host allow-list. When the URL comes
  from model output it is server-side request forgery by construction: do not expose it to untrusted
  input from a host that can reach internal services or a cloud metadata endpoint.
- `web_search_tool` ships no search backend; you supply one, and its safety is yours.

### Prompt rendering

Templates render in a Jinja2 `SandboxedEnvironment` with `loader=None` and a curated filter
allow-list, so a template cannot load files from disk or reach attribute-traversal exploits. The
sandbox raises the bar for template text you did not author, but it is not a licence to render
arbitrary attacker-supplied templates — control who writes templates, and treat a confirmed sandbox
escape as a vulnerability in both this project and Jinja2.

## Out of scope

The following are not treated as vulnerabilities in this project, though bug reports are still
welcome for the first two:

- A model producing incorrect, biased, or harmful output. That is a model property, not a library
  defect.
- Prompt injection succeeding against an agent you built. Tool exposure is the caller's decision;
  report it here only if a documented guard in this library (the `fs_read_tool` allow-list, the
  `read_file` workdir confinement) can be bypassed.
- Vulnerabilities in upstream dependencies with no strata-forge-specific exposure — report those
  upstream. If this library uses a dependency in an unsafe way, or pins a version with a known
  advisory, that is in scope.
- Anything requiring an attacker to already control the environment the process runs in, since that
  environment holds the provider credentials outright.
