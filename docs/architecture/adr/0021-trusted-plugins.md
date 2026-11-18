# ADR 0021: Plugins are named, pinned, operator-allow-listed and credential-free

**Status:** Accepted

## Context

Some work a user runs on their own compute target is not general enough to belong in forge: a
project's own data generator, a custom training loop, a process reward model (PRM) that scores
reasoning steps. Letting a spec carry code, a repository to check out or an import path would
make the inert spec executable, and let anyone who can submit a spec choose what runs on a machine
that holds a Hugging Face write token.

## Decision

A plugin is a Python distribution built from one commit of a git or Hugging Face repository. It
declares itself in a `strata-plugin.json` manifest (`strata.plugin/v1`): the entry points it
registers under the `strata_forge.plugins` group, each one's type (`runner` or `step_scorer`),
its read-only inputs (dataset splits and model weights), its one output and its closed
configuration of typed scalars.

- **Operator allow-list.** An orchestrator holds the list of exact builds (distribution, version,
  source, full commit, optionally which entry points) that may run, in its own configuration. A
  request names an allowed build and an entry point; it never carries a repository, commit, path
  or import string. The runner receives the allowed pins from the orchestrator's environment,
  never from the spec, and refuses any other build.
- **Checked as data.** The orchestrator reads the manifest at the pinned commit and checks a
  request against it (`preflight`) without building, importing or executing anything from the
  repository.
- **Exact artifact, installed by the runner, with no plugin code outside the sandbox.** The
  runner installs the pinned build into a dedicated per-build directory beside (never inside)
  the engine's venv, keyed by the pin, the engine build and the interpreter, with no credential
  in the install's environment. Every pin carries the digest of the files its build installs (a
  commit pins source, not what a build produces; a wheel's hash pins the file only until it is
  unpacked), and every dependency beyond the engine's environment is locked with its version and
  wheel hashes. Dependencies and an `hf` wheel (checked against its SHA-256 first) are only
  unpacked (`--require-hashes --only-binary=:all: --no-deps`), which runs nothing; a git source is
  fetched anonymously and built INSIDE the sandbox with only its locked build requirements, no
  network and the engine read-only. The directory is read from its files, never run, and never a
  site directory, so no `.pth` file in it is processed; one with an executable line is refused.
  The runner accepts it only when the pinned version is there, every file matches its RECORD
  hash, the files digest matches the pin, and every requirement is present, after the build and
  on every reuse. The resolved package set is recorded. The entry point is looked up by name
  inside that distribution's own metadata only.
- **Credential separation, enforced by a sandbox.** The runner reads the token from the secrets
  file into memory, unlinks the file, and refuses a token delivered in an environment variable
  (a process's starting environment stays readable in `/proc`). It downloads the declared inputs
  at their commits and pushes the output to a private repository itself. The plugin runs on the
  engine's interpreter with `-I`, its own directory appended to the import path, in the strongest
  sandbox the machine has: bubblewrap as another uid (passwordless sudo), else bubblewrap as the
  runner's uid (new PID and network namespaces, the engine, the plugin's directory and the inputs
  bound read-only, one output directory writable, no home directory). There is no unsandboxed
  mode: a process of the runner's uid outside a sandbox can rewrite the engine the runner imports
  and read its starting environment whatever the ptrace scope, so a machine with no sandbox runs
  no plugin. The isolation used is recorded. Its environment holds no credential and
  an offline, empty Hub cache, its time and resources are limited, and its progress reaches the
  runner only through a pipe the runner validates. Model files in formats that run code when
  loaded are not downloaded.
- **Nothing the plugin left is trusted.** However the plugin ends, its whole process group is
  killed and confirmed gone before its output is touched. The output is copied without following
  links, refusing links, hard links and special files, into a directory the runner owns; only
  the copy is checked and pushed. A step scorer's scores are re-checked by the runner from that
  copy. The build is checked again inside the plugin's process before any of its modules is
  imported, and each module on the entry point's path must be one of the pinned distribution's
  files.
- **Step scoring is a forge contract; the platform scores and records.** A `step_scorer` reads
  one `steps` input with the columns `item_id`, `step_index`, `context` and `step` (and
  `step_type` when present), segmented upstream and never re-split; a structured context reaches
  it as `{pid, text}` segments, never joined. Its manifest declares a `score_kind`: a
  `probability` outside [0, 1] fails the run and is never clipped, a `logit` or `unconstrained`
  score passes through; NaN and infinity always fail. A step it does not score carries a
  declared reason. Forge writes exactly one row per input step, re-checks it against the input
  before the push (a missing row is an infrastructure fault), and records the counts by reason.
  Forge picks no aggregation: reductions are opt-in, diagnostic, and written to a separate file.

## Consequences

Running a plugin needs two parties to agree: the operator allow-lists the build, and the request
names it; the runner installs exactly that build on the owner's machine. The control plane never
runs plugin code, and a plugin never holds the run's credential, at install or at run time. The
sandbox is what keeps a misbehaving plugin from the credential and the runner, at install as at
run time: plugin code, its build backend included, runs nowhere else, so a machine without
bubblewrap cannot run plugins. An operator locks a plugin's dependencies with hashes (a lock file
compiled against the engine's environment) and a git source's build backend; a dependency
published only as an sdist cannot be used, since building it would run its code outside the
sandbox. No deploy key reaches the machine, so a git source must be publicly readable; a private
build is published as an `hf` wheel, pinned by its hash.
