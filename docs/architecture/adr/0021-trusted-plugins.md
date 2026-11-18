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
- **Exact artifact, installed by the runner.** The runner installs the pinned build into a
  dedicated per-build environment beside (never inside) the engine's venv, cached by the pin and
  the engine build, with no credential in the install's environment. A git source installs
  anonymously at its commit and is pinned also by the digest of the files its build installs (a
  commit pins source, not what a build produces); a Hugging Face wheel is pinned by its SHA-256,
  checked before anything installs it. The runner accepts the environment only when the
  installed version is the pinned one, PEP 610's `direct_url.json` records the pinned commit or
  wheel hash (an index install or an editable checkout never matches), every file matches its
  RECORD hash, the files digest matches the pin, and the engine there is its own.
  The resolved package set is recorded. The entry point is looked up by name inside that
  distribution's own metadata only.
- **Credential separation, enforced by a sandbox.** The runner reads the token from the secrets
  file into memory, unlinks the file, and refuses a token delivered in an environment variable
  (a process's starting environment stays readable in `/proc`). It downloads the declared inputs
  at their commits and pushes the output to a private repository itself. The plugin runs from its
  own environment's interpreter with `-I`, in the strongest sandbox the machine has: bubblewrap
  as another uid (passwordless sudo), else bubblewrap as the runner's uid (new PID and network
  namespaces, only the environment, inputs and one output directory bound, no home directory),
  else, only when the operator allowed it for that plugin and the kernel restricts ptrace, a
  plain child process. The isolation used is recorded. Its environment holds no credential and
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
sandbox is what keeps a misbehaving plugin from the credential and the runner; without one
(`none`, operator opt-in) the plugin can read whatever the runner's uid can and reach the
network, and only the operator's review of the build vouches for it. The build backend runs
unsandboxed at install time, as the runner's uid with no credential in its environment. No
deploy key reaches the machine, so a git source must be publicly readable; a private build is
published as an `hf` wheel, pinned by its hash.
