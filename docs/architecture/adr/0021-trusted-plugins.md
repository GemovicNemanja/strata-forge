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
- **Exact installed build.** The plugin is installed on the target by its owner, with an exact
  requirement. The runner accepts it only when the installed version is the pinned one and PEP
  610's `direct_url.json` records the pinned source at the pinned commit; an index install or an
  editable checkout never matches. The entry point is looked up by name inside that
  distribution's own metadata only.
- **Credential separation.** The runner downloads the declared inputs at their commits and pushes
  the output to a private repository itself. The plugin runs in a child process with an
  allow-listed environment that holds no credential and an offline, empty Hub cache. Model
  weights in pickle formats are not downloaded.
- **Step scoring is a forge contract.** A `step_scorer` reads one `steps` input with the columns
  `item_id`, `step_index`, `context` and `step`; forge calls it in batches and writes exactly one
  finite score per step.

## Consequences

Running a plugin needs three parties to agree: the operator allow-lists the build, the owner
installs it on their own machine, and the request names it. The control plane never runs plugin
code, and a plugin never sees the run's credential. Plugin code is still the owner's chosen code
on the owner's machine: this is credential separation, not isolation from a hostile plugin.
Installing the build is outside forge (no deploy key reaches the machine).
