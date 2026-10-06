"""Runnable pipeline entrypoints executed on a compute target.

These modules are launched on a compute target by an orchestrator rather than imported
in-process. Each reads an inert run spec from the environment, composes the library's
primitives (serving / batch / storage), and appends ProgressEvents to the file named by
``FORGE_PROGRESS_PATH`` so the orchestrator can tail progress over its own channel.

An orchestrator stamps every spec with the engine version it validated the spec against
(``SPEC_VERSION``, optionally suffixed ``+<commit>`` when it installs the engine from a git
ref), and the runner refuses to execute under any other installed engine. That is the
handshake that keeps a machine which already has an older engine installed from running a
spec a newer one validated: the spec's fields may be identical, and only the behaviour behind
them changed.
"""

from strata_forge import __version__

__all__ = ["HF_TOKEN_SECRET", "SPEC_VERSION"]

# The engine version a spec is validated against, and the version an orchestrator writes into
# ``engine_version`` (alone, or as ``f"{SPEC_VERSION}+{commit}"``). It is the package version
# because the spec models live in this package: a change to their fields ships with a version
# bump, so the version names the contract.
SPEC_VERSION = __version__

# The key an orchestrator gives the Hugging Face token under in ``Task.secrets``, and so in the
# job's secrets file: the one key a runner reads. Any other key is refused by name, so this is
# part of the orchestrator contract, like the version above.
HF_TOKEN_SECRET = "HF_TOKEN"  # noqa: S105 — a key name, not a credential
