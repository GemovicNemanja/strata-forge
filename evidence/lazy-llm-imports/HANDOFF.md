# Lazy LLM package imports

## Source

Checkout: `/Users/nemanjagemovic/Documents/Personal/Code/.strata-workspaces/platform-20261110/agents/lazy-llm-imports`.
Branch: `perf/lazy-llm-imports`. Base: `cffa0ff813bb61c4cd104fbb3384b9f9cf54635b`.
Frozen commit: `55c1ae892434d3d2eb755cfe0ec93b0875ca6dae`.
Tree: `acb00b73050d1c35d110cc53c817d4fd74594fef`.
Author/committer: `Nemanja Gemovic <gemovic.nemanja@gmail.com>`.
The isolated checkout is clean and has no remotes. No source changed after this commit.

Only `llm/__init__.py` changes runtime behavior. Other changed files are focused import
contract tests, the frozen99-export surface fixture, the narrow capability cold-checker
strengthening and import-boundary documentation. Existing provider/error implementations,
registry data, methods, capability producer/schema fixture, package metadata, runners and
workers are unchanged. No public transport or model capability is added.

## Behavior and compatibility

The initializer retains the exact ordered99-name `__all__`. Eleven lightweight registry
exports remain eager, including the actual registry singleton. Eager registry binding avoids
the singleton/submodule package-attribute collision; package-first, submodule-first and
from-import-first paths are independently tested. The local existing registry YAML still loads.

The remaining88 exports use a fixed original-name→original-module map and module
`__getattr__`, caching the original object after its first access. Direct imports under
`TYPE_CHECKING` retain the original static type surface. `dir()` lists exports without loading
them. Every export and star-import binding is checked against its original defining-module
identity, not a proxy or copied value. Unknown attributes still raise `AttributeError`.

Ordinary package, message, training-method and capability imports require no LiteLLM/HTTP/
provider SDK/optional ML imports or offline environment flag. Requesting a transport-dependent
export such as `LLMClient`, or importing the full namespace, loads its real dependencies and
retains their initialization behavior. This is not a network sandbox or a provider rewrite.

## Source-qualified before/after witness

`predecessor-llm-init.py` preserves the actual unchanged base initializer. The same persistent
`import_probe.py` ran before and after the implementation in fresh processes. Each process
received an explicit clean environment and no `LITELLM_MODE` or local-cost-map flag. All socket,
DNS and subprocess audit events were denied and counted. A caught denial remained a failed
probe; no DNS query/network connection was permitted.

- Baseline `deny` mode: exit1, attempted `litellm` import, no transport successfully loaded.
  Retained in `baseline-denied.json` and its stderr file. Sample import interval0.441s.
- Baseline `observe` mode: exit1, LiteLLM caught a denied `socket.getaddrinfo` and completed
  using its fallback. The attempt counter remains nonzero; loaded roots included LiteLLM,
  HTTPX and OpenAI. Retained JSON/stderr contain the original failure. Sample interval1.515s.
- Successor `deny` mode: exit0, no denied import, no loaded transport/optional root and no
  socket/DNS/process attempt. Sample interval0.232s.
- Frozen successor `observe` mode: exit0, the same empty counters/loaded-root sets without
  denying transport imports themselves. Sample interval0.201s.

These are local one-process samples, not a hardware-controlled benchmark or an end-to-end
startup guarantee. The prior32-second cold-import sample came from a different loaded session
and is not used as this comparison's baseline.

## Validation

- `focused-imports.log`:11 passed in3.18s, including four flag-free cold import paths,
  non-loading directory inspection, attribute identity/cache, three registry orders,
  all99 export/star-import identities, static import-surface matching and the strengthened
  capability cold check.
- `final-scoped.log`:407 passed in3.41s, `-W error`. Exact selected files: import contracts,
  messages, responses, registry, routing, training methods and the full capability suite.
  No broad suite or provider call was run.
- `final-pyright.log`:0 errors/0 warnings for the changed initializer and both changed test
  files. Installed Node24.22.1/Pyright1.1.410; fixed retained environment, no installation.
- Affected Ruff lint and format checks returned0; `git diff --check` returned0 before commit.
  Their exact stdout/exit status was observed in the tool transcript, rather than captured
  as a separate original log.

The outer selected LLM tests deliberately used the existing LiteLLM local cost map because
their existing conftest explicitly imports `LLMClient`. Every lightweight cold subprocess
constructs its own flag-free environment, denies actual transport/provider/ML imports and
asserts attempt counters are zero. The separate full-namespace identity probe deliberately
selects the local cost map while denying all socket/DNS/process effects and ML extras.
No provider is invoked by that identity probe.

Python3.14.5, Pydantic2.13.4, pytest9.0.3 and the existing read-only Server virtual environment
were used with only this checkout's `src` ahead of the existing editable installation.
No package/browser/model download, extra installation or environment mutation occurred.

## Retained identities and limits

The capability leaf and exact golden fixture remain byte-identical to the base. Its schema
SHA is `ccb5be4e552c6e3e542f282835566af31b3dfd157f1c361dd517a37ba8b50b8c` and catalog SHA is
`a3d41914c887df59ec0457ff013f104f5f2f47dd58548774204e81a0d25aae90`.
`source-manifest.json` records changed files and retained-byte guards.

This closes only the bounded import prerequisite. FORGE-101, the full FORGE-102 catalog,
CONTRACT-104/106, public consumer/wire review and product-wide readiness remain independent.
Root can merge the original source ancestry locally after review; reverting this commit
restores eager imports without changing data, specs or manifests.

No service listener, browser, database, native worker, GPU, cloud/provider job, production
resource, GitHub/auth operation or research/model data was accessed. All owned scoped process
sessions completed with recorded wait results; the retained user preview and other agents'
worktrees were untouched.
