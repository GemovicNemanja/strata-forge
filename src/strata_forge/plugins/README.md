# strata_forge.plugins

Trusted plugins: code from a pinned, operator-allow-listed build that runs on the user's own
compute target, chosen by name, never by import path.

- `manifest.py`: the `strata-plugin.json` declaration (`strata.plugin/v1`), parsed as data.
- `pins.py`: the operator allow-list, exact build pins (a git source at one commit, or a Hugging
  Face wheel at one commit with its SHA-256; the installed files' digest; hash-locked
  dependencies and build requirements) and the check that a directory holds exactly that build.
- `environment.py`: the per-build directory the runner installs the pin into, running no plugin
  code outside the sandbox (hash-pinned wheels only; a git build in the sandbox), read from its
  files and verified on every reuse.
- `preflight.py`: a launch request checked against the declared interface, with no code run.
- `step_scoring.py`: the process-reward-model contract for `step_scorer` entry points.
- `sandbox.py`: the isolation plugin code runs under (another uid, or bubblewrap; never none)
  and the command that starts it.
- `child.py`: the plugin's process: progress relayed through a validated pipe, limits, and the
  whole group confirmed gone before anything else happens.
- `output.py`: the plugin's output copied without following links, and the runner's own check
  of a step scorer's scores.
- `host.py`: the child process the plugin's code runs in, without the run's credential.

The VM runner is `strata_forge.pipelines.plugin_runner`. Reference:
[`docs/modules/plugins.md`](../../../docs/modules/plugins.md); rationale:
[ADR 0021](../../../docs/architecture/adr/0021-trusted-plugins.md).
