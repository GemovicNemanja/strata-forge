# strata_forge.plugins

Trusted plugins: code from a pinned, operator-allow-listed build that runs on the user's own
compute target, chosen by name, never by import path.

- `manifest.py`: the `strata-plugin.json` declaration (`strata.plugin/v1`), parsed as data.
- `pins.py`: the operator allow-list, exact build pins (git or Hugging Face source at one commit)
  and the check that a machine holds exactly that build.
- `preflight.py`: a launch request checked against the declared interface, with no code run.
- `step_scoring.py`: the process-reward-model contract for `step_scorer` entry points.
- `host.py`: the child process the plugin's code runs in, without the run's credential.

The VM runner is `strata_forge.pipelines.plugin_runner`. Reference:
[`docs/modules/plugins.md`](../../../docs/modules/plugins.md); rationale:
[ADR 0021](../../../docs/architecture/adr/0021-trusted-plugins.md).
