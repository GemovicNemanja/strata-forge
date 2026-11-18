# Trusted plugins (`strata_forge.plugins`)

Run code that does not belong in forge (a project's own generator, a custom training loop, a
process reward model) on the user's own compute target, chosen by name from a pinned build an
operator allow-listed. Rationale: [ADR 0021](../architecture/adr/0021-trusted-plugins.md).

## The manifest

A plugin repository declares itself in `strata-plugin.json` at its root:

```json
{
  "schema": "strata.plugin/v1",
  "distribution": "acme-prm",
  "version": "0.3.0",
  "entry_points": [
    {
      "name": "prm",
      "type": "step_scorer",
      "inputs": [
        {"role": "steps", "required_columns": ["item_id", "step_index", "context", "step"]},
        {"role": "weights", "kind": "model"}
      ],
      "outputs": [{"role": "scores", "kind": "dataset"}],
      "config": [{"name": "batch_tokens", "type": "integer", "minimum": 1, "default": 4096}],
      "gpu": true
    }
  ]
}
```

The distribution registers each entry point under the `strata_forge.plugins` group
(`[project.entry-points."strata_forge.plugins"] prm = "acme_prm:create"`). `parse_manifest`
reads the file as data (64 KiB cap, JSON and validation only) and refuses anything undeclared:
entry-point names are a safe subset that can never read as `module:attr`, parameters are typed
scalars (`string` with an optional `enum` and length, `integer` / `number` with optional bounds,
`boolean`), and a `step_scorer` must read the `steps` input with the step columns and write the
`scores` dataset.

## The allow-list and the pin

`AllowedPlugin` is one operator entry: an id, a label, a `PluginPin` and optionally the entry
points that may run. A `PluginPin` is the distribution, the exact version and a `PluginSource`:

- `git`: a GitHub repository at a full commit; installed with
  `<dist> @ git+https://github.com/<repo>.git@<commit>`;
- `hf`: a Hugging Face model repository holding a built wheel at a full commit; installed from
  the wheel's `resolve` URL at that commit.

`parse_allow_list` reads the operator's JSON list and refuses a malformed one outright.
`installed_mismatch(pin)` runs on the target: the installed version must equal the pin, and PEP
610's `direct_url.json` must record the pinned source at the pinned commit (an index install or
an editable checkout never matches). `registered_entry_point(pin, name)` finds the entry point in
that distribution's own metadata only, so another distribution registering the same name is never
picked up.

## Preflight (control plane)

`preflight(manifest, allowed, request)` checks a `PluginRequest` (entry point, input bindings at
full commits, output repository, configuration) against the manifest and the operator's entry and
returns named checks (`manifest`, `entry_point`, `allowed`, `inputs`, `columns`, `output`,
`config`), each `ok`, `fail` or `unknown`. Columns nobody has read yet are `unknown`, never
guessed. The configuration comes back with defaults filled when nothing failed. Nothing in it does
I/O or imports plugin code; the runner repeats every check on the target.

## The runner (compute target)

`python -m strata_forge.pipelines.plugin_runner` reads a `PluginRunSpec` from
`STRATA_RUN_CONFIG` (the pin, the entry point as the manifest declared it, the configuration,
the input bindings and the output repository) and the allowed pins from `STRATA_ALLOWED_PLUGINS`,
which the orchestrator sets from its own configuration. In order:

1. engine version handshake (`_common.load_config`);
2. the pin must be allowed, and the bindings and configuration must fit the declared interface;
3. the installed build must be exactly the pin, and register the entry point;
4. the runner downloads each input at its commit with the run's credential: dataset splits to
   local parquet (required columns checked), model inputs as snapshots without pickle-format
   weights (`*.bin`, `*.pt`, `*.pth`, `*.pkl`, `*.ckpt`);
5. the plugin runs in `python -m strata_forge.plugins.host <job.json>`, a child process in its own
   session with `plugin_environ`: `PATH`, `HOME`, locale and time zone (and `CUDA_*` /
   `NVIDIA_*` for a GPU entry point), every credential-shaped name dropped, `PYTHONNOUSERSITE`,
   and the Hub offline on an empty `HF_HOME`; a cancellation sends SIGTERM to its process group
   and SIGKILL after 10 seconds;
6. the output is checked, `manifest.json` (`strata.run.manifest/v1`, `kind: "plugin"`) is written
   beside it, and the directory is pushed to the output repository, which must be private (an
   existing public repository is refused).

A `runner` entry point is called with a `PluginContext`: its configuration, its dataset inputs as
parquet paths, its model inputs as directories, the one output directory, and `report()` /
`phase()` for metrics and captions on the run's progress channel.

## Step scoring (PRM)

A `step_scorer` entry point is a factory called with a `StepScorerSetup` (configuration and model
input paths) that returns an object with `score(steps) -> list[float]`. The host reads the `steps`
input in batches of `batch_size`, validates each row into a `Step` (`item_id` string,
`step_index` non-negative integer, `context` and `step` strings), calls `score`, refuses anything
but one finite number per step (`check_scores`), and writes `scores.parquet` (`item_id`,
`step_index`, `score`). `aggregate(rows, "min" | "mean" | "last" | "product")` reduces step scores
to one per sample; the runner does not pick a reduction.
