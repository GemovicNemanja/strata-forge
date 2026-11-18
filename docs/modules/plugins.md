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
      "gpu": true,
      "score_kind": "probability",
      "unscored_reasons": ["unparseable", "over_token_cap", "not_scored_type"]
    }
  ]
}
```

The distribution registers each entry point under the `strata_forge.plugins` group
(`[project.entry-points."strata_forge.plugins"] prm = "acme_prm:create"`). `parse_manifest`
reads the file as data (64 KiB cap, JSON and validation only) and refuses anything undeclared:
entry-point names are a safe subset that can never read as `module:attr`, parameters are typed
scalars (`string` with an optional `enum` and length, `integer` / `number` with optional bounds,
`boolean`), and a `step_scorer` must read the `steps` input with the step columns, write the
`scores` dataset and declare its `score_kind` (below). A step scorer may also declare the
`unscored_reasons` it gives and `passthrough_columns`: further required `steps` columns copied
unchanged into the scores (never `item_id`, `step_index`, `context`, `step`, `step_type`, `score`
or `reason`).

## The allow-list and the pin

`AllowedPlugin` is one operator entry: an id, a label, a `PluginPin`, optionally the entry points
that may run, `allow_unisolated` (below) and `limits` (`PluginLimits`: `max_runtime_seconds`,
default one day and the CPU-time limit too; `max_memory_mb`, off by default because CUDA maps far
more address space than it uses; `max_file_mb`; `max_open_files`). A `PluginPin` is the
distribution, the exact version and a `PluginSource`:

- `git` (the default): a GitHub repository at a full commit, installed anonymously with
  `<dist> @ git+https://github.com/<repo>.git@<commit>`, and `files_sha256`, the digest of the
  files that build installs. A commit pins the source, not what a build of it produces, so the
  digest is required: the operator installs the build once into a scratch environment, inspects
  it, and runs `python -m strata_forge.plugins.pins digest <distribution>` there.
- `hf`: a Hugging Face model repository holding a built wheel at a full commit, with the wheel's
  `wheel_sha256` (required) and optionally `files_sha256`.

`files_digest(dist)` is the SHA-256 over the RECORD's `path,hash` lines inside the site
directory, sorted, leaving out bytecode, anything outside the site directory (console scripts
carry the environment's path) and the dist-info files an installer writes (`RECORD`,
`INSTALLER`, `REQUESTED`, `direct_url.json`, `WHEEL`).

`parse_allow_list` reads the operator's JSON list and refuses a malformed one outright.
`installed_mismatch(pin, path=...)` checks one environment's site directories: the installed
version must equal the pin, and PEP 610's `direct_url.json` must record the git repository at the
pinned commit, or the pinned wheel hash (an index install or an editable checkout never matches).
`record_mismatch(pin, path=...)` re-hashes every file the RECORD lists and compares the digest
with the pin. `registered_entry_point(pin, name, path=...)` finds the entry point in that
distribution's own metadata only, so another distribution registering the same name is never
picked up.

`RunnerAllowance` is what an orchestrator hands the runner for a job in `STRATA_ALLOWED_PLUGINS`,
from its own configuration: the operator's `AllowedPlugin` and the entry points the manifest at
the pinned commit declares, as the orchestrator read them. The runner takes the entry point's
interface, the operator's entry-point limit, the isolation policy and the limits from it, never
from the spec.

## The plugin environment (compute target)

The runner builds each allowed build its own virtual environment from the pin
(`ensure_environment`), under `~/.strata-forge/plugin-envs/<key>` (`STRATA_PLUGIN_ENVS_DIR`
overrides it). It never installs into the engine's venv, so a rebuild of that venv leaves plugin
environments in place, the engine's interpreter never has plugin code on its path, and the
operator never installs anything by hand.

- The key hashes the full pin and the engine's own install requirement (`engine_requirement`:
  its PEP 610 git commit or archive, else the exact release), so a new plugin commit or a new
  engine build gets a new environment.
- The environment holds the plugin and the same engine build (`strata-forge[hf]`), resolved in
  one `pip install`; the host runs from it.
- The install runs with `install_environ`: the locale and `PATH` only, every credential-shaped
  name dropped, a fresh empty `HOME` (no `~/.netrc`, no git or pip configuration of the
  machine's), `PIP_CONFIG_FILE` and `GIT_CONFIG_GLOBAL` set to the null device, no prompts. It
  has network access (pip fetches the dependencies). A plugin's build backend runs here, on the
  owner's machine, as the runner's uid, with no credential in its environment; it is not
  sandboxed, so the operator's review of the build is what vouches for it.
- An `hf` wheel is downloaded by the runner at the commit (with the run's credential, so a
  private repository works) and refused unless its SHA-256 is the pinned one, BEFORE anything
  installs it; pip then installs the verified local file with the hash in its URL and records it.
- After the install and on every reuse, the environment must hold exactly the pinned build
  (PEP 610, every RECORD hash, the pinned files digest) and the runner's engine version; one that
  does not, or whose build never finished (no marker), is removed and rebuilt. Builds of one
  environment are serialised with a lock file.
- The resolved package set (`pip list`, name and version) is recorded with the environment and in
  the run's manifest (`environment`).

## Preflight (control plane)

`preflight(manifest, allowed, request)` checks a `PluginRequest` (entry point, input bindings at
full commits, output repository, configuration) against the manifest and the operator's entry and
returns named checks (`manifest`, `entry_point`, `allowed`, `inputs`, `columns`, `output`,
`config`), each `ok`, `fail` or `unknown`. Columns nobody has read yet are `unknown`, never
guessed. The configuration comes back with defaults filled when nothing failed. Nothing in it does
I/O or imports plugin code; the runner repeats every check on the target.

## The sandbox (compute target)

The runner holds the Hub token in memory for the whole run. `choose_sandbox` picks the strongest
way to keep the plugin's process from it, probing each by running it:

| Isolation | When | What the plugin's process can reach |
|---|---|---|
| `uid` | `bwrap`, `setpriv` and passwordless `sudo` all work | the `bwrap` sandbox below, running as `nobody` (65534) with no supplementary groups and no new privileges |
| `bwrap` | `bwrap` works for the runner's uid | new PID, network, IPC, UTS and cgroup namespaces: no runner process is visible, there is no network at all; read-only `/usr`, `/bin`, `/sbin`, `/lib*`, `/etc`, `/opt`, the plugin environment and its base interpreter, the job file and the inputs; one writable output directory; a private `/tmp`, which is also `HOME`; no home directory of the runner's; `/sys` read-only and the GPU device nodes for a GPU entry point; dies with the runner |
| `none` | only when the operator set `allow_unisolated`, Yama's `ptrace_scope` is 1 or more, and the runner's environment holds no credential-shaped name | everything the runner's uid can: its files (an SSH key, a cached login), the network; it cannot ptrace the runner or read its starting environment |

With no usable sandbox and no `allow_unisolated`, the run is refused. The isolation used and the
`ptrace_scope` are recorded in the manifest (`sandbox`) and shown with the run.

## The runner (compute target)

`python -I -m strata_forge.pipelines.plugin_runner` reads a `PluginRunSpec` from
`STRATA_RUN_CONFIG` (the pin and the entry point's NAME, the configuration, the input bindings,
the output repository, the opt-in diagnostics), the token from the backend's secrets file (read
into memory and unlinked first), and the `RunnerAllowance` list from `STRATA_ALLOWED_PLUGINS`.
In order:

1. engine version handshake (`_common.load_config`); a token delivered in `HF_WRITE_TOKEN` is
   refused, since a process's starting environment stays readable in `/proc` while it runs;
2. the pin must be allowed and the entry point allowed by the operator; its interface is the
   allowance's, and the bindings and configuration must fit it;
3. a sandbox is chosen (above);
4. the plugin environment is built or reused and verified (above), and the pinned distribution
   in it must register the entry point;
5. the runner downloads each input at its commit with the run's credential: dataset splits to
   local parquet (required and pass-through columns checked), model inputs as snapshots without
   files that can run code when loaded (`*.bin`, `*.pt`, `*.pth`, `*.pkl`, `*.pickle`, `*.ckpt`,
   `*.joblib`, `*.dill`, `*.npy`, `*.npz`, `*.keras`, `*.h5`, `*.hdf5`);
6. the plugin runs as `<env>/bin/python -I -m strata_forge.plugins.host <job.json>` in the
   sandbox, in a session of its own, with `plugin_environ`: `PATH`, locale and time zone (and
   `CUDA_*` / `NVIDIA_*` for a GPU entry point), a private `HOME`, every credential-shaped name
   dropped, and the Hub offline on an empty `HF_HOME`. Before importing any plugin module the
   host lowers its own limits (CPU time, file size, open files, address space when set, no core
   files), checks the build again (PEP 610 and RECORD), and imports each module on the entry
   point's path only after finding its file among the pinned distribution's files. Progress
   comes back as JSON lines on the host's original standard output, a pipe the runner reads in
   bounded chunks and validates (`step` and `phase` only, metrics keyed `plugin.<name>`,
   captions prefixed `plugin:`, a bounded rate); the plugin's own prints go to standard error,
   of which the last 8 KiB are kept. The runner samples the GPU from its own process meanwhile.
   The wall-clock limit stops it. However the process ends, its whole process group is sent
   SIGTERM, then SIGKILL, and the runner waits until neither the group nor any process naming
   the job file is left; a survivor fails the run before anything is pushed;
7. the output is copied by `stage_output`: the output directory is `lstat`ed (a link is
   refused), walked with each directory and file opened relative to its parent with
   `O_NOFOLLOW`, any link, special file or hard-linked file refuses the whole output, every file
   is copied into a fresh directory the runner owns and its copy's SHA-256 checked. From then on
   only the copy is read: a step scorer's scores are checked against the input
   (`check_scores_file`), the diagnostics written, `manifest.json` (`strata.run.manifest/v1`,
   `kind: "plugin"`) written, and the copy pushed to the output repository, which must be
   private (an existing public repository is refused). Plugin output lives only there.

The manifest records the plugin (`id`, `repo`, `commit`, the full pin, the entry point, its type
and `score_kind`), the environment (key, requirements, resolved packages), the sandbox
(`isolation`, `ptrace_scope`, the limits), the configuration, each input with its `commit` and
row count, the output (every file's SHA-256 and, for a step scorer, the `scoring` counts) and the
timing.

A `runner` entry point is called with a `PluginContext`: its configuration, its dataset inputs as
parquet paths, its model inputs as directories, the one output directory, and `report()` /
`phase()` for metrics and captions on the run's progress channel. Every metric `report()` sends
is keyed `plugin.<name>`, so it reads as plugin-reported beside the GPU numbers forge measured.

## Step scoring (PRM)

The platform scores and records; what the scores feed stays in the caller's code. Steps are
segmented upstream, and forge never re-splits or re-joins them.

A `step_scorer` entry point is a factory called with a `StepScorerSetup` (configuration and model
input paths) that returns an object with `score(steps)`, returning one result per step in order:
a number, or `Unscored(reason)` with a reason from the manifest's `unscored_reasons`. The host
reads the `steps` input in batches of `batch_size` and validates each row into a `Step`:
`item_id` string, `step_index` non-negative integer, `step` string, `step_type` string when the
input has that column, and `context` either a string or a list of `{pid, text}` segments, which
reaches the scorer as `ContextSegment`s in input order, never joined.

`score_kind` decides what a number may be:

| `score_kind` | Accepted | Recorded |
|---|---|---|
| `probability` | finite, in [0, 1] (both ends allowed); anything outside fails the run, never clipped | as returned |
| `logit` | finite | as returned |
| `unconstrained` | finite | as returned |

NaN and infinity fail the run whatever the kind. `scores.parquet` has exactly one row per input
step, in input order: `item_id`, `step_index`, `step_type` (when the input has it), the declared
pass-through columns (copied from the input file, type and value unchanged), `score` (float64,
null when unscored) and `reason` (null when scored). Before anything is pushed, the runner checks
its own copy of that file against the input, trusting nothing the plugin's process checked (it
shared that process): the schema (`score` float64, `reason` string, each kept column of the
input's type), one row per input step with the exact `(item_id, step_index)` sequence and every
kept column equal to the input's (a missing or extra row is an infrastructure fault that fails
the run), every score finite and inside its `score_kind`, and a null score exactly where a
declared reason is given. The manifest's `output.scoring` records
`score_kind`, `rows`, `scored` and `unscored` counts by declared reason.

There is no default aggregation. A spec's `diagnostic_aggregations` (any of `min`, `mean`,
`last`, `product`; empty unless asked for) writes `aggregates.diagnostic.parquet` (`item_id`,
`aggregation`, `value`, `scored_steps`, `unscored_steps`; schema metadata
`strata.diagnostic: true`) beside the scores and never into them. Each reduces an item's scored
steps by `step_index`; unscored steps are counted, never imputed, and an item with none scored
has a null value. The manifest lists it under `diagnostics`, marked diagnostic.
