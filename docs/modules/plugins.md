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
that may run, and `limits` (`PluginLimits`: `max_runtime_seconds`, default one day and the
CPU-time limit too; `max_memory_mb`, off by default because CUDA maps far more address space than
it uses; `max_file_mb`; `max_open_files`). There is no unsandboxed mode to allow (below). A
`PluginPin` is the distribution, the exact version, a `PluginSource` and two hash-pinned locks:

- `source`, either `git` (the default): a GitHub repository at a full commit; or `hf`: a Hugging
  Face model repository holding a built wheel at a full commit, with the wheel's `wheel_sha256`.
  Both carry `files_sha256`, the digest of the files the build installs: a commit pins the source,
  not what a build of it produces, and a wheel's hash pins the file only until it is unpacked. The
  operator installs the build once into a scratch environment, inspects it, and runs
  `python -m strata_forge.plugins.pins digest <distribution>` there.
- `dependencies`: every distribution the plugin needs that the engine's environment does not
  provide, transitively, each a `LockedDependency` (`name`, exact `version`, and the `sha256` of
  every wheel that may be installed for it, one per platform), as a lock file with hashes
  (`uv pip compile --generate-hashes`, against the engine's environment) lists them. Empty when the
  engine's environment is enough.
- `build_requires` (a `git` source only, and required there): the build backend and what it
  needs, locked the same way. The build sees nothing else.

`files_digest(dist)` is the SHA-256 over the RECORD's `path,hash` lines inside the site
directory, sorted, leaving out bytecode, anything outside the site directory (console scripts
carry the environment's path) and the dist-info files an installer writes (`RECORD`,
`INSTALLER`, `REQUESTED`, `direct_url.json`, `WHEEL`).

`parse_allow_list` reads the operator's JSON list and refuses a malformed one outright (an entry
with a field it does not know, such as a former `allow_unisolated`, included).
`installed_mismatch(pin, path=...)` checks that the pinned distribution and version are installed
on those paths; `record_mismatch(pin, path=...)` re-hashes every file the RECORD lists and
compares the digest with the pin, which is what decides that the files are the pinned artifact
(an install record such as PEP 610's `direct_url.json` says where a build came from and proves
nothing about what is on disk). `registered_entry_point(pin, name, path=...)` finds the entry
point in that distribution's own metadata only, so another distribution registering the same
name is never picked up.

`RunnerAllowance` is what an orchestrator hands the runner for a job in `STRATA_ALLOWED_PLUGINS`,
from its own configuration: the operator's `AllowedPlugin` and the entry points the manifest at
the pinned commit declares, as the orchestrator read them. The runner takes the entry point's
interface, the operator's entry-point limit and the limits from it, never from the spec.

## The plugin environment (compute target)

**No plugin code runs outside the sandbox at any step**, the build included. The runner is the
process holding the run's credential; anything running as its uid outside a sandbox could read
its files and rewrite the engine environment it imports from (a `.pth` file there runs in every
later Python start, `-I` or not). So the runner builds each allowed build a directory of its own
(`ensure_environment`), under `~/.strata-forge/plugin-envs/<key>` (`STRATA_PLUGIN_ENVS_DIR`
overrides it), with `pip install --target`, and never runs it:

- The key hashes the full pin (locks included), the engine's own requirement
  (`engine_requirement`: its PEP 610 git commit or archive, else the exact release), the
  interpreter's cache tag and the platform, so any change gets a new directory.
- pip runs from a tools environment of its own (a `venv` of the engine's interpreter holding only
  the pip the standard library bundles, created offline once per machine by an isolated
  `python -I -m venv`), isolated (`-I`), with
  `install_environ`: the locale and `PATH` only, every credential-shaped name dropped, a fresh
  empty `HOME` (no `~/.netrc`, no git or pip configuration of the machine's), no prompts. The
  plugin directory is never on its path.
- The dependencies are installed from the pin's lock with `--require-hashes --only-binary=:all:
  --no-deps`: a file whose hash is not listed is refused, nothing is built and nothing resolved
  further, and installing a wheel unpacks it, it runs nothing.
- An `hf` wheel is downloaded by the runner at the commit (with the run's credential, so a
  private repository works) and refused unless its SHA-256 is the pinned one, then unpacked the
  same way, offline.
- A `git` source is fetched anonymously at the commit (`git fetch --depth 1` of the commit, with
  hooks, submodules, filters and local transports off; git runs none of the repository's code),
  and its wheel is built INSIDE THE SANDBOX by `pip wheel --no-build-isolation --no-index`, with
  only the locked `build_requires` on its path, no network, the engine read-only, and only the
  source tree and an output directory writable; the build is killed and confirmed gone like the
  plugin's process. It runs verbose (`-v`) so the backend's output streams, and is stopped when
  it writes nothing for 10 minutes (`BUILD_IDLE_SECONDS`) as well as at its 30-minute limit
  (`BUILD_TIMEOUT_SECONDS`): a stalled build ends there, not at the wall clock. The wheel it
  produces is unpacked as above.
- What the directory holds is read from its files (`importlib.metadata`), in the runner, and never
  executed: no interpreter is started with it on its path outside the sandbox, and it is never a
  site directory anywhere, so a `.pth` file in it is never processed. A build that ships a `.pth`
  with an executable line is refused all the same (`pth_problems`).
- After the build and on every reuse, the directory must hold exactly the pinned build (the
  pinned name and version, every RECORD hash, the pinned files digest), with every distribution
  the plugin requires present in it or in the engine's environment; one that does not, or whose
  build never finished (no marker), is removed and rebuilt. A reuse also requires the digest of
  the WHOLE directory to be the one recorded in the marker when it was built (`tree_digest`:
  every directory, and every file's path, executable bit and SHA-256, a link or special file
  refusing the directory). The pin's RECORD covers only the plugin's own files; this covers its
  dependencies, their `.data` files and anything planted beside them. Builds of one environment are
  serialised with a lock file. The directory is made readable to every uid (the `uid` sandbox
  reads it as `nobody`) and writable only by the runner.
- The resolved package set (name and version) is recorded in the run's manifest (`environment`).

## Preflight (control plane)

`preflight(manifest, allowed, request)` checks a `PluginRequest` (entry point, input bindings at
full commits, output repository, configuration) against the manifest and the operator's entry and
returns named checks (`manifest`, `entry_point`, `allowed`, `inputs`, `columns`, `output`,
`config`), each `ok`, `fail` or `unknown`. Columns nobody has read yet are `unknown`, never
guessed. The configuration comes back with defaults filled when nothing failed. Nothing in it does
I/O or imports plugin code; the runner repeats every check on the target.

## The sandbox (compute target)

The runner holds the Hub token in memory for the whole run. `choose_sandbox` picks the strongest
way to keep plugin code (the plugin's process, and a git source's build) from it, probing each by
running it:

| Isolation | When | What plugin code can reach |
|---|---|---|
| `uid` | `bwrap`, `setpriv` and passwordless `sudo` all work | the `bwrap` sandbox below, running as `nobody` (65534) with no supplementary groups and no new privileges; it dies with the runner through the supervisor (below), not with `sudo` |
| `bwrap` | `bwrap` works for the runner's uid | new PID, network, IPC, UTS and cgroup namespaces: no runner process is visible, there is no network at all; read-only `/usr`, `/bin`, `/sbin`, `/lib*`, `/etc`, `/opt`, the engine's environment and its base interpreter, the plugin's directory, the job file and the inputs; only its output writable; a private `/tmp`, which is also `HOME`; no home directory of the runner's; `/sys` read-only and the GPU device nodes for a GPU entry point; dies with the runner |

There is no unsandboxed mode. A plain child process of the runner's uid could write the engine
environment the runner imports from, and read `/proc/<runner>/environ` whatever Yama's
`ptrace_scope` (that file is gated by the ptrace access mode's credential check, which a process
of the same uid passes); so a machine with no working sandbox refuses the run. Every directory
bubblewrap creates to hold a bind is made explicitly with `--perms 0755`: as root (the `uid`
sandbox), it would make them `0700`, and `nobody` could reach nothing beneath them. The
isolation used and the `ptrace_scope` are recorded in the manifest (`sandbox`) and shown with
the run.

In a `uid` sandbox bubblewrap's parent is `sudo`, so `--die-with-parent` follows `sudo`, which
outlives a killed runner. The sandbox's first process is therefore
`python -I -m strata_forge.plugins.supervisor` (still root, before `setpriv` drops to `nobody`):
it starts the rest of the command with `/dev/null` as its standard input and watches its own, a
pipe whose write end only the runner holds (`run_plugin_process` opens it; descriptors from
`os.pipe` are not inherited). Nothing is written to it, so it reaches end of file only when the
runner closes it, which the kernel does when the runner dies, however it dies; the supervisor
then kills its child and exits, and as the sandbox's first process its exit ends everything left
in the sandbox's PID namespace. Started without such a pipe, it ends its child at once.

The sandbox itself is tested for real by `tests/unit/plugins/test_sandbox_live.py`, skipped
unless `STRATA_SANDBOX_LIVE=1` (CI runners have no usable bubblewrap): in a privileged Linux
container, once as a user without sudo (`bwrap`) and once as one with passwordless sudo (`uid`),
it runs code in the sandbox the machine offers (it must read the job and inputs, and must not
write the engine, read the runner's home or reach the network) and a whole run of a wheel that
ships a token-stealing `.pth` (refused before anything runs) and of the same build without it,
whose scores are checked by the sandboxed checker; in the `uid` container it also checks that
closing the runner's lifeline ends a running plugin.

## The runner (compute target))

`python -I -m strata_forge.pipelines.plugin_runner` reads a `PluginRunSpec` from
`STRATA_RUN_CONFIG` (the pin and the entry point's NAME, the configuration, the input bindings,
the output repository, the opt-in diagnostics), the token from the backend's secrets file (read
into memory and unlinked first), and the `RunnerAllowance` list from `STRATA_ALLOWED_PLUGINS`.
In order:

1. engine version handshake (`_common.load_config`); a token delivered in `HF_WRITE_TOKEN` is
   refused, since a process's starting environment stays readable in `/proc` while it runs;
2. the pin must be allowed and the entry point allowed by the operator; its interface is the
   allowance's, and the bindings and configuration must fit it;
3. a sandbox is chosen (above); a machine with none runs no plugin;
4. the plugin's directory is built (no plugin code outside the sandbox) or reused, and verified
   (above), and the pinned distribution in it must register the entry point;
5. the runner downloads each input at its commit with the run's credential: dataset splits to
   local parquet (required and pass-through columns checked), model inputs as snapshots without
   files that can run code when loaded (`*.bin`, `*.pt`, `*.pth`, `*.pkl`, `*.pickle`, `*.ckpt`,
   `*.joblib`, `*.dill`, `*.npy`, `*.npz`, `*.keras`, `*.h5`, `*.hdf5`);
6. the plugin runs as `<engine python> -I -m strata_forge.plugins.host <job.json>` in the
   sandbox, the engine bound read-only and the plugin's directory appended to the host's import
   path (after the engine's, never as a site directory), in a session of its own, with
   `plugin_environ`: `PATH`, locale and time zone (and
   `CUDA_*` / `NVIDIA_*` for a GPU entry point), a private `HOME`, every credential-shaped name
   dropped, and the Hub offline on an empty `HF_HOME`. Before importing any plugin module the
   host lowers its own limits (CPU time, file size, open files, address space when set, no core
   files), checks the build again (RECORD and the pinned digest), and imports each module on the
   entry point's path only after finding its file among the pinned distribution's files (a module
   already in `sys.modules` is judged by its own file, never trusted as found). Progress
   comes back as JSON lines on the host's original standard output, a pipe the runner reads in
   bounded chunks and validates (`step` and `phase` only, metrics keyed `plugin.<name>`,
   captions prefixed `plugin:`, a bounded rate); the plugin's own prints go to standard error,
   of which the last 8 KiB are kept. The runner samples the GPU from its own process meanwhile.
   The wall-clock limit stops it, and so does any process of the plugin's staying stopped (a
   ptrace stop or `SIGSTOP`) for 30 seconds: a stopped sandbox init never reaps its children,
   and the run would otherwise hang to its limit. However the process ends, its whole process group is sent
   SIGTERM, then SIGKILL, and the runner waits until neither the group nor any process naming
   the job file is left; a survivor fails the run before anything is pushed;
7. the output is copied by `stage_output`: the output directory is `lstat`ed (a link is
   refused), walked with each directory and file opened relative to its parent with
   `O_NOFOLLOW`, any link, special file or hard-linked file refuses the whole output, every file
   is copied into a fresh directory the runner owns and its copy's SHA-256 checked. From then on
   only the copy is read: a step scorer's scores are checked against the input
   (`check_scores_file`), the diagnostics written, `manifest.json` (`strata.run.manifest/v1`,
   `kind: "plugin"`) written, and the copy pushed to the output repository in ONE commit of
   exactly the staged files. The repository must be private (an existing public repository is
   refused). The run reports `<repo>@<commit>` as its destination, so the control plane reads
   the output back at the commit this run made, never at a head a later push moved. The runner
   also leaves `receipt.json` (`build_receipt`, `kind: "plugin"`: the manifest's digest and the
   repo and commit the push made) in the job's working directory. Plugin
   output lives only there.

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
null when unscored) and `reason` (null when scored). Before anything is pushed, the runner's own
copy of that file is checked against the input, trusting nothing the plugin's process checked (it
shared that process). The runner, which holds the run's credential, never parses it: it runs
`<engine python> -I -m strata_forge.plugins.checker <job.json>` in the same sandbox (no network,
no credential, the engine, the copied scores and the input steps read-only, one directory
writable), which writes a small record and the diagnostics file; the runner reads the record
(at most 64 KiB, validated field by field, its counts required to add up to the input's rows and
its reasons to be declared ones) and copies the diagnostics without following links. The check:
the schema (`score` float64, `reason` string, each kept column of the
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
