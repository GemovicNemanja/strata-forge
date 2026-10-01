# ADR 0018 — Secrets travel beside the task, never in it

**Status:** Accepted
**Supersedes:** the environment part of [ADR 0013](0013-compute-task-and-backend-shapes.md)
(a credential a job needs is no longer an entry in `Task.env`)
**Superseded by:** —

## Context

ADR 0013 made a `Task` pure data, and its `env` mapping was the only way to hand a job a value,
credentials included. A pipeline runner on a user's machine needs one: the Hugging Face token it
pushes results with. Carried as `Task.env`, that token leaked in four places on the SSH backend:

1. **The wrapper script.** Every `env` entry became an `export NAME='value'` line in
   `wrapper.sh`, a file that sits in the job's workdir for the life of the run and after it.
2. **The remote command line.** The wrapper was written by a heredoc inside the SSH exec command,
   so the whole script, token included, was the argv of a remote shell. Any user on the host can
   read `/proc/<pid>/cmdline` while it runs.
3. **The setup step's environment.** The wrapper exported the token and then ran `setup && run`,
   so the setup step (a `pip install` of several hundred third-party packages) ran with the token
   in its environment, where any of them, or anything they spawn, could read it.
4. **The model server's environment.** The inference runner started vLLM through `LocalBackend`
   with its default `env_inherit=True`, so the served process inherited the runner's whole
   environment.

Each one is a place a credential persists, or is visible to code that has no use for it.

## Decision

**A credential travels beside the task, in `Task.secrets`, and reaches the job as a private
file it reads and deletes. It is never in an environment, a command line or a script.**

### `Task.secrets`

`secrets: dict[str, SecretStr]` on `Task`, declared with `Field(exclude=True, repr=False)`:

- never in `model_dump`, `model_dump_json`, `to_yaml` or `repr`, and never accepted from YAML
  (a task file is plaintext config that gets committed and copied);
- keys shaped like environment variable names (`^[A-Z][A-Z0-9_]{0,63}$`), at most 8, values
  non-empty, and no key also in `env` (that would put the value in the environment after all);
- no rendering of a validation error carries a value. `hide_input_in_errors=True` covers only
  `str()`; `errors()` and `json()` still carry each error's input, so a `mode="before"` model
  validator wraps every value in `SecretStr` (anything that is not a string in an opaque
  placeholder) before any check runs, and every check on `secrets`, the `env` clash included,
  is a field validator, whose error input is the masked field rather than the caller's raw
  arguments. A pickled `Task` does carry the values, so a task with secrets is never pickled;
- `model_copy(update=...)` and `model_construct` skip every validator, so each backend rebuilds
  the task through them (`Task.revalidated()`) before it delivers anything;
- `env` may not set `FORGE_SECRETS_FILE`, which only a backend may set, because the runner
  deletes the file it names.

### Delivery: one private file per job

Every backend that accepts secrets writes them as one JSON object to a file named
`.secrets.json`, mode 0600, in a directory mode 0700, and exports `FORGE_SECRETS_FILE`, the
file's absolute path, to the task's `run` step. That path is the only secret-related thing the
job's environment carries, and the `setup` step does not get even that.

Both shell-launching backends compose the job's script the same way
(`backends.base.secrets_guarded_script`): an unexported variable holds the path, the OUTER shell
installs `trap 'rm -f "$_forge_secrets_file"' EXIT`, and everything the task supplies runs in a
subshell below it, with `export FORGE_SECRETS_FILE=...` between `setup` and `run`. The subshell is
load-bearing. A second `trap ... EXIT` in one shell replaces the first, and an orchestrator's
bootstrap sets one of its own (to stop a progress ticker); in a single shell that silently
dropped the removal, so a failed setup or a cancel during setup left the file on disk. A trap
set in the subshell cannot reach the outer one, and the outer shell exits with the subshell's
status, which the trap leaves untouched.

- **SSH.** The workdir is created by a single `bash -c` step that sets `umask 077`, requires the
  remote root to be a directory owned by the login user and narrows it to 0700, and creates the
  leaf with `mkdir -m 700` and no `-p`, so an existing directory or symlink fails the submit.
  The wrapper and then the secrets file are written with their contents on the SSH channel's
  **stdin**, so neither appears in any command string. Each write runs `umask 077`, changes
  into the workdir with `cd -P` and checks `test -O .` (the path is resolved once, so a root
  component swapped afterwards cannot redirect the write), refuses anything already at the name,
  and then creates the file under `set -C`. The explicit refusal is what stops a FIFO or a device:
  bash's noclobber opens an existing non-regular file for writing rather than refusing it. The
  secrets file is written LAST, immediately before the launch, so a submit that fails earlier
  strands no secret. The wrapper's trap removes the file when setup fails, when the job ends and
  when `cancel` sends SIGTERM; `cancel` also removes it after its final SIGKILL, for the jobs
  whose trap could not run.
- **SSH, failed submits.** A submit that fails after creating the workdir removes it, within a
  10 s bound because it runs after the caller's own deadline may have fired. When that removal
  fails too (the connection is gone), an ordinary failure is re-raised as `SubmitCleanupError`
  carrying a `Job` handle that `cleanup` accepts, so the orchestrator can remove the workdir
  later; a cancellation stays a cancellation. As a backstop that needs no handle, every submit's
  prepare step removes secrets files older than 10 minutes from job directories with no pid file
  (submits that never launched), before the new job's setup could read one.
- **Local.** A fresh `tempfile.mkdtemp` directory, chmod 0700, holding a file created with
  `O_CREAT | O_EXCL | O_NOFOLLOW` and `fchmod` 0600 (explicit modes, because a umask can only
  narrow and a restrictive one would lock the child out). The child runs the same guarded script;
  the backend removes the directory when the child exits and again in `cleanup`. An inherited
  `FORGE_SECRETS_FILE` from the parent's own environment is dropped, so a child never reads, and
  so deletes, its parent's file.
- **SkyPilot** raises on a non-empty `secrets` before any client call. Its only channel for a
  value is `envs`, which is the job's environment; failing closed beats a silent fallback.

### The runner reads it first, and deletes it

`strata_forge.pipelines._common.load_secrets()` runs first in `runner_main`, before the spec is
read and before any network call or subprocess. It refuses a `FORGE_SECRETS_FILE` that is not an
absolute path ending in `.secrets.json` without touching it, opens the file with `O_NOFOLLOW`
and `O_NONBLOCK` (a FIFO is refused instead of waited on), requires a regular file owned by the
effective user with no group or other bits, reads at most 64 KiB, and unlinks it whatever the
outcome. It accepts only the key `HF_TOKEN` (`strata_forge.pipelines.HF_TOKEN_SECRET`, the name
an orchestrator uses); any other key is refused, counted but not named. The values come back as `SecretStr` in a
`RunSecrets` (`hf_token`), which `runner_main` passes to the runner in place of the raw string.
Every failure is a named `RunError` whose message never carries the file's contents.

### The model server gets an allow-listed environment

The inference runner starts vLLM through `LocalBackend(env_inherit=False)` with
`model_server_environ()`: `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`,
`TMPDIR`, `LD_LIBRARY_PATH`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`,
`PYTHONUNBUFFERED`, and the `CUDA_*`, `NVIDIA_*`, `NCCL_*` and `VLLM_*` families, minus any name
containing `TOKEN`, `KEY`, `SECRET`, `PASSWORD` or `CREDENTIAL`. The spec, the secrets file's path
and any token are not the server's business. The list is wider than the minimum (user, locale,
temp and cache locations, NCCL transport settings) so a multi-GPU box keeps working, and it is
also narrower than what the server used to inherit: an ambient `HF_TOKEN`, `HF_ENDPOINT` or
`HF_HUB_*` setting no longer reaches it.

### Transition

An orchestrator from before this change still puts the token in `HF_WRITE_TOKEN`. When no
`FORGE_SECRETS_FILE` is set, the 0.4 runner reads that variable, removes it from its own
environment so no child inherits it, and records a one-line `phase` event (and a stderr
`warning:`) saying the delivery is deprecated. When a file IS configured the variable is never
read: an orchestrator that delivers by file cannot be steered back to the environment. The
fallback is removed in 0.5.0.

## Consequences

- **What closes.** The token is no longer in `wrapper.sh`, on any remote argv, in the job's
  environment, in the setup step's environment or in the model server's. In the normal path the
  file exists from submit until the runner starts.
- **What remains.** Code running as the same user during that window (the setup step's package
  installs included) can still read the file, as it can read anything else the user owns: a
  same-user process is not a boundary this design claims. It shortens the window and takes the
  value out of every place that is copied or inherited. A SIGKILL skips the trap; `cancel`'s own
  removal and the orchestrator's workdir cleanup are the backstops, and the cleanup must verify
  the removal. A submit stranded by a dead connection is covered by `SubmitCleanupError` and by
  the next submit's sweep; until one of those runs on that host, the file stays.
- **Callers.** A credential goes in `secrets`, not `env`. A `Task` with secrets cannot be written
  to YAML with them, and a SkyPilot submit with secrets fails. An SSH `remote_root` must be a
  directory dedicated to Forge: every submit narrows it to 0700 and sweeps it.
- **Runners.** `runner_main`'s callable takes `(writer, RunSecrets)` instead of
  `(writer, str | None)`.
- **Compatibility.** The `Task` shape changed, so this is a minor release under the release rule.
