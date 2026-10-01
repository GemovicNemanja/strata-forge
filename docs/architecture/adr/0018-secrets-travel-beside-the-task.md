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
- `hide_input_in_errors=True` on the model, because a validation error otherwise echoes the
  offending input, and for `secrets` that is the plaintext value;
- `env` may not set `FORGE_SECRETS_FILE`, which only a backend may set, because the runner
  deletes the file it names.

### Delivery: one private file per job

Every backend that accepts secrets writes them as one JSON object to a file named
`.secrets.json`, mode 0600, in a directory mode 0700, and sets `FORGE_SECRETS_FILE` to the file's
absolute path. That path is the only secret-related thing the job's environment carries.

- **SSH.** The workdir is created by a single `bash -c` step that sets `umask 077`, requires the
  remote root to be a directory owned by the login user and narrows it to 0700, and creates the
  leaf with `mkdir -m 700` and no `-p`, so an existing directory or symlink fails the submit. The
  secrets file and the wrapper are both written with `umask 077 && set -C && cat > path`, their
  contents sent over the SSH channel's **stdin**, so neither appears in any command string, and
  `set -C` refuses a file or symlink already at the path. The wrapper exports the file's path
  and, before anything else, installs `trap 'rm -f "$FORGE_SECRETS_FILE"' EXIT`, so the file goes
  when setup fails, when the job ends and when `cancel` sends SIGTERM.
- **Local.** A fresh `tempfile.mkdtemp` directory, chmod 0700, holding a file created with
  `O_CREAT | O_EXCL | O_NOFOLLOW` and `fchmod` 0600 (explicit modes, because a umask can only
  narrow and a restrictive one would lock the child out). The child's shell gets the same trap;
  the backend removes the directory when the child exits and again in `cleanup`. An inherited
  `FORGE_SECRETS_FILE` from the parent's own environment is dropped, so a child never reads, and
  so deletes, its parent's file.
- **SkyPilot** raises on a non-empty `secrets` before any client call. Its only channel for a
  value is `envs`, which is the job's environment; failing closed beats a silent fallback.

### The runner reads it first, and deletes it

`strata_forge.pipelines._common.load_secrets()` runs first in `runner_main`, before the spec is
read and before any network call or subprocess. It refuses a `FORGE_SECRETS_FILE` that is not an
absolute path ending in `.secrets.json` without touching it, opens the file with `O_NOFOLLOW`,
requires a regular file owned by the effective user with no group or other bits, reads at most
64 KiB, and unlinks it whatever the outcome. The values come back as `SecretStr` in a
`RunSecrets` (`hf_token`), which `runner_main` passes to the runner in place of the raw string.
Every failure is a named `RunError` whose message never carries the file's contents.

### The model server gets an allow-listed environment

The inference runner starts vLLM through `LocalBackend(env_inherit=False)` with
`model_server_environ()`: `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`,
`TMPDIR`, `LD_LIBRARY_PATH`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`,
`PYTHONUNBUFFERED`, and the `CUDA_*`, `NVIDIA_*`, `NCCL_*` and `VLLM_*` families, minus any name
containing `TOKEN`, `KEY`, `SECRET`, `PASSWORD` or `CREDENTIAL`. The spec, the secrets file's path
and any token are not the server's business.

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
  value out of every place that is copied or inherited. A SIGKILL skips the trap; the orchestrator's
  workdir cleanup is the backstop, and it must verify the removal.
- **Callers.** A credential goes in `secrets`, not `env`. A `Task` with secrets cannot be written
  to YAML with them, and a SkyPilot submit with secrets fails.
- **Runners.** `runner_main`'s callable takes `(writer, RunSecrets)` instead of
  `(writer, str | None)`.
- **Compatibility.** The `Task` shape changed, so this is a minor release under the release rule.
