# `forge.storage` — fsspec gateway, HuggingFace Hub model/dataset push/pull

`forge.storage` is the file-storage and HuggingFace Hub layer.
Two cooperating clients live here:
:class:`StorageGateway` for generic fsspec-backed file
operations across local / S3 / GCS / Azure / HTTP / HF Hub, and
:class:`HFHubClient` for higher-level model and dataset
push/pull on the HuggingFace Hub.

Integration points:

- **Gateway:** :class:`StorageGateway`, :data:`FileInfo`.
- **Hub:** :class:`HFHubClient`, :data:`RepoType`.

Both clients defer their heavy imports — ``fsspec``,
``s3fs`` / ``gcsfs`` / ``adlfs``, ``huggingface_hub`` — until
the first network call, so ``import forge.storage`` works
without the ``[storage]`` extra installed.

Module rules: [`src/forge/storage/CLAUDE.md`](../../src/forge/storage/CLAUDE.md).
Source: [`src/forge/storage/`](../../src/forge/storage/).

---

## Contents

- [Quickstart](#quickstart)
- [StorageGateway](#storagegateway)
- [HFHubClient](#hfhubclient)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

Cross-target file shuffling:

```python
from forge.storage import StorageGateway

gw = StorageGateway(
    options={
        "s3": {"key": "...", "secret": "...",
               "client_kwargs": {"region_name": "us-east-1"}},
    }
)
# Local read, cloud write.
text = await gw.read_text("./reports/2026-04.txt")
await gw.write_text("s3://my-bucket/archive/2026-04.txt", text)

# Same-protocol copy uses the filesystem's native operation.
await gw.copy("s3://my-bucket/a.bin", "s3://my-bucket/b.bin")
```

HF Hub model push/pull:

```python
from forge.storage import HFHubClient

hub = HFHubClient(token="hf_xxx")

# Upload a fine-tuned model directory to a new repo.
commit_url = await hub.push_model(
    "./checkpoints/llama-3.1-8b-sft",
    "me/llama-3.1-8b-sft",
    commit_message="initial release",
    private=True,
)

# Pull a different revision back down.
local_path = await hub.pull_model(
    "me/llama-3.1-8b-sft",
    "./weights",
    revision="v0.2.0",
)
```

## StorageGateway

`StorageGateway` is an async wrapper over `fsspec`. It picks the
right filesystem based on each URL's scheme — `./foo` and
`/tmp/bar` go through `file://`; `s3://`, `gs://`, `az://`,
`https://`, `hf://` are all routed to their respective fsspec
backends.

| Method | Returns | Notes |
|---|---|---|
| `read_bytes(url)` | `bytes` | |
| `read_text(url, *, encoding="utf-8")` | `str` | |
| `write_bytes(url, data)` | `None` | overwrites |
| `write_text(url, text, *, encoding="utf-8")` | `None` | overwrites |
| `exists(url)` | `bool` | |
| `info(url)` | `FileInfo` | per-entry dict |
| `ls(url, *, detail=False)` | `list[str]` or `list[FileInfo]` | |
| `delete(url, *, recursive=False)` | `None` | |
| `copy(src, dst, *, recursive=False)` | `None` | native when protocols match |
| `move(src, dst)` | `None` | copy + delete across protocols |
| `makedirs(url, *, exist_ok=True)` | `None` | |

### Constructor

```python
StorageGateway(*, options: Mapping[str, Mapping[str, Any]] | None = None)
```

Per-protocol options are passed verbatim to
`fsspec.filesystem(protocol, **opts)` — credentials, regions,
anonymous-access flags, custom endpoints, anything fsspec
accepts.

### Cross-protocol semantics

- **Single-file copy / move across protocols** reads source
  into memory then writes to the destination. Fine for
  manifests, configs, JSONL; not great for multi-GB tensors.
- **Recursive copy across protocols** raises
  `NotImplementedError`. Large recursive transfers want
  multipart-aware tooling (`aws s3 sync`, gsutil, `huggingface-cli
  upload-large-folder`). Forge surfaces the limit explicitly
  rather than degrading silently.

### Filesystem caching

Each gateway keeps one `fsspec` filesystem instance per protocol
in a private cache. Per-protocol options apply once at first
use. To rotate credentials, construct a new gateway.

## HFHubClient

`HFHubClient` covers the high-level operations Forge needs:
repo lifecycle, single-file transfers, whole-repo snapshots, and
convenience push/pull. It wraps `huggingface_hub.HfApi`
synchronously off-loaded to a thread.

### Constructor

```python
HFHubClient(
    *,
    token: str | None = None,
    endpoint: str | None = None,
    api: Any | None = None,
)
```

- `token`: HuggingFace access token. When `None`, the
  underlying `HfApi` falls back to `HF_TOKEN` env var or cached
  login.
- `endpoint`: custom Hub URL — useful for the Enterprise tier
  or air-gapped Hub deployments.
- `api`: pre-built `HfApi` for tests or for sharing across
  multiple clients.

### Repo lifecycle

```python
url = await hub.create_repo(
    "me/my-model", repo_type="model", private=True, exist_ok=True
)
await hub.delete_repo("me/old-thing", repo_type="dataset")
files = await hub.list_repo_files("me/my-model", revision="v1.0")
```

### Single-file operations

```python
local_path = await hub.download_file(
    "me/my-model",
    "model.safetensors",
    local_dir="./weights",
    revision="v0.2.0",
)
commit_url = await hub.upload_file(
    "./local/model.safetensors",
    "model.safetensors",
    "me/my-model",
    commit_message="bf16 weights",
)
```

### Whole-repo snapshots

```python
path = await hub.download_snapshot(
    "me/llama-finetune",
    local_dir="./snap",
    allow_patterns=("*.safetensors", "*.json"),
    ignore_patterns=("*.bin",),
)
commit_url = await hub.upload_folder(
    "./checkpoints/llama",
    "me/llama-finetune",
    commit_message="snapshot 42",
)
```

### Convenience push/pull

```python
# Create-if-missing then upload.
await hub.push_model("./ckpt", "me/llama-sft", private=True)
await hub.push_dataset("./data", "me/preference-pairs")

# Snapshot download.
await hub.pull_model("me/llama-sft", "./weights")
await hub.pull_dataset("me/preference-pairs", "./data")
```

### Escape hatch

Every public method takes an `extras: Mapping[str, Any] | None`
parameter. Anything you pass flows verbatim to the underlying
`HfApi` method — `create_pr=True`, `space_sdk="gradio"`,
`force_download=True`, etc. Forge never blocks access to the
full HF Hub surface.

## Lazy-import contract

- `fsspec` and the cloud-specific protocol packages (`s3fs`,
  `gcsfs`, `adlfs`) sit behind the `[storage]` extra. They're
  imported inside `StorageGateway._filesystem`.
- `huggingface_hub` is imported inside `HFHubClient._api` and
  `HFHubClient.download_file` / `HFHubClient.download_snapshot`.
- When the extra isn't installed, the first networked call
  raises `ImportError` with the install hint
  `pip install 'ai-forge[storage]'`.

## Troubleshooting

- **`ImportError: The [storage] extra is required`:** install
  the extra. On macOS / Linux this is a small set of pure-Python
  packages plus `s3fs` (which depends on `aiobotocore`).
- **`NotImplementedError: cross-protocol recursive copy`:**
  intentional — switch to a vendor-native tool for large
  recursive transfers (`aws s3 sync`, `gsutil rsync`, etc.).
- **`TypeError: unexpected info() result type`:** the underlying
  fsspec backend returned something other than a dict. Most
  built-in backends return dicts; custom backends might not.
  Open an issue with the protocol and backend version.
- **Hub uploads stall:** disable progress bars
  (`HF_HUB_DISABLE_PROGRESS_BARS=1`) — they can deadlock in
  some terminal environments. Or pass `extras={"silent": True}`
  to the relevant method.
- **Hub 401 / 403:** confirm `HF_TOKEN` is set or pass `token=`
  to the client. For private repos, the token needs `write`
  scope.
- **Cross-account S3 copy fails:** native fsspec copy uses the
  destination filesystem's credentials. To copy across
  accounts, configure two gateways with different `options=`
  and let the cross-protocol path stream bytes (small files) or
  use `aws s3 cp` for large transfers.
