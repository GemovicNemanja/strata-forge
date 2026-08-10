# strata_forge.storage

Two async clients that make file and artifact movement look the same everywhere. `StorageGateway`
wraps `fsspec`, so reading and writing bytes or text works identically against local disk, S3, GCS,
Azure Blob and the Hugging Face Hub — the protocol comes from the URL scheme, with listing,
copying, moving and deletion on the same surface. `HFHubClient` wraps the Hugging Face Hub for
model and dataset push/pull, including base and fine-tuned weights, and resolves its token and
endpoint from an explicit argument, then `strata_forge.config`, then `huggingface_hub`'s own
resolver.

Both need the `[storage]` extra; the import error names it if you forget.

Reference:
[docs/modules/storage.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/storage.md).
