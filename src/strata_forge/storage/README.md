# strata_forge.storage

`StorageGateway` provides async file operations over local, S3, GCS, Azure Blob and other
fsspec backends. `HFHubClient` handles repo lifecycle, snapshots and individual files with
lazy optional dependencies. Revision resolution returns full input commit SHAs; `commit_file`
and `commit_folder` return immutable `CommitRef` values while existing upload methods retain
their string return values.

See [the storage reference](../../../docs/modules/storage.md) for credentials, backend options
and public API details.
