# Agent rules — strata_forge.storage

`strata_forge.storage` is the file-storage and Hugging Face Hub layer.
It ships two cooperating clients: :class:`StorageGateway` for
generic fsspec-backed read / write / list / copy across local,
S3, GCS, Azure Blob, and any other fsspec-compatible target;
and :class:`HFHubClient` for higher-level model / dataset
push-pull semantics on the HuggingFace Hub.

## Purpose

- :class:`StorageGateway` — async fsspec wrapper. URL → protocol
  auto-detection, per-protocol options forwarding, filesystem
  caching, same-protocol native copy/move plus cross-protocol
  byte streaming.
- :data:`FileInfo` — per-entry metadata type alias.
- :class:`HFHubClient` — async wrapper over
  ``huggingface_hub.HfApi``. Repo lifecycle (create / delete /
  list), single-file ops, whole-repo snapshots, convenience
  push/pull wrappers for both models and datasets.
- :data:`RepoType` — ``"model"`` / ``"dataset"`` / ``"space"``
  Literal alias.

## Boundaries

- **Owns:** ``gateway.py``, ``hf_hub.py``.
- **Imports from inside ``forge``:** nothing — this module is a
  pure I/O leaf. Higher-level modules (``strata_forge.datasets``,
  ``strata_forge.training``, ``strata_forge.compute``) consume
  :mod:`strata_forge.storage`, never the other way around.
- **External deps:** ``fsspec`` plus its protocol backends
  (``s3fs``, ``gcsfs``, ``adlfs``) and ``huggingface_hub`` all
  ride behind the ``[storage]`` extra. Both clients lazy-import
  on first use so ``import strata_forge.storage`` works without the
  extra.

## Public API

The module's ``__init__.py`` re-exports:

- Generic file storage: :class:`StorageGateway`,
  :data:`FileInfo`.
- HuggingFace Hub: :class:`HFHubClient`, :data:`RepoType`.

Errors raised from this module are :class:`ImportError` (when
the extra is missing), :class:`TypeError` (when fsspec returns
a shape we can't make sense of), or :class:`NotImplementedError`
(for cross-protocol recursive copy).

## Internal patterns

- **Sync SDK + ``asyncio.to_thread``.** Both fsspec and
  ``huggingface_hub`` are sync-first. We wrap every call in
  ``asyncio.to_thread`` so the public surface stays async-uniform
  with the rest of Forge — same pattern as
  :class:`SkyPilotBackend`.
- **Filesystem caching keyed by protocol.** Each
  :class:`StorageGateway` keeps one ``fsspec`` filesystem
  instance per protocol so credentials / endpoints / kwargs
  resolve once.
- **Per-protocol options.** Constructors take
  ``options={"s3": {...}, "gs": {...}}`` rather than flat
  kwargs so a single gateway can talk to multiple targets with
  different auth.
- **Same-protocol native, cross-protocol streamed.** Copy and
  move use the filesystem's native ``copy``/``move`` when source
  and destination share a protocol; otherwise we read into
  memory and write to the other side. Recursive cross-protocol
  copy raises :class:`NotImplementedError` deliberately — large
  transfers need a smarter path (multipart upload, signed URLs).
- **Escape hatch on every HF Hub method.** :class:`HFHubClient`
  exposes ``extras={}`` on every public method so callers can
  pass any HfApi kwarg verbatim. Forge never blocks the full
  HfApi surface.

## Test expectations

- Unit tests under ``tests/unit/storage/``, one file per source
  module.
- Coverage: the enforced gate is the repo-wide 85 % line floor
  (``fail_under`` in ``pyproject.toml``); treat a drop in this module
  as a regression.
- Both clients are unit-tested against ``sys.modules``-injected
  fakes (``_FakeFilesystem``, ``_FakeApi``) — no real fsspec or
  hf_hub imports in the test suite.

## Gotchas

- **Don't add Forge-level retry around fsspec calls.** fsspec
  backends already implement their own retry semantics; adding
  another layer compounds delays and confuses error semantics.
- **Local paths use the ``file`` protocol.** ``./foo`` and
  ``/tmp/bar`` are routed through ``fsspec.filesystem("file")``
  — useful for testing but slower than direct pathlib I/O. If
  you need low-latency local reads on a hot path, bypass the
  gateway and use ``aiofiles`` or sync pathlib.
- **Windows drive paths** (``C:\foo``) are detected via single-
  letter scheme and routed to the ``file`` protocol; multi-letter
  schemes like ``s3://`` are protocols.
- **HFHubClient resolves the token in three layers.** When
  ``token=`` is passed explicitly it wins. Otherwise the client
  falls back to :class:`strata_forge.config.HuggingFaceConfig` (which
  reads ``HF_TOKEN`` from env or ``.env``), and finally to
  ``huggingface_hub``'s own resolver (env var or cached login).
  The fallback runs lazily inside ``_api`` so importing
  :mod:`strata_forge.storage` stays free of side effects.
- **Use ``[storage]`` for any cloud backend.** ``fsspec`` itself
  is tiny, but the protocol-specific packages (``s3fs``,
  ``gcsfs``, ``adlfs``) are not. The extra brings them all.

## When to update this file

- Adding a new storage backend wrapper (e.g. a presigned-URL
  helper, an HF Spaces helper).
- Changing the cross-protocol copy semantics.
- Changing the lazy-import boundary.
- Adding a new heavy dep to the ``[storage]`` extra.
