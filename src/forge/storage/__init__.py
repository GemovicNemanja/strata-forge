"""fsspec gateway and Hugging Face Hub model/dataset handling.

The public surface is:

- :class:`StorageGateway` — async fsspec wrapper for read /
  write / list / copy / move / delete across local, S3, GCS,
  Azure Blob, HuggingFace Hub, and any other fsspec-compatible
  target.
- :data:`FileInfo` — per-entry metadata type.
- :class:`HFHubClient` — async wrapper over
  ``huggingface_hub.HfApi`` for whole-repo push/pull and
  single-file operations.
- :data:`RepoType` — ``"model"`` / ``"dataset"`` / ``"space"``.

The fsspec module, cloud-specific backends, and
``huggingface_hub`` ride behind the ``[storage]`` extra and are
lazy-imported on first use, so ``import forge.storage`` succeeds
without them.
"""

from forge.storage.gateway import FileInfo, StorageGateway
from forge.storage.hf_hub import HFHubClient, RepoType

__all__ = [
    "FileInfo",
    "HFHubClient",
    "RepoType",
    "StorageGateway",
]
