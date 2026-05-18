"""fsspec gateway and Hugging Face Hub model/dataset handling.

The public surface is:

- :class:`StorageGateway` — async fsspec wrapper for read /
  write / list / copy / move / delete across local, S3, GCS,
  Azure Blob, HuggingFace Hub, and any other fsspec-compatible
  target.
- :data:`FileInfo` — per-entry metadata type.

The fsspec module and cloud-specific backends ride behind the
``[storage]`` extra and are lazy-imported on first use, so
``import forge.storage`` succeeds without them.
"""

from forge.storage.gateway import FileInfo, StorageGateway

__all__ = [
    "FileInfo",
    "StorageGateway",
]
