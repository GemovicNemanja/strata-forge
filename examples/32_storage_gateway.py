"""Read / write / list / copy through :class:`StorageGateway`.

Uses the local filesystem so it runs anywhere without
credentials. Touch ``options={"s3": {...}}`` on the gateway
constructor to point the same code at AWS / GCS / Azure / HF
Hub.

Usage::

    uv run python examples/32_storage_gateway.py
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from strata_forge.storage import StorageGateway


async def _main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="forge-storage-"))
    try:
        gw = StorageGateway()

        print(f"--- writing under {workdir}")
        await gw.write_text(
            str(workdir / "manifest.json"),
            '{"runs": 3, "model": "claude-opus-4-7"}\n',
        )
        await gw.write_bytes(str(workdir / "blob.bin"), b"\x00\x01\x02\xff")

        print("\n--- info on manifest.json")
        info = await gw.info(str(workdir / "manifest.json"))
        print(f"  name: {info['name']}")
        print(f"  size: {info['size']} bytes")

        print("\n--- listing workdir")
        for name in await gw.ls(str(workdir)):
            print(f"  {name}")

        print("\n--- copying manifest.json → manifest.copy.json")
        await gw.copy(
            str(workdir / "manifest.json"),
            str(workdir / "manifest.copy.json"),
        )
        for entry in await gw.ls(str(workdir), detail=True):
            print(f"  {entry['name']:50s}  {entry['size']:>4} B")

        print("\n--- reading the copy back")
        text = await gw.read_text(str(workdir / "manifest.copy.json"))
        print(text.rstrip())

        print("\n--- deleting workdir contents")
        await gw.delete(str(workdir), recursive=True)
        print(f"  exists after delete: {await gw.exists(str(workdir))}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(_main())
