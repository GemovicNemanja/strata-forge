"""Submit a small task to :class:`LocalBackend` and tail its output.

The example writes a script that prints a few lines, submits it to
the in-process :class:`LocalBackend`, polls for completion, and
prints the captured stdout. No remote infrastructure needed —
this is the smallest possible end-to-end exercise of
:mod:`strata_forge.compute`.

Usage::

    uv run python examples/30_compute_local.py
"""

from __future__ import annotations

import asyncio

from strata_forge.compute import LocalBackend, Task


async def _main() -> None:
    task = Task(
        name="hello-compute",
        run=(
            'python -c "import time, sys; '
            "[print(f'hello-{i}', flush=True) or time.sleep(0.05) for i in range(3)]; "
            'sys.exit(0)"'
        ),
    )

    backend = LocalBackend()
    print(f"--- submitting task {task.name!r}")
    job = await backend.submit(task)
    print(f"submitted job id={job.id}")

    while True:
        status = await backend.status(job)
        print(f"  status: {status.state}")
        if status.state not in ("pending", "running"):
            break
        await asyncio.sleep(0.1)

    logs = await backend.logs(job)
    print("--- logs ---")
    print(logs.rstrip())

    await backend.cleanup(job)
    print("--- cleaned up.")


if __name__ == "__main__":
    asyncio.run(_main())
