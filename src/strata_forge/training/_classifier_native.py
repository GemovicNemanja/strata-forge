"""Own primary-classifier native work until it physically returns, including on cancellation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable


async def owned_native[T](
    function: Callable[[], T], *, cancelled: threading.Event | None = None
) -> T:
    """Drain this native call; cancellation never admits the caller's next operation.

    A trainer receives its existing cooperative cancellation event immediately. A native
    call may still require transport-supervised process termination; a timeout cannot make
    its thread stop. Control exceptions are caught inside the thread so they cannot escape
    through the event loop while ordinary SDK exceptions retain their existing boundary.
    """

    def contained() -> tuple[bool, T | BaseException]:
        try:
            return True, function()
        except BaseException as error:
            return False, error

    task = asyncio.create_task(asyncio.to_thread(contained))
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
            if cancelled is not None:
                cancelled.set()
    succeeded, value = task.result()
    if interrupted:
        raise asyncio.CancelledError
    if succeeded:
        return cast("T", value)
    error = cast("BaseException", value)
    if isinstance(error, Exception):
        raise error
    raise ValidationError("classifier_native_failed") from None
