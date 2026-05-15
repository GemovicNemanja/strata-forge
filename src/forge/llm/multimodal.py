"""Image content for multimodal LLM calls.

``ImageContent`` is a :class:`ContentPart` subclass that holds either a
URL or a base64-encoded image payload. The same object can be rendered
into each provider's wire format via the ``to_*_format()`` methods —
OpenAI, Anthropic, and Gemini all expect slightly different shapes.

``downscale_image`` resizes an image to fit within a maximum dimension on
the longest side. It lazy-imports Pillow (installed via the
``[multimodal]`` extra) so the rest of the LLM module doesn't pay for
that dependency.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import model_validator

from forge.llm.messages import ContentPart

if TYPE_CHECKING:
    from forge.core.types import PathLike

__all__ = ["ImageContent", "downscale_image"]


_DEFAULT_MIME = "image/jpeg"


class ImageContent(ContentPart):
    """An image part in a multimodal message.

    Exactly one of ``url`` or ``data`` must be set. ``data`` is a base64
    string (no ``data:...;base64,`` prefix — the providers' wire formats
    add the prefix themselves when one is required).
    """

    type: Literal["image"] = "image"
    url: str | None = None
    data: str | None = None
    mime_type: str = _DEFAULT_MIME

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImageContent:
        has_url = self.url is not None
        has_data = self.data is not None
        if not has_url and not has_data:
            msg = "ImageContent requires either `url` or `data`"
            raise ValueError(msg)
        if has_url and has_data:
            msg = "ImageContent must not have both `url` and `data`"
            raise ValueError(msg)
        return self

    # --- factories -------------------------------------------------------

    @classmethod
    def from_url(cls, url: str) -> ImageContent:
        """Construct from a publicly-reachable image URL."""
        return cls(url=url)

    @classmethod
    def from_path(cls, path: PathLike) -> ImageContent:
        """Construct from a local file path.

        Reads the bytes, base64-encodes them, and guesses the MIME type
        from the file extension (defaults to ``image/jpeg``).
        """
        path_obj = Path(path)
        raw = path_obj.read_bytes()
        guessed, _ = mimetypes.guess_type(path_obj.name)
        return cls.from_bytes(raw, mime_type=guessed or _DEFAULT_MIME)

    @classmethod
    def from_bytes(cls, data: bytes, *, mime_type: str = _DEFAULT_MIME) -> ImageContent:
        """Construct from raw image bytes."""
        encoded = base64.b64encode(data).decode("ascii")
        return cls(data=encoded, mime_type=mime_type)

    # --- per-provider wire formats --------------------------------------

    def to_openai_format(self) -> dict[str, Any]:
        """Render as OpenAI's ``image_url`` content part."""
        if self.url is not None:
            return {"type": "image_url", "image_url": {"url": self.url}}
        # OpenAI accepts base64 inline via a data URL.
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime_type};base64,{self.data}"},
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Render as Anthropic's ``image`` content part."""
        if self.url is not None:
            return {
                "type": "image",
                "source": {"type": "url", "url": self.url},
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": self.data,
            },
        }

    def to_gemini_format(self) -> dict[str, Any]:
        """Render as Gemini's ``inline_data`` / ``file_data`` content part."""
        if self.url is not None:
            return {
                "file_data": {"mime_type": self.mime_type, "file_uri": self.url},
            }
        return {
            "inline_data": {"mime_type": self.mime_type, "data": self.data},
        }


def downscale_image(
    data: bytes,
    *,
    max_dimension: int = 2048,
    quality: int = 85,
) -> tuple[bytes, str]:
    """Downscale ``data`` to fit within ``max_dimension`` on the longest side.

    Returns ``(downscaled_bytes, mime_type)`` — re-encodes as JPEG for
    consistent size savings regardless of input format.

    Args:
        data: Source image bytes (PNG / JPEG / etc.).
        max_dimension: Maximum length, in pixels, of the longest side of
            the output image. Images smaller than this are returned as JPEG
            but otherwise unchanged.
        quality: JPEG quality (1-95). Higher = larger file, better quality.

    Raises:
        ImportError: When Pillow is not installed. Install it via
            ``pip install ai-forge[multimodal]``.
    """
    try:
        import io

        from PIL import Image
    except ImportError as exc:
        msg = "downscale_image requires Pillow; install via `pip install ai-forge[multimodal]`."
        raise ImportError(msg) from exc

    image = Image.open(io.BytesIO(data))
    image.thumbnail((max_dimension, max_dimension))
    # Convert any palette / RGBA modes to RGB so JPEG encoding succeeds.
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"
