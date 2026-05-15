"""Unit tests for `forge.llm.multimodal`."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError as PydanticValidationError

from forge.llm.messages import ContentPart
from forge.llm.multimodal import ImageContent, downscale_image

if TYPE_CHECKING:
    from pathlib import Path


# 1x1 transparent PNG — smallest legitimate image we can construct without Pillow
_TINY_PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII=",
)


class TestImageContentFactories:
    def test_from_url(self) -> None:
        img = ImageContent.from_url("https://example.com/cat.jpg")
        assert img.url == "https://example.com/cat.jpg"
        assert img.data is None
        assert img.mime_type == "image/jpeg"

    def test_from_bytes(self) -> None:
        img = ImageContent.from_bytes(b"\x89PNG\r\n", mime_type="image/png")
        assert img.url is None
        assert img.data is not None
        assert img.mime_type == "image/png"
        # Round-trip the base64.
        assert base64.b64decode(img.data) == b"\x89PNG\r\n"

    def test_from_bytes_default_mime(self) -> None:
        img = ImageContent.from_bytes(b"raw")
        assert img.mime_type == "image/jpeg"

    def test_from_path(self, tmp_path: Path) -> None:
        png_path = tmp_path / "tiny.png"
        png_path.write_bytes(_TINY_PNG_BYTES)
        img = ImageContent.from_path(png_path)
        assert img.url is None
        assert img.mime_type == "image/png"
        # Round-trip.
        assert base64.b64decode(img.data or "") == _TINY_PNG_BYTES

    def test_from_path_no_extension_falls_back_to_default(self, tmp_path: Path) -> None:
        # Files with no extension at all give mimetypes nothing to guess.
        path = tmp_path / "no_extension_here"
        path.write_bytes(b"raw")
        img = ImageContent.from_path(path)
        assert img.mime_type == "image/jpeg"


class TestImageContentValidation:
    def test_url_only_ok(self) -> None:
        img = ImageContent(url="https://example.com/cat.jpg")
        assert img.url == "https://example.com/cat.jpg"

    def test_data_only_ok(self) -> None:
        img = ImageContent(data="aGVsbG8=", mime_type="image/png")
        assert img.data == "aGVsbG8="

    def test_neither_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="requires either"):
            ImageContent()

    def test_both_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="must not have both"):
            ImageContent(url="https://x.example", data="abc")

    def test_is_a_content_part(self) -> None:
        # ImageContent must inherit from ContentPart so it can appear in
        # UserMessage.content lists.
        img = ImageContent.from_url("https://example.com/cat.jpg")
        assert isinstance(img, ContentPart)


class TestOpenAIFormat:
    def test_url_form(self) -> None:
        img = ImageContent.from_url("https://example.com/cat.jpg")
        assert img.to_openai_format() == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.jpg"},
        }

    def test_base64_form(self) -> None:
        img = ImageContent.from_bytes(b"raw", mime_type="image/png")
        out = img.to_openai_format()
        assert out["type"] == "image_url"
        # OpenAI's base64 form uses a data URL.
        assert out["image_url"]["url"].startswith("data:image/png;base64,")


class TestAnthropicFormat:
    def test_url_form(self) -> None:
        img = ImageContent.from_url("https://example.com/cat.jpg")
        assert img.to_anthropic_format() == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.jpg"},
        }

    def test_base64_form(self) -> None:
        img = ImageContent.from_bytes(b"raw", mime_type="image/png")
        out = img.to_anthropic_format()
        assert out == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img.data,
            },
        }


class TestGeminiFormat:
    def test_url_form(self) -> None:
        img = ImageContent.from_url("https://example.com/cat.jpg")
        assert img.to_gemini_format() == {
            "file_data": {
                "mime_type": "image/jpeg",
                "file_uri": "https://example.com/cat.jpg",
            },
        }

    def test_base64_form(self) -> None:
        img = ImageContent.from_bytes(b"raw", mime_type="image/png")
        out = img.to_gemini_format()
        assert out == {
            "inline_data": {"mime_type": "image/png", "data": img.data},
        }


class TestPerProviderShapesDiffer:
    """All three formats are distinct — sanity guard against drift."""

    def test_different_top_level_keys(self) -> None:
        img = ImageContent.from_bytes(b"x", mime_type="image/png")
        openai = img.to_openai_format()
        anthropic = img.to_anthropic_format()
        gemini = img.to_gemini_format()
        # OpenAI uses image_url, Anthropic uses source, Gemini uses inline_data.
        assert "image_url" in openai
        assert "source" in anthropic
        assert "inline_data" in gemini


class TestDownscale:
    def test_downscale_with_pillow(self) -> None:
        # Pillow ships as an optional [multimodal] extra; skip when missing.
        pytest.importorskip("PIL")
        # Build a small in-memory image with Pillow itself for the source.
        import io

        from PIL import Image

        source = Image.new("RGB", (4000, 3000), color=(255, 0, 0))
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        original_bytes = buf.getvalue()

        out_bytes, out_mime = downscale_image(original_bytes, max_dimension=512)
        assert out_mime == "image/jpeg"

        # Re-open to check the output dimensions.
        out_img = Image.open(io.BytesIO(out_bytes))
        # Longest side must be ≤ max_dimension.
        assert max(out_img.size) <= 512
        # Aspect ratio approximately preserved.
        ratio = out_img.size[0] / out_img.size[1]
        assert 1.3 < ratio < 1.4  # original was 4000:3000 ≈ 1.333

    def test_downscale_returns_jpeg_even_for_png_input(self) -> None:
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        source = Image.new("RGBA", (1024, 1024), color=(0, 255, 0, 128))
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        out_bytes, out_mime = downscale_image(buf.getvalue(), max_dimension=512)
        assert out_mime == "image/jpeg"
        # The first two bytes of a JPEG are FF D8 (start-of-image marker).
        assert out_bytes[:2] == b"\xff\xd8"

    def test_downscale_keeps_small_images_under_cap(self) -> None:
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        source = Image.new("RGB", (300, 200), color=(0, 0, 255))
        buf = io.BytesIO()
        source.save(buf, format="JPEG")
        out_bytes, _ = downscale_image(buf.getvalue(), max_dimension=2048)
        out_img = Image.open(io.BytesIO(out_bytes))
        # Pillow's thumbnail() never enlarges, so dims stay at 300x200.
        assert out_img.size == (300, 200)
