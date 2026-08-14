"""
image_engine.py
================
Safe image handling built on Pillow.

Supports PNG / JPEG / WebP loading plus resize, crop, rotate,
compress and grayscale operations. This module NEVER executes code
from an uploaded file - Pillow is used purely as a pixel-data decoder,
`Image.verify()` / re-encoding is used defensively, and Pillow's
decompression-bomb protection (`Image.MAX_IMAGE_PIXELS`) stays enabled.

Hard limits (see safety.py):
    * file size            -> MAX_IMAGE_SIZE_BYTES
    * width / height        -> MAX_IMAGE_DIMENSION
    * images per request     -> MAX_IMAGES_PER_REQUEST (enforced by the
                                caller / API layer, since this module
                                processes one image at a time)
    * processing time         -> IMAGE_PROCESSING_TIMEOUT_SECONDS
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

from .safety import (
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    IMAGE_PROCESSING_TIMEOUT_SECONDS,
    SafetyError,
    time_limit,
)

try:
    from PIL import Image, ImageOps

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when Pillow is absent
    Image = None
    ImageOps = None
    PIL_AVAILABLE = False

_ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _require_pillow() -> Optional[dict]:
    if not PIL_AVAILABLE:
        return _fail("Pillow is not installed. Run: pip install -r requirements.txt")
    return None


def _load_and_validate(image_bytes: bytes) -> "Image.Image":
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise SafetyError("Image content must be raw bytes.")
    if len(image_bytes) == 0:
        raise SafetyError("Empty image.")
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise SafetyError(f"Image exceeds the maximum allowed size ({MAX_IMAGE_SIZE_BYTES} bytes).")

    # Verify the file is a genuine, well-formed image before doing
    # anything else with it (catches truncated/malicious files early).
    try:
        probe = Image.open(io.BytesIO(image_bytes))
        probe.verify()
    except Exception as e:  # noqa: BLE001
        raise SafetyError(f"File is not a valid image. ({e})") from e

    if probe.format not in _ALLOWED_FORMATS:
        raise SafetyError(f"Unsupported image format '{probe.format}'. Allowed: {sorted(_ALLOWED_FORMATS)}.")

    # Re-open after verify() (verify() leaves the file unusable for
    # further operations) and check dimensions.
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    w, h = img.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise SafetyError(f"Image dimensions exceed the maximum allowed ({MAX_IMAGE_DIMENSION}px).")
    return img


def _encode(img: "Image.Image", fmt: str = "PNG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    save_kwargs: Dict[str, Any] = {}
    if fmt.upper() in ("JPEG", "WEBP"):
        save_kwargs["quality"] = max(1, min(int(quality), 100))
        if fmt.upper() == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
    img.save(buf, format=fmt.upper(), **save_kwargs)
    return buf.getvalue()


def get_metadata(image_bytes: bytes) -> Dict[str, Any]:
    if (err := _require_pillow()) is not None:
        return err
    try:
        with time_limit(IMAGE_PROCESSING_TIMEOUT_SECONDS):
            img = _load_and_validate(image_bytes)
        return _ok(width=img.width, height=img.height, format=img.format, mode=img.mode)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely read this image. ({e})")


def resize(image_bytes: bytes, width: int, height: int, output_format: str = "PNG") -> Dict[str, Any]:
    if (err := _require_pillow()) is not None:
        return err
    try:
        width, height = int(width), int(height)
        if width <= 0 or height <= 0 or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise SafetyError(f"Target dimensions must be between 1 and {MAX_IMAGE_DIMENSION}px.")
        with time_limit(IMAGE_PROCESSING_TIMEOUT_SECONDS):
            img = _load_and_validate(image_bytes)
            resized = img.resize((width, height), Image.LANCZOS)
            data = _encode(resized, output_format)
        return _ok(width=width, height=height, format=output_format.upper(),
                    size_bytes=len(data), image_bytes=data)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely resize this image. ({e})")


def crop(image_bytes: bytes, left: int, top: int, right: int, bottom: int,
          output_format: str = "PNG") -> Dict[str, Any]:
    if (err := _require_pillow()) is not None:
        return err
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise SafetyError("Invalid crop box.")
        with time_limit(IMAGE_PROCESSING_TIMEOUT_SECONDS):
            img = _load_and_validate(image_bytes)
            if right > img.width or bottom > img.height:
                raise SafetyError("Crop box exceeds image bounds.")
            cropped = img.crop((left, top, right, bottom))
            data = _encode(cropped, output_format)
        return _ok(width=cropped.width, height=cropped.height, format=output_format.upper(),
                    size_bytes=len(data), image_bytes=data)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely crop this image. ({e})")


def rotate(image_bytes: bytes, degrees: float, output_format: str = "PNG") -> Dict[str, Any]:
    if (err := _require_pillow()) is not None:
        return err
    try:
        degrees = float(degrees) % 360
        with time_limit(IMAGE_PROCESSING_TIMEOUT_SECONDS):
            img = _load_and_validate(image_bytes)
            rotated = img.rotate(-degrees, expand=True)
            data = _encode(rotated, output_format)
        return _ok(width=rotated.width, height=rotated.height, format=output_format.upper(),
                    size_bytes=len(data), image_bytes=data)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely rotate this image. ({e})")


def compress(image_bytes: bytes, quality: int = 70, output_format: str = "JPEG") -> Dict[str, Any]:
    if (err := _require_pillow()) is not None:
        return err
    try:
        quality = max(1, min(int(quality), 100))
        with time_limit(IMAGE_PROCESSING_TIMEOUT_SECONDS):
            img = _load_and_validate(image_bytes)
            data = _encode(img, output_format, quality)
        return _ok(format=output_format.upper(), quality=quality,
                    size_bytes=len(data), image_bytes=data)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely compress this image. ({e})")


def to_grayscale(image_bytes: bytes, output_format: str = "PNG") -> Dict[str, Any]:
    if (err := _require_pillow()) is not None:
        return err
    try:
        with time_limit(IMAGE_PROCESSING_TIMEOUT_SECONDS):
            img = _load_and_validate(image_bytes)
            gray = ImageOps.grayscale(img)
            data = _encode(gray, output_format)
        return _ok(width=gray.width, height=gray.height, format=output_format.upper(),
                    size_bytes=len(data), image_bytes=data)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely convert this image to grayscale. ({e})")
