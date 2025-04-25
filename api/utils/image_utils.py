"""
Utility functions for image processing
Includes functions for format detection, conversion, and manipulation
"""

import re
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Union
from api.config import AVIF_SUPPORTED, SVG_SUPPORTED, logger


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB mode if it's not already.
    Handles RGBA, LA, and other modes by converting to RGB.

    Args:
        image: PIL Image object

    Returns:
        PIL Image in RGB mode
    """
    logger.info(f"Image mode: {image.mode}")
    if image.mode not in ("RGB", "L"):
        if image.mode == "RGBA":
            # Create a white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            # Paste the image using the alpha channel as mask
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            # Convert any other mode to RGB
            image = image.convert("RGB")
    return image


def is_base64_image_url(url: str) -> bool:
    """
    Check if the URL is a base64 encoded image

    Args:
        url: String URL to check

    Returns:
        True if the URL is a base64 encoded image
    """
    return url.startswith("data:image/")


def decode_base64_image_url(base64_url: str) -> Tuple[bytes, str]:
    """
    Decode base64 image URL to raw bytes and format
    Format: data:image/png;base64,base64_data

    Args:
        base64_url: Base64 encoded image URL

    Returns:
        Tuple of (image_bytes, format_hint)
    """
    # Extract mime type and base64 data
    pattern = r"data:image/([a-zA-Z]+);base64,(.+)"
    match = re.match(pattern, base64_url)

    if not match:
        raise ValueError("Invalid base64 image URL format")

    image_format, base64_data = match.groups()

    # Decode base64 data
    image_data = base64.b64decode(base64_data)

    return image_data, image_format


def detect_image_format(
    content_bytes: bytes, url: Optional[str] = None, filename: Optional[str] = None
) -> Optional[str]:
    """
    Detect image format from file content, URL or filename.
    Returns the format name (lowercase) or None if unknown.

    Args:
        content_bytes: First few bytes of the image file
        url: Optional URL where the image was downloaded from
        filename: Optional filename of the image

    Returns:
        Format name (lowercase) or None if unknown
    """
    # Check magic bytes
    if content_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif content_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    elif content_bytes.startswith(b"GIF8"):
        return "gif"
    elif content_bytes.startswith(b"RIFF") and content_bytes[8:12] == b"WEBP":
        return "webp"
    elif content_bytes[:5] == b"<?xml" or content_bytes[:4] == b"<svg":
        return "svg"
    elif content_bytes.startswith(b"\x00\x00\x00\x1cftyp"):
        # AVIF files typically have 'ftyp' marker
        if b"avif" in content_bytes[4:16] or b"avis" in content_bytes[4:16]:
            return "avif"

    # Check URL or filename extension
    if url:
        url_lower = url.lower()
        if url_lower.endswith(".svg"):
            return "svg"
        elif url_lower.endswith(".png"):
            return "png"
        elif url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
            return "jpeg"
        elif url_lower.endswith(".webp"):
            return "webp"
        elif url_lower.endswith(".avif"):
            return "avif"
        elif url_lower.endswith(".gif"):
            return "gif"

    if filename:
        filename_lower = filename.lower()
        if filename_lower.endswith(".svg"):
            return "svg"
        elif filename_lower.endswith(".png"):
            return "png"
        elif filename_lower.endswith(".jpg") or filename_lower.endswith(".jpeg"):
            return "jpeg"
        elif filename_lower.endswith(".webp"):
            return "webp"
        elif filename_lower.endswith(".avif"):
            return "avif"
        elif filename_lower.endswith(".gif"):
            return "gif"

    # Unknown format
    return None
