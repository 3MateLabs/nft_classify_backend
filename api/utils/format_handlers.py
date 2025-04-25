"""
Format-specific image handlers
Contains specialized functions for processing different image formats
"""

from io import BytesIO
from PIL import Image
from typing import Optional, Union, BinaryIO
import cairosvg
from api.config import AVIF_SUPPORTED, SVG_SUPPORTED, logger


def render_svg_with_cairosvg(svg_content: Union[bytes, str]) -> Image.Image:
    """
    Render an SVG file using CairoSVG.
    Returns a PIL Image object.

    Args:
        svg_content: The SVG content as bytes or string

    Returns:
        PIL Image object
    """
    if not SVG_SUPPORTED:
        raise ImportError(
            "CairoSVG is not installed. Please install it with: pip install cairosvg"
        )

    # Convert SVG to PNG using cairosvg
    if isinstance(svg_content, bytes):
        png_data = cairosvg.svg2png(bytestring=svg_content)
    else:
        png_data = cairosvg.svg2png(string=svg_content)

    # Convert to PIL Image
    image = Image.open(BytesIO(png_data))
    return image


def open_image_with_format_specific_handling(
    content_stream: BinaryIO, format_hint: Optional[str] = None
) -> Image.Image:
    """
    Open an image using format-specific handling strategies.

    Args:
        content_stream: BytesIO or file-like object containing the image data
        format_hint: Optional format hint (e.g., 'png', 'svg', 'avif')

    Returns:
        PIL Image object
    """
    from api.utils.image_utils import detect_image_format

    # Read content to detect format if needed
    if not format_hint:
        # Save current position
        pos = content_stream.tell()
        # Read initial bytes for format detection
        content_bytes = content_stream.read(16)
        # Reset stream position
        content_stream.seek(pos)
        format_hint = detect_image_format(content_bytes)

    if format_hint == "svg":
        # For SVG, use CairoSVG rendering
        # First read the full content
        content_bytes = content_stream.read()
        # Reset stream for possible future use
        content_stream.seek(0)
        return render_svg_with_cairosvg(content_bytes)

    elif format_hint == "avif" and AVIF_SUPPORTED:
        # For AVIF, use the PIL with pillow-avif-plugin
        try:
            return Image.open(content_stream)
        except Exception as e:
            # If avif plugin fails, attempt to use another method
            raise Exception(
                f"Error opening AVIF image: {e}. Try installing pillow-avif-plugin."
            )

    else:
        # For other formats, use standard PIL
        try:
            return Image.open(content_stream)
        except Exception as e:
            raise Exception(f"Error opening image: {e}")
