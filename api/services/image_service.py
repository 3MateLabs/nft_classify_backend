"""
Image service module
Handles image downloading, processing, and preparation for model inference
"""

import time
import requests
from io import BytesIO
from PIL import Image
from typing import Union, BinaryIO
from fake_useragent import UserAgent

from api.config import (
    REQUEST_TIMEOUT,
    REQUEST_MAX_RETRIES,
    REQUEST_BACKOFF_FACTOR,
    logger,
)
from api.utils.image_utils import (
    is_base64_image_url,
    decode_base64_image_url,
    convert_to_rgb,
    detect_image_format,
)
from api.utils.format_handlers import open_image_with_format_specific_handling


def download_image(url: str) -> BytesIO:
    """
    Download an image from a URL with proper headers and retry logic

    Args:
        url: URL of the image to download

    Returns:
        BytesIO containing the image data
    """
    # Use fake-useragent to generate random user agent
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    # Add retry mechanism with backoff
    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            if attempt < REQUEST_MAX_RETRIES - 1:
                # Wait before retrying (exponential backoff)
                wait_time = REQUEST_BACKOFF_FACTOR**attempt
                logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"Failed to download image after {REQUEST_MAX_RETRIES} attempts: {str(e)}"
                )


def process_image_from_url(image_url: str) -> Image.Image:
    """
    Process an image from a URL (regular HTTP URL or base64 data URL)

    Args:
        image_url: URL of the image or base64 data URL

    Returns:
        PIL Image object ready for model processing
    """
    try:
        # Check if it's a base64 image URL
        if is_base64_image_url(image_url):
            try:
                # Decode the base64 URL
                image_data, format_hint = decode_base64_image_url(image_url)

                # Process the image data
                content_stream = BytesIO(image_data)
                image = open_image_with_format_specific_handling(
                    content_stream, format_hint
                )

                # Convert to RGB if needed
                return convert_to_rgb(image)
            except Exception as e:
                raise Exception(f"Error processing base64 image: {e}")
        else:
            # Regular URL - download the image
            content_stream = download_image(image_url)

            # Get content bytes for format detection
            content_bytes = content_stream.getvalue()
            format_hint = detect_image_format(content_bytes[:16], url=image_url)
            logger.info(f"Detected format: {format_hint}")

            # Process the image with format-specific handling
            content_stream.seek(0)  # Reset stream position

            try:
                image = open_image_with_format_specific_handling(
                    content_stream, format_hint
                )
                # Convert to RGB if needed
                return convert_to_rgb(image)
            except Exception as e:
                raise Exception(f"Error processing image from URL: {e}")
    except Exception as e:
        raise Exception(f"Error processing image URL: {e}")


def process_image_from_bytes(image_bytes: bytes, filename: str = None) -> Image.Image:
    """
    Process an image from raw bytes

    Args:
        image_bytes: Raw image bytes
        filename: Optional filename for format detection

    Returns:
        PIL Image object ready for model processing
    """
    try:
        # Create BytesIO from bytes
        content_stream = BytesIO(image_bytes)

        # Detect format
        format_hint = detect_image_format(image_bytes[:16], filename=filename)
        logger.info(f"Detected format: {format_hint}")

        # Process the image with format-specific handling
        image = open_image_with_format_specific_handling(content_stream, format_hint)

        # Convert to RGB if needed
        return convert_to_rgb(image)
    except Exception as e:
        raise Exception(f"Error processing image data: {e}")
