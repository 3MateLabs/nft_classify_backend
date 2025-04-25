#!/usr/bin/env python
"""
Test script to verify the NFT classification backend API functionality.
Supports testing various image formats through different methods:
- Direct URL testing
- Base64-encoded image testing
- File upload testing

Usage:
    python test_api.py --mode url --format avif --port 8000
    python test_api.py --mode base64 --port 8001
    python test_api.py --mode file --image-path path/to/image.png --port 8000
    python test_api.py --mode all --format all
"""
import argparse
import requests
import base64
import json
import sys
import os
from PIL import Image
from io import BytesIO
import time
import logging

# Import example URLs for testing
from example_urls import EXAMPLE_URLS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APITester:
    """Class to handle API testing for NFT classification backend"""

    def __init__(self, base_url="http://localhost", port=8000):
        """
        Initialize the API tester with the target API endpoints

        Args:
            base_url: Base URL of the API server
            port: Port number the API server is running on
        """
        self.base_url = f"{base_url}:{port}"
        self.url_endpoint = f"{self.base_url}/embed_from_url"
        self.file_endpoint = f"{self.base_url}/embed_from_file"
        logger.info(f"Initialized API tester for {self.base_url}")

    def test_health(self):
        """Test the health check endpoint"""
        try:
            response = requests.get(self.base_url)
            logger.info(f"Health check status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                logger.info(f"API response: {result}")
                return True
            else:
                logger.error(f"Health check failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Health check request failed: {str(e)}")
            return False

    def test_url_endpoint(self, image_url):
        """
        Test the /embed_from_url endpoint with a URL

        Args:
            image_url: URL of the image to test

        Returns:
            dict: API response or None if failed
        """
        logger.info(f"Testing URL endpoint with image: {image_url[:100]}...")

        payload = {"img_url": image_url}

        try:
            start_time = time.time()
            response = requests.post(self.url_endpoint, json=payload)
            elapsed_time = time.time() - start_time

            logger.info(f"Status: {response.status_code}, Time: {elapsed_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                logger.info(f"Success! Embedding length: {len(embedding)}")
                logger.info(
                    f"First few values: {embedding[0][:5] if embedding and embedding[0] else []}"
                )
                return result
            else:
                logger.error(f"Error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def test_base64_url(self, base64_url):
        """
        Test the /embed_from_url endpoint with a pre-encoded base64 image URL

        Args:
            base64_url: Base64-encoded image URL

        Returns:
            dict: API response or None if failed
        """
        logger.info(f"Testing with base64 image URL (truncated): {base64_url[:50]}...")

        payload = {"img_url": base64_url}

        try:
            start_time = time.time()
            response = requests.post(self.url_endpoint, json=payload)
            elapsed_time = time.time() - start_time

            logger.info(f"Status: {response.status_code}, Time: {elapsed_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                logger.info(f"Success! Embedding length: {len(embedding)}")
                logger.info(
                    f"First few values: {embedding[0][:5] if embedding and embedding[0] else []}"
                )
                return result
            else:
                logger.error(f"Error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def test_base64_from_file(self, image_path):
        """
        Test the /embed_from_url endpoint with a base64-encoded image created from a file

        Args:
            image_path: Path to the image file to encode

        Returns:
            dict: API response or None if failed
        """
        logger.info(f"Testing base64 endpoint with image file: {image_path}")

        try:
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()

            # Get the format from file extension
            format_name = os.path.splitext(image_path)[1].lower().replace(".", "")
            if format_name == "jpg":
                format_name = "jpeg"

            # Encode to base64
            encoded_img = base64.b64encode(img_data).decode("utf-8")
            base64_url = f"data:image/{format_name};base64,{encoded_img}"

            # Use the existing method to test
            return self.test_base64_url(base64_url)
        except Exception as e:
            logger.error(f"Error preparing base64 image: {str(e)}")
            return None

    def test_file_endpoint(self, image_path):
        """
        Test the /embed_from_file endpoint with a file upload

        Args:
            image_path: Path to the image file to upload

        Returns:
            dict: API response or None if failed
        """
        logger.info(f"Testing file upload endpoint with image: {image_path}")

        try:
            with open(image_path, "rb") as img_file:
                # Get the format from file extension
                format_name = os.path.splitext(image_path)[1].lower().replace(".", "")
                if format_name == "jpg":
                    format_name = "jpeg"

                files = {
                    "file": (
                        os.path.basename(image_path),
                        img_file,
                        f"image/{format_name}",
                    )
                }

                start_time = time.time()
                response = requests.post(self.file_endpoint, files=files)
                elapsed_time = time.time() - start_time

            logger.info(f"Status: {response.status_code}, Time: {elapsed_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                logger.info(f"Success! Embedding length: {len(embedding)}")
                logger.info(
                    f"First few values: {embedding[0][:5] if embedding and embedding[0] else []}"
                )
                return result
            else:
                logger.error(f"Error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None


def main():
    """Main function to run the tests based on command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test the NFT classification backend API"
    )
    parser.add_argument(
        "--mode",
        choices=["url", "base64", "file", "all"],
        default="all",
        help="Test mode: url, base64, file, or all",
    )
    parser.add_argument(
        "--format",
        choices=["avif", "png", "webp", "svg", "base64", "all"],
        default="all",
        help="Image format to test (for URL mode)",
    )
    parser.add_argument(
        "--image-url", help="Custom image URL to test (overrides --format)"
    )
    parser.add_argument(
        "--image-path", help="Path to local image file (for base64 and file modes)"
    )
    parser.add_argument(
        "--base-url", default="http://localhost", help="Base URL of the API"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number the API is running on"
    )

    args = parser.parse_args()

    # Initialize API tester
    tester = APITester(args.base_url, args.port)

    # Check if API is running
    if not tester.test_health():
        logger.error(f"API at {tester.base_url} is not responding. Exiting.")
        sys.exit(1)

    # Run tests based on mode
    if args.mode in ["url", "all"]:
        logger.info("=== URL TESTING MODE ===")

        # Determine which URLs to test
        if args.image_url:
            # Test custom URL if provided
            url_to_test = args.image_url
            tester.test_url_endpoint(url_to_test)
        else:
            # Test predefined URLs based on format
            if args.format == "all":
                for fmt, url in EXAMPLE_URLS.items():
                    logger.info(f"\n--- Testing {fmt.upper()} format ---")
                    tester.test_url_endpoint(url)
            else:
                url_to_test = EXAMPLE_URLS.get(args.format)
                if url_to_test:
                    tester.test_url_endpoint(url_to_test)
                else:
                    logger.error(f"No example URL found for format: {args.format}")

    if args.mode in ["base64", "all"]:
        logger.info("\n=== BASE64 TESTING MODE ===")

        if args.format == "base64" or args.format == "all":
            # Test the predefined base64 URL
            logger.info("--- Testing predefined Base64 image ---")
            tester.test_base64_url(EXAMPLE_URLS["base64"])

        if args.image_path:
            logger.info(
                f"\n--- Testing Base64 encoding of local file: {args.image_path} ---"
            )
            tester.test_base64_from_file(args.image_path)
        elif args.mode == "base64" and args.format != "base64" and args.format != "all":
            logger.warning(
                "Base64 testing from local file requires --image-path argument"
            )

    if args.mode in ["file", "all"]:
        logger.info("\n=== FILE UPLOAD TESTING MODE ===")

        if args.image_path:
            tester.test_file_endpoint(args.image_path)
        else:
            logger.warning("File upload testing requires --image-path argument")

    logger.info("\nTest completed")


if __name__ == "__main__":
    main()
