# NFT Backend API Tests

This directory contains tests and test utilities for the NFT classification backend API.

## Files

- `__init__.py` - Makes the directory a proper Python package
- `example_urls.py` - Contains example URLs for different image formats and encodings
- `test_api.py` - Main test script for testing the API endpoints

## Running Tests

From the root directory, you can run the tests using the following commands:

```bash
# Test all endpoints with all formats
python -m test.test_api --mode all

# Test with a specific port (default is 8000)
python -m test.test_api --port 8001

# Test a specific format via URL
python -m test.test_api --mode url --format avif

# Test a custom URL
python -m test.test_api --mode url --image-url https://example.com/image.png

# Test file upload with a local image
python -m test.test_api --mode file --image-path /path/to/image.png

# Test base64 encoding with a local image
python -m test.test_api --mode base64 --image-path /path/to/image.png
```

## Available Test Formats

The example_urls.py file includes URLs for the following formats:

- `avif` - AVIF image format
- `png` - PNG image format
- `webp` - WebP image format
- `svg` - SVG vector image format
- `base64` - Base64-encoded PNG image 