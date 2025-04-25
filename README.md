# NFT Classification Backend

A FastAPI backend service for NFT image classification and embedding generation.

## Features

- Image embedding generation from URLs, base64-encoded data, and file uploads
- Support for multiple image formats:
  - PNG
  - JPG/JPEG
  - WEBP
  - AVIF (requires pillow-avif-plugin)
  - SVG (converted using CairoSVG)
- Robust error handling with detailed traceback information
- Modular code organization

## Project Structure

```
api/
├── __init__.py
├── config.py                # Application-wide configuration
├── main.py                  # FastAPI application entry point
├── models/                  # Pydantic data models
│   ├── __init__.py
│   └── request_models.py
├── handlers/                # API endpoint handlers
│   ├── __init__.py
│   └── embedding_handlers.py
├── services/                # Business logic services
│   ├── __init__.py
│   ├── image_service.py     # Image processing service
│   └── model_service.py     # ML model service
└── utils/                   # Utility functions
    ├── __init__.py
    ├── format_handlers.py   # Image format-specific handlers
    └── image_utils.py       # General image utilities
```

## Dependencies

All required dependencies are listed in `requirements.txt`. The key packages include:
- `Pillow`: Core image processing
- `pillow-avif-plugin`: AVIF image support
- `cairosvg`: SVG to PNG conversion
- `fastapi` and `uvicorn`: API framework and server
- `transformers` and `torch`: ML model inference

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nft_classify_backend.git
   cd nft_classify_backend
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the FastAPI application with uvicorn:

```bash
uvicorn api.main:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

- `/` - GET: Health check endpoint
- `/embed_from_url` - POST: Generate embeddings from an image URL (supports regular URLs and base64-encoded data)
- `/embed_from_file` - POST: Generate embeddings from an uploaded image file

## Usage Examples

### Using the URL Endpoint

```python
import requests
import json

url = "http://localhost:8000/embed_from_url"
payload = {
    "img_url": "https://example.com/image.png"
}
response = requests.post(url, json=payload)
print(response.json())
```

### Using Base64-encoded Images

```python
import requests
import base64
import json

# Convert an image to base64
with open("image.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

base64_url = f"data:image/png;base64,{encoded_string}"

url = "http://localhost:8000/embed_from_url"
payload = {
    "img_url": base64_url
}
response = requests.post(url, json=payload)
print(response.json())
```

### Using the File Upload Endpoint

```python
import requests

url = "http://localhost:8000/embed_from_file"
files = {
    "file": ("image.png", open("image.png", "rb"), "image/png")
}
response = requests.post(url, files=files)
print(response.json())
```

## Testing

Use the provided test script to verify API functionality:

```bash
# Test with a URL
python test_api.py --mode url --image-url https://example.com/image.png

# Test with a local file (base64 and file upload)
python test_api.py --mode all --image-path path/to/image.png
```

## Debugging with VS Code

To debug this application using VS Code:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Press F5 in VS Code to start debugging
   - This will launch the FastAPI application with uvicorn
   - Breakpoints can be set in the code
   - The API will be available at http://localhost:8000

3. Access the API in your browser or via tools like Postman/curl
