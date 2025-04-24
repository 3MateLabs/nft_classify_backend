#!/usr/bin/env python3

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
import requests
import os
from io import BytesIO
import time
from fake_useragent import UserAgent
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import base64
from pydantic import BaseModel

API_KEY = "45334ad61f254307a32"

# Initialize the model and processor
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to("cpu")

# Create FastAPI app
app = FastAPI(title="Image Embedding Service")

@app.get("/health")
async def health():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

# API key authentication dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    return x_api_key

# Define the request models
class ImageUrlRequest(BaseModel):
    img_url: str

class ImageBase64Request(BaseModel):
    img_base64: str

def infer(image_url_or_obj):
    """
    Process an image from a URL or a PIL Image object and return embeddings.

    Args:
        image_url_or_obj: Either a URL string or a PIL Image object
    """
    if isinstance(image_url_or_obj, str):
        # It's a URL, need to download
        try:
            # Download the image from URL with proper headers
            # Use fake-useragent to generate random user agent
            ua = UserAgent()
            headers = {"User-Agent": ua.random}

            # Add retry mechanism with backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        image_url_or_obj, headers=headers, timeout=10
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        # Wait before retrying (exponential backoff)
                        time.sleep(2**attempt)
                    else:
                        raise e

            # Process the image directly from memory instead of saving to disk
            image = Image.open(BytesIO(response.content)).convert("RGB")

        except Exception as e:
            raise Exception(f"Error downloading or processing image from URL: {e}")
    else:
        # Assume it's already a PIL Image
        image = image_url_or_obj

    # Process through the model
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    pooler_output = torch.mean(outputs.last_hidden_state, dim=1)

    return pooler_output

@app.post("/embed_from_url")
async def embed_from_url(request: ImageUrlRequest, api_key: str = Depends(verify_api_key)):
    """
    Endpoint to get embeddings from an image URL
    """
    try:
        embedding = infer(request.img_url)
        # Convert to list for JSON serialization
        embedding_list = embedding.detach().cpu().numpy().tolist()[0]  # Get the first (and only) embedding
        return JSONResponse(content={"embedding": embedding_list})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/embed_from_base64")
async def embed_from_base64(request: ImageBase64Request, api_key: str = Depends(verify_api_key)):
    """
    Endpoint to get embeddings from a base64 encoded image
    """
    try:
        # Decode base64 string to image
        image_data = base64.b64decode(request.img_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        embedding = infer(image)
        # Convert to list for JSON serialization
        embedding_list = embedding.detach().cpu().numpy().tolist()[0]  # Get the first (and only) embedding
        return JSONResponse(content={"embedding": embedding_list})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/embed_from_file")
async def embed_from_file(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    Endpoint to get embeddings from an uploaded image file
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        embedding = infer(image)
        # Convert to list for JSON serialization
        embedding_list = embedding.detach().cpu().numpy().tolist()[0]  # Get the first (and only) embedding
        return JSONResponse(content={"embedding": embedding_list})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)
