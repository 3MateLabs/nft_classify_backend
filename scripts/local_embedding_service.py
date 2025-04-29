#!/usr/bin/env python3

"""
Simplified Local Embedding Service

This script provides a local alternative to the remote image embedding service
that was experiencing 504 Gateway Timeout errors.
"""

import os
import sys
import torch
import uvicorn
import requests
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel

# Create FastAPI app
app = FastAPI(
    title="Local NFT Image Embedding Service",
    description="Local service for generating embeddings from NFT images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ImageUrlRequest(BaseModel):
    img_url: str

# Load the model and processor
print("Loading model and processor...")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to("cpu")
print("Model and processor loaded successfully")

# Function to download image from URL
def download_image(url: str) -> Image.Image:
    """Download image from URL and convert to PIL Image"""
    try:
        # Add user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Print response headers for debugging
        print(f"Response headers for {url}: {response.headers}")
        
        # Save the image to a temporary file
        temp_file = BytesIO(response.content)
        
        # Try to open with PIL
        try:
            img = Image.open(temp_file)
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            return img
        except Exception as img_error:
            # If PIL fails, try to convert the image using another method
            print(f"PIL failed to open image: {str(img_error)}. Trying alternative method...")
            
            # For AVIF images, we need to use pillow_avif
            if 'image/avif' in response.headers.get('Content-Type', ''):
                try:
                    # Import pillow_avif to handle AVIF images
                    import pillow_avif
                    img = Image.open(BytesIO(response.content))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    return img
                except ImportError:
                    raise HTTPException(status_code=400, detail=f"AVIF image format not supported. Please install pillow_avif.")
                except Exception as avif_error:
                    raise HTTPException(status_code=400, detail=f"Error processing AVIF image: {str(avif_error)}")
            else:
                raise HTTPException(status_code=400, detail=f"Cannot identify image format: {str(img_error)}")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

# Function to generate embedding
def generate_embedding(image: Image.Image) -> List[float]:
    """Generate embedding for an image"""
    try:
        # Process through the model
        inputs = processor(image, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use mean pooling to get a single embedding vector
        pooler_output = torch.mean(outputs.last_hidden_state, dim=1)
        
        # Convert to list for JSON serialization
        embedding_list = pooler_output.detach().cpu().numpy().tolist()[0]
        
        return embedding_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Local NFT Image Embedding Service is running"}

# Embedding from URL endpoint
@app.post("/embed_from_url")
async def embed_from_url(request: ImageUrlRequest):
    try:
        # Download and process the image
        image = download_image(request.img_url)
        
        # Generate embedding
        embedding = generate_embedding(image)
        
        return {"embedding": embedding}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Main function to run the server
def main():
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
