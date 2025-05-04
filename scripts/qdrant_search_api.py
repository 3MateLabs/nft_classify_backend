#!/usr/bin/env python3
"""
Standalone FastAPI service for searching NFT embeddings in Qdrant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import time
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO
import os
import sys

# Add the project root to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from api.config import QDRANT_API_URL, QDRANT_API_KEY
except ImportError:
    # Fallback if import fails
    QDRANT_API_URL = "https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io"

# Collection name in Qdrant
COLLECTION_NAME = "nft_embeddings"

# Model configuration - using local embedding service
try:
    from scripts.local_embedding_service import get_embedding_from_image
except ImportError:
    # Fallback function if import fails
    def get_embedding_from_image(image):
        """Fallback function that uses a remote embedding service"""
        import base64
        from io import BytesIO
        
        try:
            print("Converting image to base64...")
            # Convert image to base64
            buffered = BytesIO()
            
            # Convert to RGB mode if necessary (JPEG doesn't support alpha channel)
            if image.mode == 'RGBA':
                print("Converting RGBA image to RGB mode")
                image = image.convert('RGB')
                
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"Image converted to base64, length: {len(img_str)}")
            
            # Call remote embedding service using embed_from_url endpoint with base64 data
            url = "http://localhost:3001/embed_from_url"
            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": "45334ad61f254307a32"
            }
            # Format as a data URL
            base64_url = f"data:image/jpeg;base64,{img_str}"
            payload = {"img_url": base64_url}
            
            print(f"Sending request to embedding service at: {url}")
            response = requests.post(url, json=payload, headers=headers)
            print(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            if "embedding" in data:
                print("Successfully received embedding from service")
                return data["embedding"]
            else:
                print(f"Embedding not found in response: {data}")
                return None
        except Exception as e:
            print(f"Error in get_embedding_from_image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Request and response models
class ImageUrlRequest(BaseModel):
    img_url: str
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.7  # Similarity threshold

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int
    query_time_ms: float

# Initialize FastAPI app
app = FastAPI(
    title="NFT Vector Search API",
    description="Search for similar NFTs using vector embeddings in Qdrant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
async def generate_embedding_from_url(image_url: str) -> List[float]:
    """
    Process an image from URL and generate embedding
    """
    try:
        print(f"Downloading image from URL: {image_url}")
        # Download image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        print(f"Image downloaded successfully, content type: {response.headers.get('Content-Type')}")
        # Convert to PIL Image
        image = Image.open(BytesIO(response.content))
        print(f"Image opened successfully, format: {image.format}, mode: {image.mode}, size: {image.size}")
        
        # Generate embedding using our model service
        print("Generating embedding...")
        embedding = get_embedding_from_image(image)
        
        if embedding:
            print(f"Embedding generated successfully, length: {len(embedding)}")
        else:
            print("Failed to generate embedding")
        
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def search_qdrant(embedding: List[float], limit: int = 10, threshold: float = 0.7):
    """Search for similar NFTs in Qdrant"""
    try:
        # Prepare the search request
        search_url = f"{QDRANT_API_URL}/collections/{COLLECTION_NAME}/points/search"
        headers = {
            "Content-Type": "application/json",
            "api-key": QDRANT_API_KEY
        }
        payload = {
            "vector": embedding,
            "limit": limit,
            "with_payload": True,  # Include all payload data
            "score_threshold": threshold  # Only return results above this similarity threshold
        }
        
        # Make the request to Qdrant
        response = requests.post(search_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Process results
        data = response.json()
        results = []
        
        if "result" in data:
            for item in data["result"]:
                # Extract all available metadata from the payload
                payload = item.get("payload", {})
                result = {
                    "score": item.get("score", 0),
                    "object_id": payload.get("object_id", ""),
                    "collection_id": payload.get("collection_id", ""),
                    "name": payload.get("name", ""),
                    "image_url": payload.get("image_url", ""),
                    "nft_type": payload.get("nft_type", ""),
                    "nft_collection_name": payload.get("nft_collection_name", ""),
                    "creator": payload.get("creator", ""),
                    "description": payload.get("description", ""),
                    "created_time": payload.get("created_time", "")
                }
                results.append(result)
        
        return results
    except Exception as e:
        print(f"Error searching Qdrant: {str(e)}")
        return []

# API Endpoints
@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "NFT Vector Search API is running"}

@app.post("/search", response_model=SearchResponse)
async def search(request: ImageUrlRequest):
    """
    Search for similar NFTs based on image URL
    """
    try:
        # Generate embedding from image URL
        embedding = await generate_embedding_from_url(request.img_url)
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding from image")
        
        # Start timer for query
        start_time = time.time()
        
        # Search Qdrant
        results = search_qdrant(embedding, request.limit, request.threshold)
        
        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000
        
        # Sort results by similarity score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "results": results,
            "count": len(results),
            "query_time_ms": query_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/collections")
async def get_collections():
    """Get all collections in Qdrant"""
    url = f"{QDRANT_API_URL}/collections"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collections: {str(e)}")

@app.get("/collection/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get information about a specific collection"""
    url = f"{QDRANT_API_URL}/collections/{collection_name}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collection info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
