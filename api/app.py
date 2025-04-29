#!/usr/bin/env python3

"""
FastAPI application for NFT image similarity search and NFT image downloading
"""

import os
import sys
import json
import uuid
import hashlib
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import (
    EMBEDDING_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
IMAGE_EMBEDDING_API_URL = "http://localhost:3001"
IMAGE_EMBEDDING_API_KEY = "45334ad61f254307a32"

# Qdrant configuration
QDRANT_API_URL = "https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io"
QDRANT_COLLECTION = "nft_embeddings"

# Path to NFT data CSV
NFT_DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data/embeddings/all_nft_data.csv")

# Create FastAPI app
app = FastAPI(
    title="NFT Classification API",
    description="API for NFT image similarity search and NFT data management",
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

# Pydantic models
class ImageSearchRequest(BaseModel):
    img_url: HttpUrl = Field(..., description="URL of the image to search for")
    top_k: int = Field(5, description="Number of similar images to return")
    threshold: float = Field(0.7, description="Similarity threshold (0-1)")

class NFTMetadata(BaseModel):
    object_id: str = Field(..., description="NFT object ID")
    collection_id: str = Field(..., description="NFT collection ID")
    name: str = Field(..., description="NFT name")
    image_url: str = Field(..., description="NFT image URL")
    similarity: float = Field(..., description="Similarity score (0-1)")

class ImageSearchResponse(BaseModel):
    query_image_url: str = Field(..., description="URL of the query image")
    similar_nfts: List[NFTMetadata] = Field(..., description="List of similar NFTs")
    
class CollectionDownloadRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection to download")
    limit: Optional[int] = Field(None, description="Maximum number of NFTs to download")

class CollectionDownloadResponse(BaseModel):
    collection_name: str = Field(..., description="Name of the collection")
    status: str = Field(..., description="Download status")
    message: str = Field(..., description="Status message")
    task_id: Optional[str] = Field(None, description="Background task ID")

# Helper functions
def generate_embedding_from_url(image_url: str) -> Optional[List[float]]:
    """Generate embedding for an image using the image embedding service"""
    url = f"{IMAGE_EMBEDDING_API_URL}/embed_from_url"
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": IMAGE_EMBEDDING_API_KEY
    }
    
    payload = {
        "img_url": image_url
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Error response for {image_url}: Status {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        if "embedding" in data:
            # Flatten the embedding if it's 2D
            embedding = data["embedding"]
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = [item for sublist in embedding for item in sublist]
            return embedding
        else:
            logger.error(f"No embedding found in response for {image_url}")
            return None
    except Exception as e:
        logger.error(f"Error generating embedding for {image_url}: {str(e)}")
        return None

def hex_to_uuid(hex_string: str) -> str:
    """Convert a hex string to a valid UUID"""
    # Remove '0x' prefix if present
    if hex_string.startswith('0x'):
        hex_string = hex_string[2:]
    
    # Generate a deterministic UUID based on the hex string
    # Use MD5 hash to ensure consistent length and format
    md5_hash = hashlib.md5(hex_string.encode()).hexdigest()
    
    # Format as UUID
    return str(uuid.UUID(md5_hash))

def search_similar_images(vector: List[float], top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Search for similar images in Qdrant"""
    url = f"{QDRANT_API_URL}/collections/{QDRANT_COLLECTION}/points/search"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }
    
    payload = {
        "vector": vector,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
        "score_threshold": threshold
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Error response from Qdrant: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        if "result" in data and len(data["result"]) > 0:
            return data["result"]
        else:
            logger.info("No similar images found")
            return []
    except Exception as e:
        logger.error(f"Error searching for similar images: {str(e)}")
        return []

def download_nft_images(collection_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Download NFT images for a specific collection"""
    try:
        # Load NFT data from CSV
        if not os.path.exists(NFT_DATA_CSV):
            logger.error(f"NFT data file {NFT_DATA_CSV} not found")
            return {"status": "error", "message": f"NFT data file not found"}
        
        nfts_df = pd.read_csv(NFT_DATA_CSV)
        
        # Check if 'nft_collection_name' column exists
        collection_column = None
        if 'nft_collection_name' in nfts_df.columns:
            collection_column = 'nft_collection_name'
        elif 'nftCollectionName' in nfts_df.columns:
            collection_column = 'nftCollectionName'
        
        if collection_column is None:
            logger.error("Could not find collection name column in CSV")
            return {"status": "error", "message": "Could not find collection name column in CSV"}
        
        # Filter by collection name
        collection_nfts = nfts_df[nfts_df[collection_column] == collection_name]
        
        if len(collection_nfts) == 0:
            logger.error(f"No NFTs found for collection '{collection_name}'")
            return {"status": "error", "message": f"No NFTs found for collection '{collection_name}'"}
        
        # Apply limit if specified
        if limit is not None:
            collection_nfts = collection_nfts.head(limit)
        
        # Convert DataFrame to list of dictionaries
        nfts = collection_nfts.to_dict('records')
        
        # Create download directory if it doesn't exist
        download_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   f"data/images/{collection_name}")
        os.makedirs(download_dir, exist_ok=True)
        
        # Download images
        success_count = 0
        failed_count = 0
        
        def download_image(nft):
            nonlocal success_count, failed_count
            
            object_id = nft.get("object_id")
            image_url = nft.get("image_url")
            
            if not image_url:
                logger.error(f"No image URL found for NFT {object_id}")
                failed_count += 1
                return
            
            # Create filename from object_id
            filename = f"{object_id}.jpg"
            filepath = os.path.join(download_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                logger.info(f"Image already exists for NFT {object_id}")
                success_count += 1
                return
            
            try:
                # Download image
                response = requests.get(image_url, stream=True)
                
                if response.status_code != 200:
                    logger.error(f"Error downloading image for NFT {object_id}: Status {response.status_code}")
                    failed_count += 1
                    return
                
                # Save image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                
                logger.info(f"Downloaded image for NFT {object_id}")
                success_count += 1
            except Exception as e:
                logger.error(f"Error downloading image for NFT {object_id}: {str(e)}")
                failed_count += 1
        
        # Download images in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(download_image, nfts)
        
        return {
            "status": "success",
            "message": f"Downloaded {success_count} images, failed {failed_count}",
            "collection_name": collection_name,
            "total_nfts": len(nfts),
            "success_count": success_count,
            "failed_count": failed_count
        }
    except Exception as e:
        logger.error(f"Error downloading NFT images: {str(e)}")
        return {"status": "error", "message": f"Error downloading NFT images: {str(e)}"}

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NFT Classification API"}

@app.post("/search", response_model=ImageSearchResponse)
async def search_image(request: ImageSearchRequest):
    """Search for similar NFT images"""
    # Generate embedding for the query image
    embedding = generate_embedding_from_url(str(request.img_url))
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="Failed to generate embedding for the query image")
    
    # Search for similar images
    similar_images = search_similar_images(embedding, request.top_k, request.threshold)
    
    if not similar_images:
        return {
            "query_image_url": str(request.img_url),
            "similar_nfts": []
        }
    
    # Format response
    similar_nfts = []
    for item in similar_images:
        payload = item.get("payload", {})
        
        # Skip if payload is missing required fields
        if not all(key in payload for key in ["object_id", "collection_id", "name", "image_url"]):
            continue
        
        similar_nfts.append({
            "object_id": payload["object_id"],
            "collection_id": payload["collection_id"],
            "name": payload["name"],
            "image_url": payload["image_url"],
            "similarity": item.get("score", 0.0)
        })
    
    return {
        "query_image_url": str(request.img_url),
        "similar_nfts": similar_nfts
    }

@app.post("/download", response_model=CollectionDownloadResponse)
async def download_collection(request: CollectionDownloadRequest, background_tasks: BackgroundTasks):
    """Download NFT images for a collection"""
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Start download in background
    background_tasks.add_task(download_nft_images, request.collection_name, request.limit)
    
    return {
        "collection_name": request.collection_name,
        "status": "started",
        "message": f"Started downloading NFT images for collection '{request.collection_name}'",
        "task_id": task_id
    }

@app.get("/collections")
async def list_collections():
    """List all available collections"""
    try:
        # Load NFT data from CSV
        if not os.path.exists(NFT_DATA_CSV):
            raise HTTPException(status_code=404, detail=f"NFT data file {NFT_DATA_CSV} not found")
        
        nfts_df = pd.read_csv(NFT_DATA_CSV)
        
        # Check if 'nft_collection_name' column exists
        collection_column = None
        if 'nft_collection_name' in nfts_df.columns:
            collection_column = 'nft_collection_name'
        elif 'nftCollectionName' in nfts_df.columns:
            collection_column = 'nftCollectionName'
        
        if collection_column is None:
            raise HTTPException(status_code=500, detail="Could not find collection name column in CSV")
        
        # Get unique collections
        collections = nfts_df[collection_column].unique().tolist()
        
        # Count NFTs per collection
        collection_counts = {}
        for collection in collections:
            collection_counts[collection] = len(nfts_df[nfts_df[collection_column] == collection])
        
        return {
            "collections": [
                {
                    "name": collection,
                    "nft_count": collection_counts[collection]
                }
                for collection in collections
            ]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
