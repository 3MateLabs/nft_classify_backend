#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import numpy as np
import os
import logging
import time
import json
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://image-embedding-service.3matelabs.com/embed_from_url")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "45334ad61f254307a32")
QDRANT_URL = os.getenv("QDRANT_URL", "https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io")
# Add local embedding URL as a fallback
LOCAL_EMBEDDING_URL = os.getenv("LOCAL_EMBEDDING_URL", "http://localhost:3001/embed_from_url")

# Initialize FastAPI app
app = FastAPI(
    title="NFT Image Search API",
    description="API for searching NFT collections by image similarity",
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

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0  # Increased timeout
    )
    logger.info("Connected to Qdrant successfully")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

# Collection mapping
COLLECTION_MAPPING = {
    "doubleup": "0x862810efecf0296db2e9df3e075a7af8034ba374e73ff1098e88cc4bb7c15437::doubleup_citizens::DoubleUpCitizen",
    "aeon": "0x5d0c9a06d3351e6714e7935a6efd3aed24b71994a0d709e1cd2d692b45d0cad3::aeon::Aeon",
    "prime_machine": "0x0f4e5cad3b76d8c2d4f3b532d9f1a78f77cc89b5e2b1dbd9fb0533fc3b8a1e66::prime_machine::PrimeMachine",
    "rootlets": "0x5a7eca70b90bd321b1a475200c6a68a6e66f9d3f6b32deb93bb3e436c7f89fb5::rootlets::Rootlet",
    "killaclub": "0x9d0cdb04a16a5b7d86c8b3ad400ac27c56c23ed8a7a77ec9afc39dcfbfb6aff3::killa_club::KillaClub",
    "kumo": "0xd3d2c5edc3d8b3aafbdd50a2b9539d0ed0d87af1d3b96b61e2d33c4d6b5ebd6a::kumo::Kumo"
}

# Response models
class SearchResult(BaseModel):
    collection: str = Field(..., description="NFT collection name")
    collection_object_type: str = Field(..., description="NFT collection object type")
    object_id: str = Field(..., description="NFT object ID")
    similarity_score: float = Field(..., description="Similarity score (0-1)")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Search results")
    query_embedding_size: int = Field(..., description="Size of the query embedding vector")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

# Helper functions
def generate_embedding_from_url(image_url: str) -> np.ndarray:
    """Generate embedding for an image URL using the embedding service"""
    try:
        headers = {"x-api-key": EMBEDDING_API_KEY}
        payload = {"img_url": image_url}
        
        logger.info(f"Generating embedding for {image_url}")
        response = requests.post(
            EMBEDDING_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Error from embedding service: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {response.text}"
            )
        
        embedding = np.array(response.json()["embedding"])
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Try local embedding service as fallback
        try:
            logger.info(f"Trying local embedding service as fallback for {image_url}")
            headers = {"x-api-key": EMBEDDING_API_KEY}
            payload = {"img_url": image_url}
            
            response = requests.post(
                LOCAL_EMBEDDING_URL,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Error from local embedding service: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate embedding from both remote and local services"
                )
            
            embedding = np.array(response.json()["embedding"])
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as local_e:
            logger.error(f"Error generating embedding from local service: {local_e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {str(e)}. Local fallback also failed: {str(local_e)}"
            )

def generate_embedding_from_data_url(data_url: str) -> np.ndarray:
    """Generate embedding for a data URL using the embedding service"""
    try:
        # Extract the base64 data
        if not data_url.startswith('data:image/'):
            raise ValueError("Not a valid image data URL")
            
        # Extract the image format and base64 data
        format_end = data_url.find(";base64,")
        if format_end == -1:
            raise ValueError("Not a valid base64 image data URL")
            
        base64_data = data_url[format_end + 8:]  # Skip ";base64,"
        
        # Try to use the remote embedding service first
        try:
            # Use a different endpoint for base64 data
            # If the service doesn't support base64 directly, we'll catch the exception and try local
            headers = {"x-api-key": EMBEDDING_API_KEY}
            payload = {"img_base64": base64_data}
            
            logger.info("Generating embedding from data URL using remote service")
            response = requests.post(
                EMBEDDING_API_URL.replace("embed_from_url", "embed_from_base64"),
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Remote service error: {response.status_code} - {response.text}")
            
            embedding = np.array(response.json()["embedding"])
        except Exception as e:
            logger.warning(f"Remote embedding service failed for data URL: {e}")
            # Try local embedding service as fallback
            logger.info("Trying local embedding service for data URL")
            headers = {"x-api-key": EMBEDDING_API_KEY}
            payload = {"img_base64": base64_data}
            
            response = requests.post(
                LOCAL_EMBEDDING_URL.replace("embed_from_url", "embed_from_base64"),
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Local service error: {response.status_code} - {response.text}")
            
            embedding = np.array(response.json()["embedding"])
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding from data URL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding from data URL: {str(e)}"
        )

def search_similar_nfts(embedding: np.ndarray, threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for similar NFTs in Qdrant"""
    try:
        if qdrant_client is None:
            logger.error("Qdrant client not initialized")
            return []
            
        # Search for similar NFTs in Qdrant
        search_result = qdrant_client.search(
            collection_name="nft_embeddings",
            query_vector=embedding.tolist(),
            limit=limit,
            score_threshold=threshold
        )
        
        results = []
        for scored_point in search_result:
            # Extract collection and object ID from payload
            collection = scored_point.payload.get("collection") or scored_point.payload.get("collection_type")
            object_id = scored_point.payload.get("object_id")
            
            # Map collection name to object type
            object_type = COLLECTION_MAPPING.get(collection, collection)
            
            results.append({
                "collection": collection,
                "collection_object_type": object_type,
                "object_id": object_id,
                "similarity_score": scored_point.score
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching for similar NFTs: {e}")
        return []

# API endpoints
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "NFT Image Search API",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status = "healthy"
    qdrant_status = "disconnected"
    embedding_status = "unknown"
    
    # Check Qdrant connection
    try:
        if qdrant_client is not None:
            # Check if Qdrant is responsive
            qdrant_client.get_collections()
            qdrant_status = "connected"
        else:
            qdrant_status = "not initialized"
            status = "degraded"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        qdrant_status = f"error: {str(e)}"
        status = "degraded"
    
    # Check embedding service
    try:
        # Simple health check request to the embedding service
        response = requests.get(
            EMBEDDING_API_URL.split("/embed_from_url")[0] + "/health",
            timeout=5
        )
        embedding_status = "healthy" if response.status_code == 200 else f"error: {response.status_code}"
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        embedding_status = f"error: {str(e)}"
        status = "degraded"
    
    return {
        "status": status,
        "services": {
            "qdrant": qdrant_status,
            "embedding": embedding_status
        }
    }

@app.get("/search", response_model=SearchResponse)
async def search_image(img_url: str = Query(..., description="URL of the image to search for"),
                      threshold: float = Query(0.7, description="Minimum similarity threshold (0-1)"),
                      limit: int = Query(5, description="Maximum number of results to return")):
    """Search for similar NFTs by image URL"""
    start_time = time.time()
    
    try:
        # Check if it's a data URL
        if img_url.startswith('data:image/'):
            # Generate embedding from data URL
            embedding = generate_embedding_from_data_url(img_url)
        else:
            # Generate embedding from image URL
            embedding = generate_embedding_from_url(img_url)
        
        # Search for similar NFTs
        results = search_similar_nfts(
            embedding=embedding,
            threshold=threshold,
            limit=limit
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "results": results,
            "query_embedding_size": len(embedding),
            "processing_time_ms": processing_time
        }
    except Exception as e:
        logger.error(f"Error in search_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
def get_collections():
    """Get available NFT collections"""
    collections = []
    
    for name, object_type in COLLECTION_MAPPING.items():
        collections.append({
            "name": name,
            "object_type": object_type
        })
    
    return {
        "collections": collections
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("support_api_no_mocks:app", host="0.0.0.0", port=8000, reload=True)
