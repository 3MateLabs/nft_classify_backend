#!/bin/bash

# Script to prepare the NFT Classifier repository for GitHub upload
# This script organizes the file structure and removes sensitive data

echo "Preparing NFT Classifier repository for GitHub..."

# Create necessary directories if they don't exist
mkdir -p api/services
mkdir -p data/embeddings
mkdir -p data/images

# Create empty directories with .gitkeep
touch data/embeddings/.gitkeep
touch data/images/.gitkeep

# Ensure scripts directory exists
mkdir -p scripts

# Move config.py if it doesn't exist in api directory
if [ ! -f api/config.py ]; then
  echo "Creating api/config.py..."
  cat > api/config.py << 'EOF'
#!/usr/bin/env python3

"""
Configuration settings for the NFT Classification API
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = os.path.join(BASE_DIR, "data/embeddings")
IMAGES_DIR = os.path.join(BASE_DIR, "data/images")

# Create directories if they don't exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Qdrant configuration
QDRANT_API_URL = os.getenv("QDRANT_API_URL", "https://your-qdrant-instance.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your_qdrant_api_key")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nft_embeddings")

# Image Embedding Service
IMAGE_EMBEDDING_API_URL = os.getenv("IMAGE_EMBEDDING_API_URL", "http://localhost:3001")
IMAGE_EMBEDDING_API_KEY = os.getenv("IMAGE_EMBEDDING_API_KEY", "your_embedding_api_key")

# Data Paths
NFT_DATA_CSV = os.getenv("NFT_DATA_CSV", os.path.join(BASE_DIR, "data/embeddings/all_nft_data.csv"))
EOF
fi

# Create __init__.py files
touch api/__init__.py
touch api/services/__init__.py

# Create models.py if it doesn't exist
if [ ! -f api/models.py ]; then
  echo "Creating api/models.py..."
  cat > api/models.py << 'EOF'
#!/usr/bin/env python3

"""
Pydantic models for the NFT Classification API
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

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
EOF
fi

# Create services files
if [ ! -f api/services/embedding_service.py ]; then
  echo "Creating api/services/embedding_service.py..."
  cat > api/services/embedding_service.py << 'EOF'
#!/usr/bin/env python3

"""
Client for the image embedding service
"""

import requests
import logging
from typing import List, Optional

from api.config import IMAGE_EMBEDDING_API_URL, IMAGE_EMBEDDING_API_KEY

logger = logging.getLogger(__name__)

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
EOF
fi

if [ ! -f api/services/qdrant_service.py ]; then
  echo "Creating api/services/qdrant_service.py..."
  cat > api/services/qdrant_service.py << 'EOF'
#!/usr/bin/env python3

"""
Client for the Qdrant vector database
"""

import uuid
import hashlib
import requests
import logging
from typing import List, Dict, Any, Optional

from api.config import QDRANT_API_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

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
EOF
fi

# Make script executable
chmod +x api/start_api.sh

echo "Repository preparation complete!"
echo "You can now upload to GitHub with the following structure:"
echo ""
echo "nft-classifier/"
echo "├── api/                  # FastAPI application"
echo "│   ├── app.py            # Main application"
echo "│   ├── config.py         # Configuration"
echo "│   ├── models.py         # Pydantic models"
echo "│   ├── services/         # Service clients"
echo "│   └── start_api.sh      # API startup script"
echo "├── scripts/              # Utility scripts"
echo "├── data/                 # Data directory (mostly gitignored)"
echo "├── .gitignore            # Git ignore file"
echo "├── README.md             # Project documentation"
echo "└── example.env           # Example environment variables"
echo ""
echo "Next steps:"
echo "1. Initialize Git repository: git init"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'Initial commit'"
echo "4. Create GitHub repository and push"
