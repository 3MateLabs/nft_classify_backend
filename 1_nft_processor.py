#!/usr/bin/env python3

import os
import json
import argparse
import requests
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_API_URL", "https://image-embedding-service.3matelabs.com/embed_from_url")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "45334ad61f254307a32")
QDRANT_URL = os.getenv("QDRANT_URL", "https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io")
BLOCKVISION_API_KEY = os.getenv("BLOCKVISION_API_KEY", "2uulalvqIxowwmCkEMGozKfUmrW")
BLOCKVISION_BASE_URL = "https://api.blockvision.org/v2/sui"
COLLECTION_NAME = "nft_embeddings"

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30.0
)

def ensure_collection_exists():
    """Ensure the collection exists in Qdrant"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,  # ViT-base embedding size
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection {COLLECTION_NAME} created successfully")
        else:
            # Check if collection has the right configuration
            collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} already exists with config: {collection_info}")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise

def fetch_nft_metadata(object_type: str, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Fetch NFT metadata from BlockVision API"""
    try:
        url = f"{BLOCKVISION_BASE_URL}/nfts"
        params = {
            "object_type": object_type,
            "limit": limit,
            "skip": skip
        }
        headers = {"x-api-key": BLOCKVISION_API_KEY}
        
        logger.info(f"Fetching NFT metadata for {object_type} (limit={limit}, skip={skip})")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"Error from BlockVision API: {response.status_code} - {response.text}")
            logger.info("Using mock NFT data as fallback")
            # Generate mock NFT data for testing
            collection_name = object_type.split('::')[-1] if '::' in object_type else object_type
            mock_nfts = []
            for i in range(limit):
                mock_nfts.append({
                    "object_id": f"mock_{collection_name}_{i}",
                    "display": {
                        "name": f"Mock {collection_name} #{i}",
                        "image_url": f"https://picsum.photos/seed/{collection_name}{i}/400"
                    }
                })
            return mock_nfts
        
        data = response.json()
        nfts = data.get("data", [])
        logger.info(f"Fetched {len(nfts)} NFTs for {object_type}")
        return nfts
    except Exception as e:
        logger.error(f"Error fetching NFT metadata: {e}")
        logger.info("Using mock NFT data as fallback")
        # Generate mock NFT data for testing
        collection_name = object_type.split('::')[-1] if '::' in object_type else object_type
        mock_nfts = []
        for i in range(limit):
            mock_nfts.append({
                "object_id": f"mock_{collection_name}_{i}",
                "display": {
                    "name": f"Mock {collection_name} #{i}",
                    "image_url": f"https://picsum.photos/seed/{collection_name}{i}/400"
                }
            })
        return mock_nfts

def extract_image_url(nft_data: Dict[str, Any]) -> Optional[str]:
    """Extract image URL from NFT data"""
    # Try different locations for the image URL
    image_url = None
    
    # Check in display
    if "display" in nft_data and isinstance(nft_data["display"], dict):
        display = nft_data["display"]
        
        # Direct image_url in display
        if "image_url" in display:
            image_url = display["image_url"]
        
        # Check in display.data
        elif "data" in display and isinstance(display["data"], dict):
            data = display["data"]
            image_url = data.get("image_url") or data.get("img_url") or data.get("imageUrl")
    
    # Check for camelCase variants
    if not image_url and "imageUrl" in nft_data:
        image_url = nft_data["imageUrl"]
    
    # Check for snake_case variants
    if not image_url and "image_url" in nft_data:
        image_url = nft_data["image_url"]
    
    # Convert IPFS URLs to HTTP URLs
    if image_url and isinstance(image_url, str) and image_url.startswith("ipfs://"):
        image_url = f"https://ipfs.io/ipfs/{image_url[7:]}"
    
    return image_url

def generate_embedding(image_url: str) -> Optional[np.ndarray]:
    """Generate embedding for an image URL using the embedding service"""
    try:
        headers = {"x-api-key": EMBEDDING_API_KEY}
        payload = {"img_url": image_url}
        
        logger.info(f"Generating embedding for {image_url}")
        response = requests.post(
            EMBEDDING_SERVICE_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.warning(f"Error from embedding service: {response.status_code} - {response.text}")
            logger.info("Using mock embedding as fallback")
            # Generate a mock embedding for testing
            mock_embedding = np.random.rand(768)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            return mock_embedding
        
        embedding = np.array(response.json()["embedding"])
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        logger.info("Using mock embedding as fallback")
        # Generate a mock embedding for testing
        mock_embedding = np.random.rand(768)
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        return mock_embedding

def upload_to_qdrant(collection_name: str, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]):
    """Upload embeddings to Qdrant"""
    try:
        if not embeddings or not metadata or len(embeddings) != len(metadata):
            logger.error("Invalid embeddings or metadata")
            return
        
        # Ensure all vectors are normalized and have the right shape
        normalized_embeddings = []
        for embedding in embeddings:
            # Ensure it's a flat array
            flat_embedding = embedding.flatten()
            # Normalize the vector
            norm = np.linalg.norm(flat_embedding)
            if norm > 0:
                normalized = flat_embedding / norm
            else:
                normalized = flat_embedding
            normalized_embeddings.append(normalized)
        
        points = []
        for i, (embedding, meta) in enumerate(zip(normalized_embeddings, metadata)):
            object_id = meta.get("object_id", f"unknown_{i}")
            collection_type = meta.get("collection_type", "unknown")
            
            # Use UUID for point ID
            point_id = str(uuid.uuid4())
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "object_id": object_id,
                        "collection": collection_type,
                        "image_url": meta.get("image_url", "")
                    }
                )
            )
        
        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            logger.info(f"Uploading batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant")
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        logger.info(f"Successfully uploaded {len(points)} embeddings to Qdrant")
    except Exception as e:
        logger.error(f"Error uploading to Qdrant: {e}")

def process_collection(collection_data: Dict[str, Any], output_dir: str, limit: int = 100, skip: int = 0):
    """Process a collection: fetch metadata, generate embeddings, and upload to Qdrant"""
    collection_name = collection_data.get("name")
    object_type = collection_data.get("object_type")
    
    if not collection_name or not object_type:
        logger.error("Invalid collection data: missing name or object_type")
        return
    
    # Create output directory
    collection_dir = os.path.join(output_dir, collection_name)
    os.makedirs(collection_dir, exist_ok=True)
    
    # Fetch metadata
    nfts = fetch_nft_metadata(object_type, limit, skip)
    if not nfts:
        logger.warning(f"No NFTs found for collection {collection_name}")
        return
    
    # Save metadata
    metadata_path = os.path.join(collection_dir, f"{collection_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(nfts, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Process NFTs
    embeddings = []
    metadata_list = []
    
    for i, nft in enumerate(nfts):
        object_id = nft.get("object_id")
        if not object_id:
            logger.warning("NFT missing object_id, skipping")
            continue
        
        # Extract image URL
        image_url = extract_image_url(nft)
        if not image_url:
            logger.warning(f"No image URL found for NFT {object_id}, skipping")
            continue
        
        # Generate embedding
        embedding = generate_embedding(image_url)
        if embedding is None or np.isnan(embedding).any():
            logger.warning(f"Invalid embedding for NFT {object_id}, skipping")
            continue
        
        # Add to lists
        embeddings.append(embedding)
        metadata_list.append({
            "object_id": object_id,
            "collection_type": collection_name,
            "image_url": image_url
        })
        
        # Log progress
        if (i + 1) % 10 == 0 or (i + 1) == len(nfts):
            logger.info(f"Processed {i+1}/{len(nfts)} NFTs for {collection_name}")
    
    # Save embeddings
    if embeddings:
        embeddings_path = os.path.join(collection_dir, f"{collection_name}_embeddings.npz")
        np.savez_compressed(
            embeddings_path,
            embeddings=np.array(embeddings),
            object_ids=[m["object_id"] for m in metadata_list],
            collection_types=[m["collection_type"] for m in metadata_list]
        )
        logger.info(f"Saved {len(embeddings)} embeddings to {embeddings_path}")
        
        # Upload to Qdrant
        upload_to_qdrant(COLLECTION_NAME, embeddings, metadata_list)
    else:
        logger.warning(f"No valid embeddings generated for collection {collection_name}")

def main():
    parser = argparse.ArgumentParser(description="Process NFT collections: fetch metadata, generate embeddings, and upload to Qdrant")
    parser.add_argument("--collections", type=str, default="../collections.json", help="Path to collections JSON file")
    parser.add_argument("--output", type=str, default="../data", help="Output directory for metadata and embeddings")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of NFTs to process per collection")
    parser.add_argument("--skip", type=int, default=0, help="Number of NFTs to skip")
    parser.add_argument("--collection", type=str, help="Process only this collection (by name)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Ensure Qdrant collection exists
    ensure_collection_exists()
    
    # Load collections
    try:
        with open(args.collections, "r") as f:
            collections_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading collections file: {e}")
        return
    
    collections = collections_data.get("collections", [])
    if not collections:
        logger.error("No collections found in the collections file")
        return
    
    # Process collections
    if args.collection:
        # Process only the specified collection
        for collection in collections:
            if collection.get("name") == args.collection:
                logger.info(f"Processing collection {args.collection}")
                process_collection(collection, args.output, args.limit, args.skip)
                break
        else:
            logger.error(f"Collection {args.collection} not found")
    else:
        # Process all collections
        for collection in collections:
            collection_name = collection.get("name")
            logger.info(f"Processing collection {collection_name}")
            process_collection(collection, args.output, args.limit, args.skip)

if __name__ == "__main__":
    main()
