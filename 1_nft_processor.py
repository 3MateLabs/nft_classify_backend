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

# Initialize Qdrant client with increased timeout and retry settings
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0  # Increase timeout to 60 seconds
    )
    logger.info("Successfully initialized Qdrant client")
except Exception as e:
    logger.error(f"Error initializing Qdrant client: {e}")
    qdrant_client = None

def ensure_collection_exists():
    """Ensure the collection exists in Qdrant"""
    try:
        if qdrant_client is None:
            logger.error("Qdrant client is not initialized")
            return False
            
        try:
            # Check if collection exists with increased timeout
            collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} already exists with config: {collection_info}")
            return True
        except Exception as e:
            if "Not found" in str(e):
                logger.info(f"Creating collection {COLLECTION_NAME}")
                # Create collection with appropriate vector size for ViT-base model (768 dimensions)
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=768,  # ViT-base embedding size
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfig(
                        indexing_threshold=0  # Index immediately
                    ),
                    timeout=60.0  # Increase timeout for collection creation
                )
                logger.info(f"Collection {COLLECTION_NAME} created successfully")
                return True
            else:
                logger.error(f"Error ensuring collection exists: {e}")
                return False
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        return False

def fetch_nft_metadata(object_type: str, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Fetch NFT metadata from BlockVision API"""
    try:
        # Format the object type for the URL (URL-encoded)
        encoded_object_type = requests.utils.quote(object_type)
        
        # Use the correct endpoint format as shown in your example
        url = f"{BLOCKVISION_BASE_URL}/nft/list"
        
        # Set up parameters - note that we use cursor instead of skip
        # The cursor is a specific object ID, but we can use 0x0 for the initial request
        cursor = f"0x{skip:x}" if skip > 0 else "0x0"
        params = {
            "objectType": encoded_object_type,
            "cursor": cursor
        }
        
        # Set up headers
        headers = {
            "accept": "application/json",
            "x-api-key": BLOCKVISION_API_KEY
        }
        
        logger.info(f"Fetching NFT metadata for {object_type} (cursor={cursor})")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"Error from BlockVision API: {response.status_code} - {response.text}")
            return []
        
        # Parse the response - the structure is different from what we expected
        data = response.json()
        
        # The NFT data is in result.data, not data.list
        nfts = data.get("result", {}).get("data", [])
        
        # Get the next cursor for pagination
        next_cursor = data.get("result", {}).get("cursor")
        if next_cursor:
            logger.info(f"Next cursor for pagination: {next_cursor}")
        
        logger.info(f"Fetched {len(nfts)} NFTs for {object_type}")
        return nfts
    except Exception as e:
        logger.error(f"Error fetching NFT metadata: {e}")
        return []

def extract_image_url(nft_data: Dict[str, Any]) -> Optional[str]:
    """Extract image URL from NFT data"""
    try:
        # The structure is different from what we expected
        # Based on the API response, the image URL is directly in the 'imageUrl' field
        if "imageUrl" in nft_data:
            return nft_data["imageUrl"]
        
        # Fallback to other possible locations
        if "content" in nft_data and "image_url" in nft_data["content"]:
            return nft_data["content"]["image_url"]
        
        if "display" in nft_data and "image_url" in nft_data["display"]:
            return nft_data["display"]["image_url"]
            
        logger.warning(f"Could not find image URL in NFT data: {nft_data}")
        return None
    except Exception as e:
        logger.error(f"Error extracting image URL: {e}")
        return None

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
            return None
        
        embedding = np.array(response.json()["embedding"])
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def upload_to_qdrant(collection_name: str, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]):
    """Upload embeddings to Qdrant"""
    try:
        if qdrant_client is None:
            logger.error("Cannot upload to Qdrant: client is not initialized")
            return
            
        if not embeddings or not metadata or len(embeddings) != len(metadata):
            logger.error(f"Invalid embeddings or metadata: embeddings={len(embeddings) if embeddings else 0}, metadata={len(metadata) if metadata else 0}")
            return
        
        # Ensure all vectors are normalized and have the right shape
        normalized_embeddings = []
        for embedding in embeddings:
            try:
                # Ensure it's a flat array
                flat_embedding = embedding.flatten()
                # Normalize the vector
                norm = np.linalg.norm(flat_embedding)
                if norm > 0:
                    normalized = flat_embedding / norm
                else:
                    normalized = flat_embedding
                normalized_embeddings.append(normalized)
            except Exception as e:
                logger.error(f"Error normalizing embedding: {e}")
                return
        
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
        batch_size = 50  # Reduced batch size for better reliability
        total_batches = (len(points) - 1) // batch_size + 1
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Uploading batch {batch_num}/{total_batches} to Qdrant ({len(batch)} points)")
            
            try:
                # Use increased timeout for upsert operations
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    timeout=60.0  # Increase timeout for upload operations
                )
                logger.info(f"Successfully uploaded batch {batch_num}/{total_batches}")
            except Exception as e:
                logger.error(f"Error uploading batch {batch_num}/{total_batches} to Qdrant: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        logger.info(f"Completed uploading {len(points)} points to Qdrant collection {collection_name}")
    except Exception as e:
        logger.error(f"Error uploading to Qdrant: {e}")

def process_collection(collection_data: Dict[str, Any], output_dir: str, limit: int = 100, skip: int = 0):
    """Process a collection: fetch metadata, generate embeddings, and upload to Qdrant"""
    try:
        collection_name = collection_data.get("name")
        object_type = collection_data.get("object_type")
        
        if not collection_name or not object_type:
            logger.error(f"Invalid collection data: {collection_data}")
            return
        
        logger.info(f"Processing collection {collection_name}")
        
        # Create output directory for this collection
        collection_dir = os.path.join(output_dir, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Fetch NFT metadata
        nfts = fetch_nft_metadata(object_type, limit, skip)
        
        if not nfts:
            logger.warning(f"No NFTs found for collection {collection_name}")
            return
        
        logger.info(f"Processing {len(nfts)} NFTs for collection {collection_name}")
        
        # Extract image URLs and generate embeddings
        embeddings = []
        processed_nfts = []
        
        for nft in nfts:
            # Extract image URL
            image_url = extract_image_url(nft)
            if not image_url:
                logger.warning(f"Could not extract image URL for NFT: {nft.get('objectId', 'unknown')}")
                continue
            
            # Generate embedding
            embedding = generate_embedding(image_url)
            if embedding is None:
                logger.warning(f"Could not generate embedding for image: {image_url}")
                continue
            
            # Add metadata for Qdrant
            object_id = nft.get("objectId") or nft.get("object_id", "unknown")
            
            # Add to lists
            embeddings.append(embedding)
            processed_nfts.append({
                "object_id": object_id,
                "collection_type": collection_name,
                "image_url": image_url
            })
        
        if not embeddings or not processed_nfts:
            logger.warning(f"No valid embeddings generated for collection {collection_name}")
            return
        
        # Save metadata to file
        metadata_file = os.path.join(collection_dir, f"{collection_name}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(processed_nfts, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
        
        # Save embeddings to file
        embeddings_file = os.path.join(collection_dir, f"{collection_name}_embeddings.npy")
        np.save(embeddings_file, np.array(embeddings))
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Upload to Qdrant
        if qdrant_client is not None:
            # Check if collection exists before uploading
            if ensure_collection_exists():
                upload_to_qdrant(COLLECTION_NAME, embeddings, processed_nfts)
            else:
                logger.error("Cannot upload to Qdrant: collection does not exist")
        else:
            logger.error("Cannot upload to Qdrant: client is not initialized")
        
        logger.info(f"Completed processing collection {collection_name}")
    except Exception as e:
        logger.error(f"Error processing collection: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process NFT collections: fetch metadata, generate embeddings, and upload to Qdrant")
    parser.add_argument("--collections", type=str, default="./collections.json", help="Path to collections JSON file")
    parser.add_argument("--output", type=str, default="./data", help="Output directory for metadata and embeddings")
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
