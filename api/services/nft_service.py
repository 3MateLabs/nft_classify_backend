"""NFT service module
Handles NFT downloading, processing, and embedding generation
"""

import requests
import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from api.config import (
    BLOCKVISION_API_KEY,
    IMAGE_EMBEDDING_API_KEY,
    QDRANT_API_KEY,
    BLOCKVISION_API_URL,
    IMAGE_EMBEDDING_API_URL,
    QDRANT_API_URL,
    IMAGE_DIR,
    EMBEDDING_DIR,
    logger
)


class NFTService:
    """Service for handling NFT operations"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing NFTService...")
            cls._instance = super(NFTService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize service"""
        # Create directories for downloaded images and embeddings if they don't exist
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(EMBEDDING_DIR, exist_ok=True)
        logger.info(f"Initialized directories: {IMAGE_DIR}, {EMBEDDING_DIR}")

    def get_top_nft_collections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top NFT collections from BlockVision API"""
        url = f"{BLOCKVISION_API_URL}/nft/collections"
        
        params = {
            "page": 1,
            "limit": limit,
            "sort": "volume",  # Sort by trading volume
            "order": "desc"   # Descending order (highest first)
        }
        
        headers = {
            "accept": "application/json",
            "X-API-KEY": BLOCKVISION_API_KEY
        }
        
        try:
            logger.info(f"Fetching top {limit} NFT collections from BlockVision API")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data and "collections" in data["data"]:
                collections = data["data"]["collections"]
                logger.info(f"Successfully fetched {len(collections)} collections")
                return collections
            else:
                logger.error("Unexpected response format from BlockVision API")
                return []
        except Exception as e:
            logger.error(f"Error fetching top NFT collections: {str(e)}")
            return []

    def get_nfts_by_collection(self, collection_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get NFTs for a specific collection from BlockVision API"""
        url = f"{BLOCKVISION_API_URL}/nft/collection/nfts"
        
        params = {
            "collection": collection_id,
            "page": 1,
            "limit": limit
        }
        
        headers = {
            "accept": "application/json",
            "X-API-KEY": BLOCKVISION_API_KEY
        }
        
        try:
            logger.info(f"Fetching NFTs for collection {collection_id} from BlockVision API")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data and "nfts" in data["data"]:
                nfts = data["data"]["nfts"]
                logger.info(f"Successfully fetched {len(nfts)} NFTs for collection {collection_id}")
                return nfts
            else:
                logger.error(f"Unexpected response format from BlockVision API for collection {collection_id}")
                return []
        except Exception as e:
            logger.error(f"Error fetching NFTs for collection {collection_id}: {str(e)}")
            return []

    def get_nft_metadata(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific NFT from BlockVision API"""
        url = f"{BLOCKVISION_API_URL}/nft/object/{object_id}"
        
        headers = {
            "accept": "application/json",
            "X-API-KEY": BLOCKVISION_API_KEY
        }
        
        try:
            logger.info(f"Fetching metadata for NFT {object_id} from BlockVision API")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data:
                metadata = data["data"]
                logger.info(f"Successfully fetched metadata for NFT {object_id}")
                return metadata
            else:
                logger.error(f"Unexpected response format from BlockVision API for NFT {object_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching metadata for NFT {object_id}: {str(e)}")
            return None

    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download an image from URL and save to disk"""
        try:
            logger.info(f"Downloading image from {image_url}")
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded image to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {str(e)}")
            return False

    def generate_embedding(self, image_url: str) -> Optional[List[List[float]]]:
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
            logger.info(f"Generating embedding for image {image_url}")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if "embedding" in data:
                logger.info(f"Successfully generated embedding for image {image_url}")
                return data["embedding"]
            else:
                logger.error(f"No embedding found in response for {image_url}")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding for {image_url}: {str(e)}")
            return None

    def upload_to_qdrant(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Upload vectors to Qdrant database"""
        # First, check if collection exists, if not create it
        create_collection_url = f"{QDRANT_API_URL}/collections/{collection_name}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": QDRANT_API_KEY
        }
        
        try:
            # Check if collection exists
            logger.info(f"Checking if collection {collection_name} exists in Qdrant")
            response = requests.get(create_collection_url, headers=headers)
            
            # If collection doesn't exist, create it
            if response.status_code == 404:
                logger.info(f"Collection {collection_name} not found, creating it")
                create_payload = {
                    "vectors": {
                        "size": 768,  # Size of ViT-base-patch16-224 embeddings
                        "distance": "Cosine"
                    }
                }
                
                response = requests.put(create_collection_url, json=create_payload, headers=headers)
                response.raise_for_status()
                logger.info(f"Successfully created collection {collection_name} in Qdrant")
            
            # Upload points to collection
            upload_url = f"{QDRANT_API_URL}/collections/{collection_name}/points"
            
            upload_payload = {
                "points": points
            }
            
            logger.info(f"Uploading {len(points)} points to collection {collection_name} in Qdrant")
            response = requests.put(upload_url, json=upload_payload, headers=headers)
            response.raise_for_status()
            
            logger.info(f"Successfully uploaded {len(points)} points to collection {collection_name} in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error uploading points to collection {collection_name} in Qdrant: {str(e)}")
            return False

    def search_similar_nfts(self, collection_name: str, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar NFTs in Qdrant database"""
        search_url = f"{QDRANT_API_URL}/collections/{collection_name}/points/search"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": QDRANT_API_KEY
        }
        
        payload = {
            "vector": query_vector,
            "limit": limit
        }
        
        try:
            logger.info(f"Searching for similar NFTs in collection {collection_name} in Qdrant")
            response = requests.post(search_url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if "result" in data:
                results = data["result"]
                logger.info(f"Found {len(results)} similar NFTs in collection {collection_name}")
                return results
            else:
                logger.error(f"Unexpected response format from Qdrant for search in collection {collection_name}")
                return []
        except Exception as e:
            logger.error(f"Error searching for similar NFTs in collection {collection_name}: {str(e)}")
            return []

    def process_collection(self, collection_id: str, nfts_limit: int = 20) -> List[Dict[str, Any]]:
        """Process a collection: download NFTs, generate embeddings, and upload to Qdrant"""
        # Get NFTs for this collection
        nfts = self.get_nfts_by_collection(collection_id, limit=nfts_limit)
        
        if not nfts:
            logger.warning(f"No NFTs found for collection {collection_id}")
            return []
        
        # Process each NFT
        processed_nfts = []
        qdrant_points = []
        
        for nft in nfts:
            object_id = nft.get("objectId")
            
            # Get detailed metadata
            metadata = self.get_nft_metadata(object_id)
            
            if not metadata:
                logger.warning(f"No metadata found for NFT {object_id}. Skipping.")
                continue
            
            # Extract image URL
            image_url = metadata.get("imageUrl")
            
            if not image_url:
                logger.warning(f"No image URL found for NFT {object_id}. Skipping.")
                continue
            
            # Create a safe filename
            safe_name = f"{collection_id}_{object_id}.jpg".replace(':', '_')
            image_path = os.path.join(IMAGE_DIR, safe_name)
            
            # Download image
            download_success = self.download_image(image_url, image_path)
            
            if not download_success:
                logger.warning(f"Failed to download image for NFT {object_id}. Skipping.")
                continue
            
            # Generate embedding
            embedding = self.generate_embedding(image_url)
            
            if not embedding:
                logger.warning(f"Failed to generate embedding for NFT {object_id}. Skipping.")
                continue
            
            # Save embedding to file
            embedding_path = os.path.join(EMBEDDING_DIR, f"{safe_name}.json")
            with open(embedding_path, 'w') as f:
                json.dump(embedding, f)
            
            # Add to list of processed NFTs
            processed_nft = {
                "object_id": object_id,
                "collection_id": collection_id,
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "image_url": image_url,
                "image_path": image_path,
                "embedding_path": embedding_path
            }
            
            processed_nfts.append(processed_nft)
            
            # Prepare data for Qdrant
            qdrant_point = {
                "id": object_id,
                "vector": embedding[0],  # Assuming the embedding is a 2D array with a single vector
                "payload": {
                    "object_id": object_id,
                    "collection_id": collection_id,
                    "name": metadata.get("name", ""),
                    "image_url": image_url
                }
            }
            
            qdrant_points.append(qdrant_point)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Upload embeddings to Qdrant
        if qdrant_points:
            upload_success = self.upload_to_qdrant("nft_embeddings", qdrant_points)
            if not upload_success:
                logger.error(f"Failed to upload embeddings to Qdrant for collection {collection_id}")
        
        return processed_nfts


# Create a global service instance
nft_service = NFTService()
