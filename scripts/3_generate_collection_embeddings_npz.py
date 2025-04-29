#!/usr/bin/env python3

"""
Script to generate embeddings for a specific NFT collection and store them efficiently in a single NPZ file
This approach processes one collection at a time for better manageability
"""

import requests
import pandas as pd
import numpy as np
import os
import json
import time
import sys
import concurrent.futures
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# Add the project root to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import (
    EMBEDDING_DIR
)

# Image embedding service URL and API key
IMAGE_EMBEDDING_API_URL = "http://localhost:3001"
IMAGE_EMBEDDING_API_KEY = "45334ad61f254307a32"

# Constants
MAX_WORKERS = 10  # Number of concurrent workers
BATCH_SIZE = 50    # Batch size for saving embeddings

# Create embedding directory if it doesn't exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Function to generate embedding from image URL
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
        
        # Print response status for debugging if there's an error
        if response.status_code != 200:
            print(f"Error response for {image_url}: Status {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        if "embedding" in data:
            return data["embedding"]
        else:
            print(f"No embedding found in response for {image_url}")
            return None
    except Exception as e:
        print(f"Error generating embedding for {image_url}: {str(e)}")
        return None

# Function to process a single NFT
def process_nft(nft: Dict[str, Any]) -> Tuple[bool, str, Optional[List[float]]]:
    """Process a single NFT to generate its embedding"""
    # Handle different possible column names
    object_id = nft.get("object_id")
    collection_name = nft.get("nft_collection_name")
    name = nft.get("name", "")
    image_url = nft.get("image_url")
    
    if not object_id:
        print("Error: NFT missing object_id")
        return False, "", None
    
    if not image_url:
        print(f"No image URL found for NFT {object_id}")
        return False, object_id, None
    
    # Generate embedding
    embedding = generate_embedding_from_url(image_url)
    
    if embedding is None:
        print(f"Failed to generate embedding for NFT {object_id}")
        return False, object_id, None
    
    return True, object_id, embedding

# Function to save embeddings in batches to NPZ file
def save_embeddings_batch(collection_name: str, embeddings_dict: Dict[str, List[float]], 
                         metadata_dict: Dict[str, Dict[str, Any]], mode: str = 'a') -> None:
    """Save a batch of embeddings to the NPZ file and update metadata"""
    # Create filenames based on collection name
    npz_filename = f"{collection_name}_embeddings.npz"
    metadata_filename = f"{collection_name}_metadata.json"
    
    npz_path = os.path.join(EMBEDDING_DIR, npz_filename)
    metadata_path = os.path.join(EMBEDDING_DIR, metadata_filename)
    
    # Convert embeddings to numpy arrays
    np_embeddings = {}
    for object_id, embedding in embeddings_dict.items():
        np_embeddings[object_id] = np.array(embedding)
    
    # Save embeddings to NPZ file
    if mode == 'w' or not os.path.exists(npz_path):
        # Create new NPZ file
        np.savez_compressed(npz_path, **np_embeddings)
    else:
        # Append to existing NPZ file
        # Since NPZ doesn't support direct append, we need to load existing data,
        # merge with new data, and save everything back
        existing_data = dict(np.load(npz_path))
        existing_data.update(np_embeddings)
        np.savez_compressed(npz_path, **existing_data)
    
    # Update metadata file
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = {}
    
    existing_metadata.update(metadata_dict)
    
    with open(metadata_path, 'w') as f:
        json.dump(existing_metadata, f)
    
    print(f"Saved batch of {len(embeddings_dict)} embeddings to {npz_path}")

# Function to get NFTs for a specific collection
def get_collection_nfts(collection_name: str, csv_path: str) -> List[Dict[str, Any]]:
    """Get NFTs for a specific collection from the CSV file"""
    # Load NFT data from CSV file
    if not os.path.exists(csv_path):
        print(f"Error: NFT data file {csv_path} not found")
        return []
    
    try:
        nfts_df = pd.read_csv(csv_path)
        print(f"Loaded CSV with columns: {list(nfts_df.columns)}")
        
        # Check if 'nft_collection_name' column exists
        collection_column = None
        if 'nft_collection_name' in nfts_df.columns:
            collection_column = 'nft_collection_name'
        elif 'nftCollectionName' in nfts_df.columns:
            collection_column = 'nftCollectionName'
        
        if collection_column is None:
            print("Error: Could not find collection name column in CSV")
            return []
        
        # Filter by collection name
        collection_nfts = nfts_df[nfts_df[collection_column] == collection_name]
        
        print(f"Found {len(collection_nfts)} NFTs for collection '{collection_name}'")
        
        # Convert DataFrame to list of dictionaries
        return collection_nfts.to_dict('records')
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return []

# Function to list all collections in the CSV file
def list_collections(csv_path: str) -> List[str]:
    """List all collections in the CSV file"""
    if not os.path.exists(csv_path):
        print(f"Error: NFT data file {csv_path} not found")
        return []
    
    try:
        nfts_df = pd.read_csv(csv_path)
        
        # Check if 'nft_collection_name' column exists
        collection_column = None
        if 'nft_collection_name' in nfts_df.columns:
            collection_column = 'nft_collection_name'
        elif 'nftCollectionName' in nfts_df.columns:
            collection_column = 'nftCollectionName'
        
        if collection_column is None:
            print("Error: Could not find collection name column in CSV")
            return []
            
        collections = nfts_df[collection_column].unique().tolist()
        return collections
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return []

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings for a specific NFT collection')
    parser.add_argument('--collection', type=str, help='Name of the collection to process')
    parser.add_argument('--list', action='store_true', help='List all available collections')
    parser.add_argument('--csv', type=str, default='../data/embeddings/all_nft_data.csv', help='Path to the CSV file with NFT data')
    args = parser.parse_args()
    
    # Get the CSV path
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)
    
    # List collections if requested
    if args.list:
        collections = list_collections(csv_path)
        print(f"\nAvailable collections ({len(collections)}):\n")
        for i, collection in enumerate(collections):
            print(f"{i+1}. {collection}")
        return
    
    # Check if collection name is provided
    if not args.collection:
        print("Error: Please provide a collection name using --collection or use --list to see available collections")
        return
    
    collection_name = args.collection
    
    # Get NFTs for the specified collection
    nfts = get_collection_nfts(collection_name, csv_path)
    
    if not nfts:
        print(f"No NFTs found for collection '{collection_name}'")
        return
    
    print(f"Found {len(nfts)} NFTs for collection '{collection_name}'")
    
    # Create filenames based on collection name
    npz_filename = f"{collection_name}_embeddings.npz"
    metadata_filename = f"{collection_name}_metadata.json"
    
    npz_path = os.path.join(EMBEDDING_DIR, npz_filename)
    metadata_path = os.path.join(EMBEDDING_DIR, metadata_filename)
    
    # Check if files already exist
    if os.path.exists(npz_path) and os.path.exists(metadata_path):
        print(f"Files already exist for collection '{collection_name}'")
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
        print(f"Existing metadata contains {len(existing_metadata)} entries")
        
        # Ask if user wants to overwrite
        response = input("Do you want to overwrite existing files? (y/n): ")
        if response.lower() != 'y':
            print("Aborting operation")
            return
        
        # Remove existing files
        os.remove(npz_path)
        os.remove(metadata_path)
        print("Removed existing files")
    
    # Initialize counters and storage
    success_count = 0
    failed_count = 0
    failed_nfts = []
    
    # Initialize batch storage
    batch_embeddings = {}
    batch_metadata = {}
    batch_count = 0
    
    # Process NFTs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_nft = {executor.submit(process_nft, nft): nft for nft in nfts}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_nft), total=len(nfts), 
                          desc=f"Processing '{collection_name}' NFTs"):
            nft = future_to_nft[future]
            try:
                success, object_id, embedding = future.result()
                
                if success and embedding is not None:
                    # Add to batch
                    batch_embeddings[object_id] = embedding
                    
                    # Store metadata
                    batch_metadata[object_id] = {
                        "object_id": object_id,
                        "nft_collection_name": nft.get("nft_collection_name"),
                        "name": nft.get("name", ""),
                        "image_url": nft.get("image_url"),
                        "embedding_dimensions": len(embedding)
                    }
                    
                    batch_count += 1
                    success_count += 1
                    
                    # Save batch if it reaches the batch size
                    if batch_count >= BATCH_SIZE:
                        save_embeddings_batch(collection_name, batch_embeddings, batch_metadata)
                        batch_embeddings = {}
                        batch_metadata = {}
                        batch_count = 0
                else:
                    failed_count += 1
                    failed_nfts.append({
                        "object_id": object_id,
                        "nft_collection_name": nft.get("nft_collection_name"),
                        "name": nft.get("name", ""),
                        "image_url": nft.get("image_url")
                    })
            except Exception as e:
                print(f"Error processing NFT {nft.get('object_id')}: {str(e)}")
                failed_count += 1
                failed_nfts.append({
                    "object_id": nft.get("object_id", ""),
                    "nft_collection_name": nft.get("nft_collection_name", ""),
                    "name": nft.get("name", ""),
                    "image_url": nft.get("image_url", ""),
                    "error": str(e)
                })
    
    # Save any remaining embeddings in the batch
    if batch_count > 0:
        save_embeddings_batch(collection_name, batch_embeddings, batch_metadata)
    
    # Save failed NFTs to CSV
    if failed_nfts:
        failed_df = pd.DataFrame(failed_nfts)
        failed_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      f"{collection_name}_failed_nfts.csv")
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Failed NFTs saved to {failed_csv_path}")
    
    print(f"\nProcessing complete for collection '{collection_name}'!")
    print(f"Successfully generated embeddings for {success_count} NFTs")
    print(f"Failed to generate embeddings for {failed_count} NFTs")
    print(f"All embeddings saved to {npz_path}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
