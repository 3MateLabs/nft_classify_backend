########## CELL 1: Import Libraries and Setup ##########
"""
Script to generate embeddings for a specific NFT collection and store them efficiently in a single NPZ file
This approach processes one collection at a time for better manageability
"""

# Import required libraries
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
try:
    from api.config import EMBEDDING_DIR
except ImportError:
    # Fallback if import fails
    EMBEDDING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "embeddings")
    print(f"Using fallback EMBEDDING_DIR: {EMBEDDING_DIR}")

# Create embedding directory if it doesn't exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)

########## CELL 2: Configuration ##########
# Image embedding service URL and API key
IMAGE_EMBEDDING_API_URL = "http://localhost:3001"
IMAGE_EMBEDDING_API_KEY = "45334ad61f254307a32"  # From memory

# Constants
MAX_WORKERS = 10  # Number of concurrent workers for parallel processing
BATCH_SIZE = 50   # Batch size for saving embeddings to reduce memory usage

# Default CSV path for NFT data
DEFAULT_CSV_PATH = '2_all_nft_datas.csv'

########## CELL 3: Embedding Generation Function ##########
def generate_embedding_from_url(image_url: str) -> Optional[List[float]]:
    """
    Generate embedding for an image using the image embedding service
    
    Args:
        image_url: URL of the image to generate embedding for
        
    Returns:
        List of embedding values or None if failed
    """
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

########## CELL 4: NFT Processing Function ##########
def process_nft(nft: Dict[str, Any]) -> Tuple[bool, str, Optional[List[float]]]:
    """
    Process a single NFT to generate its embedding
    
    Args:
        nft: Dictionary containing NFT data
        
    Returns:
        Tuple of (success, object_id, embedding)
    """
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

########## CELL 5: Batch Saving Function ##########
def save_embeddings_batch(collection_name: str, embeddings_dict: Dict[str, List[float]], 
                         metadata_dict: Dict[str, Dict[str, Any]], mode: str = 'a') -> None:
    """
    Save a batch of embeddings to the NPZ file and update metadata
    
    Args:
        collection_name: Name of the NFT collection
        embeddings_dict: Dictionary of object_id -> embedding
        metadata_dict: Dictionary of object_id -> metadata
        mode: 'w' for write (overwrite), 'a' for append (default)
    """
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

########## CELL 6: Collection Data Loading Functions ##########
def get_collection_nfts(collection_name: str, csv_path: str) -> List[Dict[str, Any]]:
    """
    Get NFTs for a specific collection from the CSV file
    
    Args:
        collection_name: Name of the collection to filter by
        csv_path: Path to the CSV file with NFT data
        
    Returns:
        List of dictionaries containing NFT data
    """
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

def list_collections(csv_path: str) -> List[str]:
    """
    List all collections in the CSV file
    
    Args:
        csv_path: Path to the CSV file with NFT data
        
    Returns:
        List of collection names
    """
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

########## CELL 7: List Available Collections ##########
# Get the CSV path - adjust this to your file location
csv_path = DEFAULT_CSV_PATH
if not os.path.isabs(csv_path):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

# List all available collections
collections = list_collections(csv_path)
print(f"\nAvailable collections ({len(collections)}):\n")
for i, collection in enumerate(collections):
    print(f"{i+1}. {collection}")

########## CELL 8: Select Collection to Process ##########
# Set the collection name here - replace with your desired collection
collection_name = "DoubleUp Citizen"  # Example - change this to your target collection

# Get NFTs for the specified collection
nfts = get_collection_nfts(collection_name, csv_path)

if not nfts:
    print(f"No NFTs found for collection '{collection_name}'")
else:
    print(f"Found {len(nfts)} NFTs for collection '{collection_name}'")
    
    # Display a sample NFT to verify data
    print("\nSample NFT data:")
    sample_nft = nfts[0]
    for key, value in sample_nft.items():
        print(f"{key}: {value}")

########## CELL 9: Check for Existing Embeddings ##########
# Create filenames based on collection name
npz_filename = f"{collection_name}_embeddings.npz"
metadata_filename = f"{collection_name}_metadata.json"

npz_path = os.path.join(EMBEDDING_DIR, npz_filename)
metadata_path = os.path.join(EMBEDDING_DIR, metadata_filename)

# Check if files already exist
overwrite_existing = False
if os.path.exists(npz_path) and os.path.exists(metadata_path):
    print(f"Files already exist for collection '{collection_name}'")
    with open(metadata_path, 'r') as f:
        existing_metadata = json.load(f)
    print(f"Existing metadata contains {len(existing_metadata)} entries")
    
    # In notebook, we'll set this manually rather than prompting
    overwrite_existing = True  # Set to True to overwrite, False to keep existing
    
    if overwrite_existing:
        # Remove existing files
        os.remove(npz_path)
        os.remove(metadata_path)
        print("Removed existing files")
    else:
        print("Keeping existing files - processing will be skipped")

########## CELL 10: Process Collection and Generate Embeddings ##########
# Only run this cell if we have NFTs and either no existing files or we want to overwrite
if nfts and (not os.path.exists(npz_path) or overwrite_existing):
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

########## CELL 11: Verify Results ##########
# Check the generated embeddings and metadata
if os.path.exists(npz_path) and os.path.exists(metadata_path):
    # Load the NPZ file
    embeddings = np.load(npz_path)
    
    # Load the metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Embeddings file contains {len(embeddings.files)} embeddings")
    print(f"Metadata file contains {len(metadata)} entries")
    
    # Display a sample embedding
    if embeddings.files:
        sample_key = embeddings.files[0]
        sample_embedding = embeddings[sample_key]
        print(f"\nSample embedding for {sample_key}:")
        print(f"Shape: {sample_embedding.shape}")
        print(f"First 5 values: {sample_embedding[:5]}")
        
        # Display corresponding metadata
        if sample_key in metadata:
            print(f"\nMetadata for {sample_key}:")
            for k, v in metadata[sample_key].items():
                print(f"{k}: {v}")
else:
    print("No embeddings or metadata files found")
