#!/usr/bin/env python3

"""
Script to upload collection-specific NPZ-stored embeddings to Qdrant efficiently
This script works with the NPZ file format created by generate_collection_embeddings_npz.py
"""

import requests
import numpy as np
import pandas as pd
import os
import json
import time
import sys
import concurrent.futures
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import queue
import threading
import uuid
import hashlib

# Add the project root to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import (
    EMBEDDING_DIR
)

# Qdrant configuration
QDRANT_API_URL = "https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io"

# Constants
COLLECTION_NAME = "nft_embeddings"  # Name of the collection in Qdrant
BATCH_SIZE = 100  # Number of points to upload in a single request
MAX_WORKERS = 10  # Number of concurrent workers

# Function to check if collection exists
def check_collection_exists(collection_name: str) -> bool:
    """Check if a collection exists in Qdrant"""
    url = f"{QDRANT_API_URL}/collections/{collection_name}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking if collection exists: {str(e)}")
        return False

# Function to create collection
def create_collection(collection_name: str, vector_size: int = 768) -> bool:
    """Create a new collection in Qdrant"""
    url = f"{QDRANT_API_URL}/collections/{collection_name}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }
    
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        }
    }
    
    try:
        response = requests.put(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"Successfully created collection {collection_name}")
        return True
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        return False

# Function to upload points to Qdrant
def upload_points(collection_name: str, points: List[Dict[str, Any]]) -> bool:
    """Upload points to Qdrant"""
    url = f"{QDRANT_API_URL}/collections/{collection_name}/points"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }
    
    payload = {
        "points": points
    }
    
    try:
        # Print the first point for debugging
        if points:
            print(f"\nSample point being sent to Qdrant:")
            sample_point = points[0].copy()
            if 'vector' in sample_point:
                sample_point['vector'] = f"[vector with {len(sample_point['vector'])} dimensions]"
            print(json.dumps(sample_point, indent=2))
            
        response = requests.put(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"Error response from Qdrant: {response.status_code}")
            print(f"Response content: {response.text}")
            
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error uploading points: {str(e)}")
        return False

# Function to convert hex string to UUID
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

# Worker function for uploading batches
def upload_worker(collection_name: str, task_queue: queue.Queue, progress: Dict[str, Any]):
    """Worker function to upload batches of points to Qdrant"""
    while True:
        try:
            batch = task_queue.get(block=False)
            if batch is None:  # Sentinel value to stop the worker
                break
                
            success = upload_points(collection_name, batch)
            with progress["lock"]:
                if success:
                    progress["success"] += len(batch)
                else:
                    progress["failed"] += len(batch)
                progress["pbar"].update(len(batch))
                
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error in worker: {str(e)}")
        finally:
            task_queue.task_done()

# Function to list available collections in the embeddings directory
def list_available_collections() -> List[str]:
    """List available collections with NPZ and metadata files"""
    collections = []
    
    # Get all files in the embeddings directory
    files = os.listdir(EMBEDDING_DIR)
    
    # Find NPZ files
    npz_files = [f for f in files if f.endswith('_embeddings.npz')]
    
    # Extract collection names
    for npz_file in npz_files:
        collection_name = npz_file.replace('_embeddings.npz', '')
        metadata_file = f"{collection_name}_metadata.json"
        
        # Check if both NPZ and metadata files exist
        if metadata_file in files:
            collections.append(collection_name)
    
    return collections

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upload collection embeddings to Qdrant')
    parser.add_argument('--collection', type=str, help='Name of the collection to upload')
    parser.add_argument('--list', action='store_true', help='List all available collections')
    args = parser.parse_args()
    
    # List collections if requested
    if args.list:
        collections = list_available_collections()
        print(f"\nAvailable collections ({len(collections)}):\n")
        for i, collection in enumerate(collections):
            print(f"{i+1}. {collection}")
        return
    
    # Check if collection name is provided
    if not args.collection:
        print("Error: Please provide a collection name using --collection or use --list to see available collections")
        return
    
    collection_name = args.collection
    
    # Check if NPZ file and metadata file exist
    npz_filename = f"{collection_name}_embeddings.npz"
    metadata_filename = f"{collection_name}_metadata.json"
    
    npz_path = os.path.join(EMBEDDING_DIR, npz_filename)
    metadata_path = os.path.join(EMBEDDING_DIR, metadata_filename)
    
    if not os.path.exists(npz_path):
        print(f"Error: Embeddings file {npz_path} not found")
        return
        
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} not found")
        return
    
    # Load embeddings from NPZ file
    print(f"Loading embeddings from {npz_path}...")
    embeddings = np.load(npz_path)
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get the number of embeddings
    num_embeddings = len(metadata)
    print(f"Loaded {num_embeddings} embeddings for collection '{collection_name}'")
    
    # Check if Qdrant collection exists, create if not
    if not check_collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} does not exist in Qdrant, creating it...")
        # Get vector size from the first embedding
        first_key = list(embeddings.keys())[0]
        vector_size = embeddings[first_key].shape[0]
        create_collection(COLLECTION_NAME, vector_size)
    else:
        print(f"Collection {COLLECTION_NAME} already exists in Qdrant")
    
    # Prepare batches for upload
    batches = []
    current_batch = []
    
    # Process all embeddings
    for object_id, embedding_data in metadata.items():
        # Skip if object_id is not in the embeddings file
        if object_id not in embeddings:
            print(f"Warning: Object ID {object_id} not found in embeddings file, skipping")
            continue
            
        # Get the embedding vector
        vector = embeddings[object_id]
        
        # Flatten the vector if it's 2D (e.g., shape (1, 768))
        if len(vector.shape) > 1:
            vector = vector.flatten()
        
        # Convert to list for JSON serialization
        vector = vector.tolist()
        
        # Print debug info for the first few points
        if len(current_batch) < 3:
            print(f"\nDebug - Point {len(current_batch)+1}:")
            print(f"ID: {object_id}")
            print(f"Vector type: {type(vector)}")
            print(f"Vector shape before flattening: {embeddings[object_id].shape}")
            print(f"Vector length after flattening: {len(vector)}")
            print(f"Payload: {embedding_data}")
        
        # Create point for Qdrant
        point = {
            "id": hex_to_uuid(object_id),
            "vector": vector,
            "payload": {
                "object_id": object_id,
                "collection_id": embedding_data.get("nft_collection_name", ""),
                "name": embedding_data.get("name", ""),
                "image_url": embedding_data.get("image_url", "")
            }
        }
        
        current_batch.append(point)
        
        # If batch is full, add to batches list
        if len(current_batch) >= BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []
    
    # Add any remaining points to the batches list
    if current_batch:
        batches.append(current_batch)
    
    print(f"Prepared {len(batches)} batches for upload")
    
    # Upload batches in parallel
    task_queue = queue.Queue()
    for batch in batches:
        task_queue.put(batch)
    
    # Add sentinel values to stop workers
    for _ in range(MAX_WORKERS):
        task_queue.put(None)
    
    # Initialize progress tracking
    progress = {
        "success": 0,
        "failed": 0,
        "lock": threading.Lock(),
        "pbar": tqdm(total=num_embeddings, desc=f"Uploading '{collection_name}' to Qdrant")
    }
    
    # Start workers
    workers = []
    for _ in range(MAX_WORKERS):
        worker = threading.Thread(target=upload_worker, args=(COLLECTION_NAME, task_queue, progress))
        worker.start()
        workers.append(worker)
    
    # Wait for all tasks to complete
    task_queue.join()
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join()
    
    # Close progress bar
    progress["pbar"].close()
    
    print(f"\nUpload complete for collection '{collection_name}'!")
    print(f"Successfully uploaded {progress['success']} points")
    print(f"Failed to upload {progress['failed']} points")

if __name__ == "__main__":
    main()
