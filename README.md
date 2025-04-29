# NFT Classification and Similarity Search

A comprehensive system for NFT image embedding generation, storage, and similarity search using vector database technology.

## Features

- **Efficient Embedding Storage**: Generate and store NFT image embeddings in compressed NPZ files
- **Vector Database Integration**: Upload and search embeddings in Qdrant vector database
- **Collection-Based Processing**: Process NFT collections individually for better manageability
- **Similarity Search API**: Find similar NFTs based on image URL
- **Metadata Filtering**: Filter NFTs by collection, name, and other attributes
- **Background Processing**: Download NFT images in the background
- **RESTful API**: Clean, well-documented API endpoints

## Components

### 1. FastAPI Application

The main API service provides endpoints for:
- Searching similar NFTs by image URL
- Downloading NFT images for specific collections
- Listing available collections

### 2. Image Embedding Service

A separate service that generates embeddings from images using a ViT model.

### 3. Scripts

Utility scripts for:
- Generating embeddings for NFT collections
- Uploading embeddings to Qdrant
- Inspecting the Qdrant database

## Getting Started

### Prerequisites

- Python 3.9+
- Qdrant instance (cloud or local)
- Access to NFT image data

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/nft-classifier.git
cd nft-classifier
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables (copy example.env to .env and fill in your values)
```bash
cp example.env .env
```

### Running the Services

1. Start the image embedding service
```bash
cd image_embedding_service
python -m uvicorn api.main:app --host 0.0.0.0 --port 3001
```

2. Start the FastAPI application
```bash
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Usage

### Generate Embeddings for a Collection

```bash
python scripts/generate_collection_embeddings_npz.py --collection "Collection Name"
```

### Upload Embeddings to Qdrant

```bash
python scripts/upload_collection_embeddings_to_qdrant.py --collection "Collection Name"
```

### API Endpoints

- `GET /collections` - List all available collections
- `POST /search` - Search for similar NFTs by image URL
- `POST /download` - Download NFT images for a collection

## Data Structure

- **NPZ Files**: Store embeddings efficiently in compressed format
- **JSON Files**: Store metadata for each embedding
- **CSV Files**: Source data for NFT collections

## License

[MIT License](LICENSE)
