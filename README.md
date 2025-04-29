# NFT Classification and Similarity Search

A comprehensive system for NFT image embedding generation, storage, and similarity search using vector database technology.

## Quick Start Guide

Follow these steps to get the NFT Classification system up and running quickly:

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

3. Set up environment variables (optional)
```bash
# Create a .env file with your configuration
# You can use the example.env as a template
cp example.env .env
```

### Running the System

#### 1. Start the Image Embedding Service

The image embedding service generates vector embeddings from NFT images.

```bash
cd image_embedding_service
python -m uvicorn api.main:app --host 0.0.0.0 --port 3001
```

#### 2. Start the FastAPI Application

The main API provides endpoints for searching and managing NFTs.

```bash
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

#### 3. Generate Embeddings for a Collection

Process NFT collections to generate embeddings and store them efficiently.

```bash
# List available collections
python scripts/generate_collection_embeddings_npz.py --list

# Generate embeddings for a specific collection
python scripts/generate_collection_embeddings_npz.py --collection "Prime Machin"
```

#### 4. Upload Embeddings to Qdrant

Upload the generated embeddings to the Qdrant vector database.

```bash
# Upload a collection's embeddings to Qdrant
python scripts/upload_collection_embeddings_to_qdrant.py --collection "Prime Machin"
```

### Using the API

Once the system is running, you can use the following API endpoints:

#### Search for Similar NFTs

```bash
curl -X POST -H "Content-Type: application/json" -d '{"img_url": "https://example.com/nft-image.jpg", "top_k": 5, "threshold": 0.7}' http://localhost:8000/search
```

#### List Available Collections

```bash
curl http://localhost:8000/collections
```

#### Download NFT Images for a Collection

```bash
curl -X POST -H "Content-Type: application/json" -d '{"collection_name": "Prime Machin", "limit": 10}' http://localhost:8000/download
```

## System Architecture

### Components

1. **FastAPI Application**: Main API for searching and managing NFTs
2. **Image Embedding Service**: Generates embeddings from NFT images using a ViT model
3. **Qdrant Vector Database**: Stores and indexes embeddings for similarity search
4. **NPZ Storage**: Efficient storage format for embeddings and metadata

### Data Flow

1. NFT images are processed by the embedding service to generate vector embeddings
2. Embeddings are stored in NPZ files along with metadata
3. Embeddings are uploaded to Qdrant for efficient similarity search
4. The API allows searching for similar NFTs based on image URLs

## Troubleshooting

### Common Issues

1. **Embedding Service Not Running**
   - Ensure the embedding service is running on port 3001
   - Check that the ViT model is properly loaded

2. **Qdrant Connection Issues**
   - Verify your Qdrant URL and API key
   - Check that the collection exists in Qdrant

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt` to install all required packages

### Logs

- API logs are available in the console when running the FastAPI application
- Embedding generation logs are printed during the collection processing

## File Structure

```
nft-classifier/
│
├── api/
│   ├── app.py                   # Main FastAPI application
│   ├── config.py                # Configuration settings
│   └── start_api.sh             # API startup script
│
├── scripts/
│   ├── generate_collection_embeddings_npz.py    # Generate embeddings
│   └── upload_collection_embeddings_to_qdrant.py # Upload to Qdrant
│
├── data/
│   └── embeddings/              # Storage for NPZ files and metadata
│
├── requirements.txt             # Python dependencies
└── README.md                    # This documentation
```

## License

[MIT License](LICENSE)
