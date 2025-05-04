# NFT Classification and Similarity Search

A comprehensive system for NFT image embedding generation, storage, and similarity search using vector database technology.

## Quick Start Guide

Follow these steps to get the NFT Classification system up and running quickly:

### Prerequisites

- Python 3.12ß+
- Qdrant instance (cloud or local)
- Access to NFT image data

### Installation

1. Clone this repository

```bash
git clone https://github.com/yourusername/nft-classify-backend.git
cd nft-classify-backend
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

#### 1. Start the Local Embedding Service

The local embedding service generates vector embeddings from NFT images.

```bash
cd scripts
python local_embedding_service.py
```

This will start the embedding service on a port.

#### 2. Start the Qdrant Search API

The search API provides endpoints for finding similar NFTs.

```bash
cd scripts
python qdrant_search_api.py
```

This will start the search API on a different port.

#### 3. Generate Embeddings for a Collection (Optional)

If you want to process new NFT collections to generate embeddings:

```bash
# Generate embeddings for a collection
python scripts/3_generate_collection_embeddings_npz_notebook.py
```

#### 4. Upload Embeddings to Qdrant (Optional)

If you have new embeddings to upload to the Qdrant vector database:

```bash
# Upload a collection's embeddings to Qdrant
python scripts/4_upload_collection_embeddings_to_qdrant_notebook.py
```

### Using the API

Once both services are running, you can use the following API endpoints:

#### Search for Similar NFTs

```bash
curl -X POST -H "Content-Type: application/json" -d '{"img_url": "https://example.com/nft-image.jpg", "limit": 5, "threshold": 0.7}' http://localhost:8003/search
```

Example response:

```json
{
  "results": [
    {
      "score": 0.9866967,
      "object_id": "0x7094f7f87bbe3b6be9fbfad6732dc3aa2c2ecc0e9e10f6472381ed0d7d0cf8e1",
      "collection_id": "Prime Machin",
      "name": "Prime Machin #123",
      "image_url": "https://img.sm.xyz/0x7094f7f87bbe3b6be9fbfad6732dc3aa2c2ecc0e9e10f6472381ed0d7d0cf8e1/",
      "nft_type": "0x034c162f6b594cb5a1805264dd01ca5d80ce3eca6522e6ee37fd9ebfb9d3ddca::factory::PrimeMachin",
      "nft_collection_name": "Prime Machin",
      "creator": "0x3c5e1e63fcb09456baf9424a02a1758acbfb85f69faa3ee41247f850104cd10e",
      "description": "",
      "created_time": "1712475154974"
    }
  ],
  "count": 1,
  "query_time_ms": 855.8011054992676
}
```

#### Health Check

```bash
curl http://localhost:8003/health
```

## System Architecture

### Components

1. **Local Embedding Service**: Generates embeddings from NFT images using a ViT model
2. **Qdrant Search API**: Provides endpoints for searching similar NFTs
3. **Qdrant Vector Database**: Stores and indexes embeddings for similarity search
4. **NPZ Storage**: Efficient storage format for embeddings and metadata

### Data Flow

1. NFT images are processed by the embedding service to generate vector embeddings
2. Embeddings are stored in NPZ files along with metadata
3. Embeddings are uploaded to Qdrant for efficient similarity search
4. The API allows searching for similar NFTs based on image URLs

```

## API Keys and Credentials

The system uses the following API keys:

- **BlockVision API**: For fetching NFT metadata
- **Image Embedding Service**: For generating embeddings
- **Qdrant**: For vector database access

Make sure to set up your own API keys in the configuration files before running the system.

## License

[MIT License](LICENSE)
```
