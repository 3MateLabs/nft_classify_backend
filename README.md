# NFT Image Classification System

## Files

1. **1_nft_processor.py** - Process NFT collections and generate embeddings (Run first)
2. **2_nft_search_api.py** - API for searching similar NFTs (Run second)
3. **collections.json** - NFT collection configuration

## Quick Start

### Setup

```bash
# Install dependencies
pip install fastapi uvicorn qdrant-client numpy requests pydantic python-dotenv

# Create .env file with API keys
### Example Usage

```bash
# 1. Process NFT collections
python 1_nft_processor.py --collection doubleup --limit 100

# 2. Run the search API
python -m uvicorn support_api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health` - Check API health
- `GET /search?img_url={url}` - Search for similar NFTs
- `GET /collections` - List available collections

## Supported Collections

- DoubleUp, Aeon, Prime Machine, Rootlets, KillaClub, Kumo

## Features

- Fetches NFT metadata from BlockVision API
- Generates embeddings using 3MateLabs service
- Stores vectors in Qdrant for similarity search
- Includes fallback mechanisms for API failures
