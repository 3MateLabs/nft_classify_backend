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
cat > .env << EOL
BLOCKVISION_API_KEY=2uulalvqIxowwmCkEMGozKfUmrW
EMBEDDING_API_KEY=45334ad61f254307a32
EMBEDDING_API_URL=https://image-embedding-service.3matelabs.com/embed_from_url
QDRANT_URL=https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io
EOL
```

### Usage

```bash
# 1. Process NFT collections
python 1_nft_processor.py --collection doubleup --limit 100

# 2. Run the search API
python -m uvicorn 2_nft_search_api:app --reload
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
