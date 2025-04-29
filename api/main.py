"""
Main FastAPI application entry point
"""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_TITLE, API_DESCRIPTION, API_VERSION, logger
from api.models.request_models import ImageUrlRequest, CollectionProcessRequest
from api.handlers.embedding_handlers import url_to_embedding, file_to_embedding
from api.handlers.nft_handlers import process_collection, search_similar_nfts, get_top_collections

import os
import dotenv

dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Health check endpoint
@app.get("/")
async def read_root():
    """Health check endpoint"""
    print("Health check endpoint")
    print(os.getenv("TEST"))
    return {"status": "healthy", "message": "NFT Classification Backend is running"}


# Embedding endpoints
@app.post("/embed_from_url")
async def embed_from_url(request: ImageUrlRequest):
    """
    Endpoint to get embeddings from an image URL

    Supports regular HTTP/HTTPS URLs and base64 encoded data URLs

    Args:
        request: ImageUrlRequest object containing the image URL

    Returns:
        JSON response with embedding data or error details
    """
    return await url_to_embedding(request)


@app.post("/embed_from_file")
async def embed_from_file(file: UploadFile = File(...)):
    """
    Endpoint to get embeddings from an uploaded image file

    Supports various image formats including PNG, JPG, WEBP, AVIF, and SVG

    Args:
        file: Uploaded file

    Returns:
        JSON response with embedding data or error details
    """
    return await file_to_embedding(file)


@app.post("/search_similar")
async def search_similar(request: ImageUrlRequest):
    """
    Endpoint to search for similar NFTs based on an image URL

    Args:
        request: ImageUrlRequest object containing the image URL

    Returns:
        JSON response with search results or error details
    """
    return await search_similar_nfts(request)


# NFT Collection endpoints
@app.get("/nft/collections")
async def get_collections(limit: int = 10):
    """
    Endpoint to get top NFT collections

    Args:
        limit: Maximum number of collections to return (default: 10)

    Returns:
        JSON response with collections or error details
    """
    return await get_top_collections(limit)


@app.post("/nft/process_collection")
async def trigger_collection_processing(request: CollectionProcessRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger processing of an NFT collection
    
    This is an asynchronous endpoint that will return immediately and process the collection in the background.

    Args:
        request: CollectionProcessRequest object containing the collection ID and processing parameters
        background_tasks: FastAPI BackgroundTasks for background processing

    Returns:
        JSON response with status or error details
    """
    return await process_collection(request, background_tasks)


# Log server startup
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    
    # Create data directories if they don't exist
    from api.config import IMAGE_DIR, EMBEDDING_DIR
    import os
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    logger.info(f"Created data directories: {IMAGE_DIR}, {EMBEDDING_DIR}")


# Log server shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {API_TITLE}")
