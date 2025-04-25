"""
Main FastAPI application entry point
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_TITLE, API_DESCRIPTION, API_VERSION, logger
from api.models.request_models import ImageUrlRequest
from api.handlers.embedding_handlers import url_to_embedding, file_to_embedding

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
    Endpoint to search for similar NFTs based on an embedding

    Args:
        request: SearchSimilarRequest object containing the embedding

    Returns:
        JSON response with search results or error details
    """
    # TODO
    # return await search_similar(request)
    # your endpoint, the one you already have
    return {"status": "success", "message": "Search similar NFTs"}


# Log server startup
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")


# Log server shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {API_TITLE}")
