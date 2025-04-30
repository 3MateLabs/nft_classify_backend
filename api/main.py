"""
Main FastAPI application entry point
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_TITLE, API_DESCRIPTION, API_VERSION, logger
from api.models.request_models import ImageUrlRequest, ImageUrlWithPayloadRequest
from api.handlers.embedding_handlers import url_to_embedding, file_to_embedding

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct

import os
import dotenv
import json

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

qdrant_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
collection_name = os.getenv("QDRANT_COLLECTION_NAME")


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


@app.get("/check_collection_exist")
async def check_collection_exist():
    """
    Endpoint to check if the collection exists in Qdrant
    """
    return await qdrant_client.collection_exists(collection_name=collection_name)


@app.post("/upload_url_with_payload")
async def upsert_url_with_payload_to_db(request: ImageUrlWithPayloadRequest):
    """
    Endpoint to upload an image URL and a file to Qdrant
    """
    embedding_response = await url_to_embedding(request)
    payload = json.loads(request.fields)
    vector = embedding_response["embedding"]
    try:
        # Upsert the vector with payload to Qdrant
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=payload.get("id", None), vector=vector, payload=payload)
            ],
        )
    except Exception as e:
        # Check if collection doesn't exist
        collection_exists = await qdrant_client.collection_exists(
            collection_name=collection_name
        )
        if not collection_exists:
            # Create collection with the dimension of the vector
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": len(vector), "distance": "cosine"},
            )
            # Try upsert again
            await qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=payload.get("id", None), vector=vector, payload=payload
                    )
                ],
            )
        else:
            # If it's another error, re-raise it
            raise e

    return {"status": "success", "message": "NFT uploaded to Qdrant"}


@app.post("/search_similar")
async def search_similar(request: ImageUrlRequest):
    """
    Endpoint to search for similar NFTs based on an embedding

    Args:
        request: SearchSimilarRequest object containing the embedding

    Returns:
        JSON response with search results or error details
    """
    embedding = await url_to_embedding(request)
    result = await qdrant_client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=10,
    )
    return result


# Log server startup
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")


# Log server shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {API_TITLE}")
