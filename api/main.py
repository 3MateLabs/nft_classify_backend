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
import traceback
from typing import Dict, Any, Union, List, Optional
import uuid

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

qdrant_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
)
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
    try:
        payload = request.payload
        embedding_response = await url_to_embedding(request)
        embedding = json.loads(embedding_response.body)["embedding"][0]

        # Check if Qdrant URL is configured
        if not os.getenv("QDRANT_URL"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Qdrant URL not configured. Please set QDRANT_URL in .env file."
                },
            )

        # First, search if there's an exact match for this vector
        try:
            # Check if collection exists before searching
            collection_exists = await qdrant_client.collection_exists(
                collection_name=collection_name
            )

            existing_point_id = None
            if collection_exists:
                # Search for exact matches with high threshold
                search_results = await qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=1,
                    score_threshold=0.99,  # High threshold for exact match
                )

                if search_results and len(search_results) > 0:
                    # Found an exact match, use its ID
                    existing_point_id = search_results[0].id
                    logger.info(f"Found exact match with ID: {existing_point_id}")

            # Use existing ID if found, otherwise use provided ID or generate new one
            point_id = existing_point_id or payload.get("id") or str(uuid.uuid4())
            points = [PointStruct(id=point_id, vector=embedding, payload=payload)]

            # Upsert the vector with payload to Qdrant
            await qdrant_client.upsert(collection_name=collection_name, points=points)

        except Exception as e:
            try:
                # Check if collection doesn't exist
                collection_exists = await qdrant_client.collection_exists(
                    collection_name=collection_name
                )
                if not collection_exists:
                    # Create collection with the dimension of the vector
                    await qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config={"size": len(embedding), "distance": "Cosine"},
                    )
                    # Generate a new ID since there can't be any matches in a new collection
                    point_id = payload.get("id") or str(uuid.uuid4())
                    points = [
                        PointStruct(id=point_id, vector=embedding, payload=payload)
                    ]
                    # Try upsert again
                    await qdrant_client.upsert(
                        collection_name=collection_name, points=points
                    )
                else:
                    # If it's another error, return a meaningful error message
                    logger.error(f"Error upserting to Qdrant: {str(e)}")
                    return {"error": f"Error upserting to Qdrant: {str(e)}"}
            except Exception as inner_e:
                # Handle connection errors
                logger.error(f"Error connecting to Qdrant: {str(inner_e)}")
                return {"error": f"Error connecting to Qdrant: {str(inner_e)}"}

        return {
            "status": "success",
            "message": "NFT uploaded to Qdrant",
            "embedding": embedding[:5],
            "point_id": point_id,
            "updated_existing": existing_point_id is not None,
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in upload_url_with_payload: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": error_traceback},
        )


@app.post("/search_similar")
async def search_similar(request: ImageUrlRequest):
    """
    Endpoint to search for similar NFTs based on an embedding

    Args:
        request: SearchSimilarRequest object containing the embedding

    Returns:
        JSON response with search results or error details
    """
    try:
        embedding_response = await url_to_embedding(request)

        # Check if embedding_response is a JSONResponse and extract content
        if hasattr(embedding_response, "body"):
            import json

            embedding_response = json.loads(embedding_response.body)

        vector = embedding_response["embedding"][0]

        # Check if Qdrant URL is configured
        if not os.getenv("QDRANT_URL"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Qdrant URL not configured. Please set QDRANT_URL in .env file."
                },
            )

        try:
            result = await qdrant_client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=10,
            )
            return result
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error connecting to Qdrant: {str(e)}"},
            )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in search_similar: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": error_traceback},
        )


# Log server startup
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")


# Log server shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {API_TITLE}")
