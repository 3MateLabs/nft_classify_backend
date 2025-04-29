"""API handlers for NFT-related endpoints"""

import traceback
from fastapi import HTTPException, UploadFile, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models.request_models import ImageUrlRequest, CollectionProcessRequest, ErrorResponse
from api.services.image_service import process_image_from_url
from api.services.nft_service import nft_service
from api.services.model_service import model_service
from api.config import logger


async def process_collection(request: CollectionProcessRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    """Handler for processing an NFT collection

    Args:
        request: CollectionProcessRequest object containing the collection ID
        background_tasks: FastAPI BackgroundTasks for background processing

    Returns:
        JSONResponse with status or error details
    """
    try:
        collection_id = request.collection_id
        limit = request.limit if request.limit else 20
        
        # Start processing in the background
        logger.info(f"Starting background processing for collection {collection_id} with limit {limit}")
        background_tasks.add_task(nft_service.process_collection, collection_id, limit)
        
        # Return immediate response
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "message": f"Processing started for collection {collection_id} with limit {limit}",
                "collection_id": collection_id
            }
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in process_collection: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e), traceback=error_traceback).dict(),
        )


async def search_similar_nfts(request: ImageUrlRequest) -> JSONResponse:
    """Handler for searching similar NFTs based on an image

    Args:
        request: ImageUrlRequest object containing the image URL

    Returns:
        JSONResponse with search results or error details
    """
    try:
        # Process the image from URL
        logger.info(f"Processing image from URL: {request.img_url[:100]}...")
        image = process_image_from_url(request.img_url)

        # Generate embedding
        logger.info("Generating embedding...")
        embedding = model_service.generate_embedding(image)

        # Search for similar NFTs in Qdrant
        logger.info("Searching for similar NFTs...")
        results = nft_service.search_similar_nfts("nft_embeddings", embedding[0], limit=10)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.get("score", 0),
                "object_id": result.get("payload", {}).get("object_id", ""),
                "collection_id": result.get("payload", {}).get("collection_id", ""),
                "name": result.get("payload", {}).get("name", ""),
                "image_url": result.get("payload", {}).get("image_url", "")
            })

        # Return results
        return JSONResponse(
            content={
                "results": formatted_results
            }
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in search_similar_nfts: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e), traceback=error_traceback).dict(),
        )


async def get_top_collections(limit: int = 10) -> JSONResponse:
    """Handler for getting top NFT collections

    Args:
        limit: Maximum number of collections to return

    Returns:
        JSONResponse with collections or error details
    """
    try:
        # Get top collections
        logger.info(f"Getting top {limit} NFT collections")
        collections = nft_service.get_top_nft_collections(limit=limit)

        # Return collections
        return JSONResponse(
            content={
                "collections": collections
            }
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in get_top_collections: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e), traceback=error_traceback).dict(),
        )
