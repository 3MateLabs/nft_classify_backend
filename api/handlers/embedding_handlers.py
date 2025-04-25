"""
API handlers for embedding generation endpoints
"""

import traceback
from fastapi import HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse

from api.models.request_models import ImageUrlRequest, EmbeddingResponse, ErrorResponse
from api.services.image_service import process_image_from_url, process_image_from_bytes
from api.services.model_service import model_service
from api.config import logger


async def url_to_embedding(request: ImageUrlRequest) -> JSONResponse:
    """
    Handler for generating embeddings from image URLs

    Args:
        request: ImageUrlRequest object containing the image URL

    Returns:
        JSONResponse with embedding data or error details
    """
    try:
        # Process the image from URL
        logger.info(f"Processing image from URL: {request.img_url[:100]}...")
        image = process_image_from_url(request.img_url)

        # Generate embedding
        logger.info("Generating embedding...")
        embedding = model_service.generate_embedding(image)

        # Return result
        return JSONResponse(content={"embedding": embedding})
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in url_to_embedding: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e), traceback=error_traceback).dict(),
        )


async def file_to_embedding(file: UploadFile) -> JSONResponse:
    """
    Handler for generating embeddings from uploaded files

    Args:
        file: UploadFile object containing the image data

    Returns:
        JSONResponse with embedding data or error details
    """
    try:
        # Read file content
        logger.info(f"Processing uploaded file: {file.filename}")
        contents = await file.read()

        # Process the image
        image = process_image_from_bytes(contents, filename=file.filename)

        # Generate embedding
        logger.info("Generating embedding...")
        embedding = model_service.generate_embedding(image)

        # Return result
        return JSONResponse(content={"embedding": embedding})
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in file_to_embedding: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500 if isinstance(e, HTTPException) else 400,
            content=ErrorResponse(error=str(e), traceback=error_traceback).dict(),
        )
