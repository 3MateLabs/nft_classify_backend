"""
Data models for API requests
Uses Pydantic for validation and serialization
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List


class ImageUrlRequest(BaseModel):
    """
    Request model for image URL-based embedding
    Accepts both regular HTTP URLs and base64 encoded data URLs
    """

    img_url: str = Field(..., description="URL of the image or base64 encoded data URL")


class EmbeddingResponse(BaseModel):
    """
    Response model for embedding results
    """

    embedding: List[List[float]] = Field(
        ..., description="2D array of embedding values"
    )


class ErrorResponse(BaseModel):
    """
    Response model for errors
    """

    error: str = Field(..., description="Error message")
    traceback: Optional[str] = Field(None, description="Error traceback for debugging")


class CollectionProcessRequest(BaseModel):
    """
    Request model for processing an NFT collection
    """

    collection_id: str = Field(..., description="ID of the NFT collection to process")
    limit: Optional[int] = Field(20, description="Maximum number of NFTs to process")


class SearchSimilarRequest(BaseModel):
    """
    Request model for searching similar NFTs
    """

    embedding: List[float] = Field(..., description="Image embedding vector")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")
