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


class ImageUrlWithPayloadRequest(BaseModel):
    """
    Request model for image URL-based embedding with additional fields
    """

    img_url: str = Field(..., description="URL of the image or base64 encoded data URL")
    payload: dict = Field(
        ..., description="Dictionary containing key-value pairs to upload as JSON"
    )


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
