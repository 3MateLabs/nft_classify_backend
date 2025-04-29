"""
Configuration module for the application
Contains application-wide settings, constants, and feature flags
"""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path configurations
MODEL_PATH = "./local_model"
PROCESSOR_PATH = "./local_processor"

# API settings
API_TITLE = "NFT Classification Backend"
API_DESCRIPTION = "Backend service for NFT classification with image embedding"
API_VERSION = "1.0.0"

# Request settings
REQUEST_TIMEOUT = 10  # seconds
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_FACTOR = 2  # exponential backoff

# Feature flags
try:
    import pillow_avif

    AVIF_SUPPORTED = True
except ImportError:
    AVIF_SUPPORTED = False
    logger.warning("pillow-avif-plugin not found. AVIF support may be limited.")

try:
    import cairosvg

    SVG_SUPPORTED = True
except ImportError:
    SVG_SUPPORTED = False
    logger.warning("CairoSVG not found. SVG support will be limited.")

# NFT Service settings
BLOCKVISION_API_KEY = "2uulalvqIxowwmCkEMGozKfUmrW"
IMAGE_EMBEDDING_API_KEY = "45334ad61f254307a32"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2FwGSL4xcHHqtrNJ3-Nffi6Ext0qpI5VzC9MrK153io"

# API Endpoints
BLOCKVISION_API_URL = "https://api.blockvision.org/v1/sui/mainnet"
IMAGE_EMBEDDING_API_URL = "http://localhost:8000"
QDRANT_API_URL = "https://55daf392-afac-492f-bf66-2871e1510fc7.us-east4-0.gcp.cloud.qdrant.io:6333"

# Storage paths
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images")
EMBEDDING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "embeddings")
