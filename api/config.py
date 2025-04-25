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
