"""
ML model service
Handles model loading, inference, and management
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Union
from transformers import AutoImageProcessor, AutoModel

from api.config import MODEL_PATH, PROCESSOR_PATH, logger


# processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
# model = AutoModel.from_pretrained("google/vit-base-patch16-224").to("cpu")

import os
import pickle


# def save_model_and_processor(
#     model, processor, model_dir="model_files", processor_file="processor.pkl"
# ):
#     """
#     Save the model in multiple files and the processor in a single file.

#     Args:
#         model: The model to be saved.
#         processor: The processor to be saved.
#         model_dir: Directory to save model files.
#         processor_file: File name to save the processor.
#     """
#     # Ensure the directory exists
#     os.makedirs(model_dir, exist_ok=True)

#     # Save the processor
#     with open(processor_file, "wb") as f:
#         pickle.dump(processor, f)

#     # Save the model in chunks
#     model_state_dict = model.state_dict()
#     model_bytes = pickle.dumps(model_state_dict)
#     chunk_size = 50 * 1024 * 1024  # 50 MB
#     num_chunks = (len(model_bytes) + chunk_size - 1) // chunk_size

#     for i in range(num_chunks):
#         chunk = model_bytes[i * chunk_size : (i + 1) * chunk_size]
#         with open(os.path.join(model_dir, f"model_chunk_{i}.pkl"), "wb") as f:
#             f.write(chunk)


# # Save the model and processor
# save_model_and_processor(model, processor)


def load_model_and_processor(model_dir="model_files", processor_file="processor.pkl"):
    """
    Load the model from multiple files and the processor from a single file.

    Args:
        model_dir: Directory where model files are saved.
        processor_file: File name where the processor is saved.

    Returns:
        model: The loaded model.
        processor: The loaded processor.
    """
    # Load the processor
    with open(processor_file, "rb") as f:
        processor = pickle.load(f)

    # Load the model chunks
    model_bytes = bytearray()
    for i in range(10):  # Assuming 10 chunks
        try:
            with open(os.path.join(model_dir, f"model_chunk_{i}.pkl"), "rb") as f:
                model_bytes.extend(f.read())
        except FileNotFoundError:
            break

    model_state_dict = pickle.loads(model_bytes)
    model = AutoModel.from_pretrained("google/vit-base-patch16-224")
    model.load_state_dict(model_state_dict)

    return model, processor


# Singleton pattern to ensure model is loaded only once
class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing ModelService...")
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model and processor"""
        try:
            # Load the model and processor
            model, processor = load_model_and_processor()
            logger.info(f"Loading processor from {PROCESSOR_PATH}")
            self.processor = processor

            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = model

            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or processor: {str(e)}")
            raise

    def generate_embedding(self, image: Image.Image) -> List[List[float]]:
        """
        Generate embedding for an image

        Args:
            image: PIL Image object

        Returns:
            List of embedding values (as nested list for proper JSON serialization)
        """
        # Process through the model
        inputs = self.processor(image, return_tensors="pt").to("cpu")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling to get a single embedding vector
        pooler_output = torch.mean(outputs.last_hidden_state, dim=1)

        # Convert to numpy array and then to list for JSON serialization
        embedding_list = pooler_output.detach().cpu().numpy().tolist()

        return embedding_list


# Create a global service instance
model_service = ModelService()
