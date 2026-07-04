# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import logging
from services.face_recognition.facenet_model import get_embedder

logger = logging.getLogger(__name__)

def generate_embedding(face_bgr):
    """
    Generates a 512-dimensional embedding vector for a BGR face crop.
    Args:
        face_bgr: BGR image crop of a face.
    Returns:
        List of 512 floats representing the face embedding, or None on failure.
    """
    try:
        # Resize to 160x160 required by FaceNet
        face_resized = cv2.resize(face_bgr, (160, 160))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Get FaceNet embedder
        embedder = get_embedder()
        
        # keras-facenet expects batch input [N, H, W, C]
        face_batch = np.expand_dims(face_rgb, axis=0)
        
        # Generate embeddings
        embeddings = embedder.embeddings(face_batch)
        
        # Extract the single embedding vector and convert to list of floats for JSON/Mongo compatibility
        embedding_vector = embeddings[0].tolist()
        
        return embedding_vector
    except Exception as e:
        logger.error(f"Error generating face embedding: {e}")
        return None
