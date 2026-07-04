import numpy as np
import logging
from config import Config
from database.models import StudentModel

logger = logging.getLogger(__name__)

def compare_embeddings(live_embedding, students=None):
    """
    Compares a live face embedding against all enrolled student embeddings.
    Args:
        live_embedding: List of floats (512-dim).
        students: Optional list of student docs. If None, loaded from database.
    Returns:
        dict: Matched student doc, distance, and confidence, or None if no match.
    """
    if live_embedding is None:
        return None

    # Load students from database if not provided
    if students is None:
        try:
            students = StudentModel.get_all()
        except Exception as e:
            logger.error(f"Error loading students for comparison: {e}")
            return None

    if not students:
        return None

    live_emb_arr = np.array(live_embedding)
    best_match = None
    min_distance = float('inf')

    for student in students:
        # Check if student has a valid embedding
        if 'embedding' not in student or not student['embedding']:
            continue
            
        student_emb_arr = np.array(student['embedding'])
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(live_emb_arr - student_emb_arr)
        
        if distance < min_distance:
            min_distance = distance
            best_match = student

    threshold = Config.RECOGNITION_THRESHOLD
    
    if best_match and min_distance <= threshold:
        # Calculate a human-readable confidence percentage
        # FaceNet threshold is 0.6. If distance is 0.0, confidence is 100%.
        # If distance is equal to threshold, confidence is 60%.
        # Formula: confidence = max(0.0, 1.0 - (distance * 0.5)) * 100
        confidence = max(0.0, (1.0 - (min_distance / 1.5)) * 100)
        
        return {
            "student": best_match,
            "distance": float(min_distance),
            "confidence": round(confidence, 2)
        }
        
    return None
