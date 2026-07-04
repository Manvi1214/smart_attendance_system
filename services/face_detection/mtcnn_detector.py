import os
# Force CPU device placement for TensorFlow to avoid MPS crashes on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# pyrefly: ignore [missing-import]
import cv2
import logging
# pyrefly: ignore [missing-import]
from mtcnn import MTCNN

logger = logging.getLogger(__name__)

# Initialize MTCNN detector as a singleton/module-level variable
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        logger.info("Initializing MTCNN Face Detector...")
        try:
            _detector = MTCNN()
            logger.info("MTCNN Face Detector initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MTCNN detector: {e}")
            raise e
    return _detector

def detect_faces(image_bgr):
    """
    Detects faces in a BGR image using MTCNN.
    Returns:
        List of dicts, each containing 'box' (x, y, w, h), 'confidence', and 'keypoints'.
    """
    detector = get_detector()
    if detector is None:
        return []
    
    try:
        # MTCNN requires RGB image format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        return faces
    except Exception as e:
        logger.error(f"Error during face detection: {e}")
        return []
