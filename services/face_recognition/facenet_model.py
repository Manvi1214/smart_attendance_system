import os
# Force CPU device placement for TensorFlow to avoid MPS crashes on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import logging
# pyrefly: ignore [missing-import]
from keras_facenet import FaceNet

logger = logging.getLogger(__name__)

# Global embedder instance
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Initializing FaceNet model (keras-facenet)...")
        try:
            # Configure local model cache directory in project models folder
            import os
            from config import Config
            model_cache = os.path.join(Config.BASE_DIR, 'models', 'facenet')
            _embedder = FaceNet(cache_folder=model_cache)
            logger.info(f"FaceNet model initialized successfully from local path: {model_cache}")
        except Exception as e:
            logger.error(f"Error loading FaceNet model: {e}")
            raise e
    return _embedder
