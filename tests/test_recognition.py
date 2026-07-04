import unittest
import numpy as np
from services.face_recognition.facenet_model import get_embedder
from services.face_recognition.embedding_generator import generate_embedding

class TestFaceRecognition(unittest.TestCase):
    def test_embedder_initialization(self):
        """Test if FaceNet model can be loaded."""
        try:
            embedder = get_embedder()
            self.assertIsNotNone(embedder, "FaceNet model failed to initialize.")
        except Exception as e:
            self.fail(f"FaceNet initialization threw an exception: {e}")

    def test_embedding_generation_shape(self):
        """Test if embedding generator outputs 512 dimensions for an arbitrary crop."""
        # Create a dummy BGR face crop (160x160x3)
        dummy_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        try:
            embedding = generate_embedding(dummy_face)
            self.assertIsNotNone(embedding, "Failed to generate embedding.")
            self.assertEqual(len(embedding), 512, f"Embedding shape must be 512, got {len(embedding)}")
        except Exception as e:
            self.fail(f"generate_embedding threw an exception: {e}")

if __name__ == '__main__':
    unittest.main()
