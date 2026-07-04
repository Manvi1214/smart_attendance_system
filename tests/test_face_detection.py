import unittest
import numpy as np
from services.face_detection.mtcnn_detector import get_detector, detect_faces

class TestFaceDetection(unittest.TestCase):
    def test_detector_initialization(self):
        """Test if MTCNN detector can be loaded."""
        try:
            detector = get_detector()
            self.assertIsNotNone(detector, "MTCNN detector failed to initialize.")
        except Exception as e:
            self.fail(f"MTCNN initialization threw an exception: {e}")

    def test_detect_faces_empty_image(self):
        """Test detector on a blank/black frame."""
        # Create a black image
        blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
        try:
            faces = detect_faces(blank_image)
            self.assertEqual(len(faces), 0, "Detector should find 0 faces in a blank black image.")
        except Exception as e:
            self.fail(f"detect_faces threw an exception on blank image: {e}")

if __name__ == '__main__':
    unittest.main()
