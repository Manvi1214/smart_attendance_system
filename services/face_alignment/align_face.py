# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def align_face(image_bgr, detection):
    """
    Aligns a face using eye coordinates from MTCNN keypoints and applies rotation.
    Args:
        image_bgr: The original BGR frame.
        detection: A single detection dict from MTCNN.
    Returns:
        Aligned and cropped BGR image of the face.
    """
    try:
        box = detection['box']
        keypoints = detection['keypoints']
        
        # Extract eye coordinates
        left_eye = keypoints['left_eye']   # (x, y)
        right_eye = keypoints['right_eye'] # (x, y)
        
        # Calculate angle of rotation
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        
        # Avoid division by zero
        if dx == 0:
            angle = 0.0
        else:
            angle = np.degrees(np.arctan2(dy, dx))
            
        # Get midpoint between the eyes
        eye_center = (
            int((left_eye[0] + right_eye[0]) // 2),
            int((left_eye[1] + right_eye[1]) // 2)
        )
        
        # Get rotation matrix
        h, w = image_bgr.shape[:2]
        M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        
        # Warp the entire image
        rotated = cv2.warpAffine(image_bgr, M, (w, h))
        
        # Recalculate bounding box in the rotated frame
        # For simplicity, we can crop using the original box coordinates on the rotated image,
        # or expand it slightly to prevent boundary clippings
        x, y, w_box, h_box = box
        
        # Make coordinates positive/within image limits
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(w, x_start + w_box)
        y_end = min(h, y_start + h_box)
        
        # Crop the face
        face_crop = rotated[y_start:y_end, x_start:x_end]
        
        # If crop is valid, return it
        if face_crop.size > 0:
            return face_crop
            
    except Exception as e:
        logger.error(f"Error during face alignment: {e}. Falling back to standard crop.")
        
    # Fallback to direct crop if alignment fails
    x, y, w_box, h_box = detection['box']
    h, w = image_bgr.shape[:2]
    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(w, x_start + w_box), min(h, y_start + h_box)
    return image_bgr[y_start:y_end, x_start:x_end]
