# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import logging
from config import Config
from services.anti_spoofing.eye_aspect_ratio import calculate_ear

logger = logging.getLogger(__name__)

# Try importing MediaPipe for accurate EAR tracking
try:
    # pyrefly: ignore [missing-import]
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        mp_face_mesh = mp.solutions.face_mesh
        _mesh_loaded = True
    else:
        _mesh_loaded = False
        logger.warning("MediaPipe solutions face_mesh not available. Blink detection will fall back to landmark variance.")
except Exception as e:
    _mesh_loaded = False
    logger.warning(f"MediaPipe loading failed ({e}). Blink detection will fall back to landmark variance.")

class BlinkDetector:
    def __init__(self):
        self.blink_count = 0
        self.eye_state = "OPEN" # "OPEN" or "CLOSED"
        self.consecutive_closed_frames = 0
        
        # fallback variance trackers
        self.prev_box = None
        self.movement_accumulator = 0.0
        
        if _mesh_loaded:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                logger.error(f"Error loading MediaPipe Face Mesh: {e}")
                self.face_mesh = None
        else:
            self.face_mesh = None

    def check_liveness(self, frame_bgr, detection):
        """
        Determines liveness (returns boolean is_live, details dict)
        by analyzing blinks or coordinate variance in consecutive frames.
        """
        is_live = False
        h, w = frame_bgr.shape[:2]
        
        # 1. MediaPipe EAR-based blink detection (Main approach)
        if self.face_mesh:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # MediaPipe indices for left and right eyes
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]
                    
                    left_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in left_eye_indices]
                    right_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in right_eye_indices]
                    
                    left_ear = calculate_ear(left_eye_pts)
                    right_ear = calculate_ear(right_eye_pts)
                    
                    # Average EAR for both eyes
                    ear = (left_ear + right_ear) / 2.0
                    
                    # Check EAR threshold
                    if ear < Config.EYE_BLINK_THRESHOLD:
                        self.consecutive_closed_frames += 1
                        self.eye_state = "CLOSED"
                    else:
                        if self.consecutive_closed_frames >= Config.CONSECUTIVE_FRAMES_BLINK:
                            self.blink_count += 1
                            logger.info(f"Blink detected! Total blinks: {self.blink_count}")
                        self.consecutive_closed_frames = 0
                        self.eye_state = "OPEN"
                    
                    # Real person has blinking
                    if self.blink_count >= 1:
                        is_live = True
                        
                    return is_live, {
                        "method": "mediapipe_ear",
                        "ear": round(ear, 3),
                        "blink_count": self.blink_count,
                        "eye_state": self.eye_state
                    }
            except Exception as e:
                logger.error(f"Error during MediaPipe liveness check: {e}")
                
        # 2. Bounding Box Variance Fallback
        # (Measures if there are organic movements in the box over frames, 
        # since static printed photographs show 0 or flat coordinate variance)
        box = detection.get('box', None) if detection else None
        if box:
            x, y, wb, hb = box
            curr_box = np.array([x, y, wb, hb])
            
            if self.prev_box is not None:
                diff = np.abs(curr_box - self.prev_box)
                # Accumulate pixel adjustments
                self.movement_accumulator += float(np.sum(diff))
                
                # If there's high variance but not a jump cut (e.g. micro adjustments 2px to 25px per frame)
                # it indicates a live dynamic face rather than a still printout.
                if 2 <= self.movement_accumulator <= 1000:
                    is_live = True
            
            self.prev_box = curr_box
            return is_live, {
                "method": "bbox_variance",
                "movement_score": round(self.movement_accumulator, 1),
                "is_live": is_live
            }

        return True, {"method": "default", "info": "Unable to verify. Liveness assumed."}
        
    def reset(self):
        self.blink_count = 0
        self.consecutive_closed_frames = 0
        self.eye_state = "OPEN"
        self.prev_box = None
        self.movement_accumulator = 0.0
