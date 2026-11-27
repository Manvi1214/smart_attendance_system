import cv2
import numpy as np
import logging
import time
from services.anti_spoofing.texture_analysis import TextureAnalyzer

logger = logging.getLogger(__name__)

class MultiFactorLivenessDetector:
    """
    Comprehensive multi-factor face anti-spoofing and liveness detector.
    Protects against:
    1. Printed photograph presentation attacks (paper texture, flat 2D plane).
    2. Digital screen replay attacks (mobile/tablet LCD/OLED screen Moiré pattern & specular glare).
    3. Static image presentation attacks (lack of physiological blinking and 3D parallax).
    
    Combines:
    - Passive frequency domain & chrominance texture analysis.
    - Active physiological eye blink detection.
    - Active 3D head movement and yaw/pitch perspective tracking.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Resets all tracking buffers for a new person / verification attempt."""
        self.blink_count = 0
        self.eye_state = "OPEN" # "OPEN" or "CLOSED"
        self.consecutive_closed_frames = 0
        self.blink_verified = False
        
        # Head movement history (sliding window)
        self.yaw_history = []
        self.pitch_history = []
        self.head_movement_verified = False
        
        # Bounding box & landmark variation tracker
        self.box_history = []
        self.static_frame_count = 0
        
        # Texture score history
        self.texture_scores = []
        self.spoof_detected = False
        self.spoof_reason = ""
        
        self.frames_analyzed = 0
        self.verified_timestamp = 0

    def evaluate_eye_openness(self, frame_bgr, left_eye_pt, right_eye_pt, face_box):
        """
        Evaluates eye openness based on vertical gradient energy and pupil intensity.
        Open eye has strong vertical gradient (pupil/iris against sclera) and dark center.
        Closed eyelid has uniform smooth skin and low vertical gradient.
        """
        h, w = frame_bgr.shape[:2]
        _, _, fw, fh = face_box
        eye_w = max(10, int(fw * 0.22))
        eye_h = max(8, int(fh * 0.16))
        
        openness_scores = []
        
        for pt in [left_eye_pt, right_eye_pt]:
            ex, ey = int(pt[0]), int(pt[1])
            x1 = max(0, ex - eye_w // 2)
            y1 = max(0, ey - eye_h // 2)
            x2 = min(w, x1 + eye_w)
            y2 = min(h, y1 + eye_h)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Vertical gradient (Sobel Y)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            v_grad = float(np.mean(np.abs(sobel_y)))
            
            # Central pupil darkness vs surrounding skin
            min_val, _, _, _ = cv2.minMaxLoc(gray)
            mean_val = float(np.mean(gray))
            darkness_ratio = (mean_val - min_val) / (mean_val + 1e-5)
            
            # Combined openness metric
            metric = v_grad * (1.0 + darkness_ratio)
            openness_scores.append(metric)
            
        if not openness_scores:
            return 25.0 # default assume open
            
        return float(np.mean(openness_scores))

    def check_liveness(self, frame_bgr, detection):
        """
        Main multi-factor liveness assessment.
        Args:
            frame_bgr: Full BGR video frame.
            detection: Dict with 'box' [x, y, w, h] and 'keypoints' {left_eye, right_eye, nose, mouth_left, mouth_right}.
        Returns:
            is_live (bool): True if verified real living human.
            status (str): UI status string matching requirements:
                          'Detecting Face...', 'Face Detected', 'Verifying Liveness...',
                          'Real Person Verified', 'Spoofing Attempt Detected'.
            details (dict): Diagnostic scores and metrics.
        """
        self.frames_analyzed += 1
        
        if detection is None or 'box' not in detection:
            return False, "Detecting Face...", {"reason": "No face found"}
            
        box = detection['box']
        x, y, w, h = box
        kps = detection.get('keypoints', {})
        
        # Ensure bounding box is within frame boundaries
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face_crop = frame_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return False, "Detecting Face...", {"reason": "Empty crop"}

        # -------------------------------------------------------------
        # Factor 1: Passive Texture & Screen/Print Anti-Spoofing Check
        # -------------------------------------------------------------
        texture_res = TextureAnalyzer.analyze_texture(face_crop)
        self.texture_scores.append(texture_res["score"])
        
        # If digital screen or printed paper is explicitly detected
        if texture_res["is_screen"] or texture_res["is_print"]:
            self.spoof_detected = True
            self.spoof_reason = texture_res["reason"]
            return False, "Spoofing Attempt Detected", {
                "spoof_type": "Digital Screen / Printed Photo" if texture_res["is_screen"] else "Paper Printout",
                "diagnosis": texture_res["reason"],
                "texture_score": texture_res["score"],
                "fft_ratio": texture_res["fft_ratio"],
                "lap_var": texture_res["lap_var"]
            }

        # -------------------------------------------------------------
        # Factor 2: Physiological Blink Detection (Eyes Open/Close)
        # -------------------------------------------------------------
        if 'left_eye' in kps and 'right_eye' in kps:
            openness = self.evaluate_eye_openness(frame_bgr, kps['left_eye'], kps['right_eye'], box)
            
            # Openness thresholds: Open typically > 25.0, Closed typically < 19.0
            OPEN_THRESHOLD = 23.0
            CLOSED_THRESHOLD = 18.5
            
            if openness < CLOSED_THRESHOLD:
                self.consecutive_closed_frames += 1
                self.eye_state = "CLOSED"
            elif openness >= OPEN_THRESHOLD:
                # If eyes were closed for 1 to 4 frames and reopened, register a legitimate blink
                if 1 <= self.consecutive_closed_frames <= 5:
                    self.blink_count += 1
                    self.blink_verified = True
                    logger.info(f"Verified natural blink cycle! Total: {self.blink_count}")
                self.consecutive_closed_frames = 0
                self.eye_state = "OPEN"
        else:
            openness = 25.0

        # -------------------------------------------------------------
        # Factor 3: 3D Head Movement & Perspective Parallax Tracking
        # -------------------------------------------------------------
        if 'left_eye' in kps and 'right_eye' in kps and 'nose' in kps and 'mouth_left' in kps and 'mouth_right' in kps:
            lx, ly = kps['left_eye']
            rx, ry = kps['right_eye']
            nx, ny = kps['nose']
            mlx, mly = kps['mouth_left']
            mrx, mry = kps['mouth_right']
            
            iod = np.hypot(rx - lx, ry - ly)
            dl = np.hypot(nx - lx, ny - ly)
            dr = np.hypot(nx - rx, ny - ry)
            
            # Horizontal Yaw ratio
            yaw = (dl - dr) / (iod + 1e-6)
            
            # Vertical Pitch ratio
            ey = (ly + ry) / 2.0
            my = (mly + mry) / 2.0
            pitch = (ny - ey) / (my - ey + 1e-6)
            
            self.yaw_history.append(float(yaw))
            self.pitch_history.append(float(pitch))
            if len(self.yaw_history) > 12:
                self.yaw_history.pop(0)
                self.pitch_history.pop(0)
                
            if len(self.yaw_history) >= 4:
                yaw_range = max(self.yaw_history) - min(self.yaw_history)
                pitch_range = max(self.pitch_history) - min(self.pitch_history)
                
                # If head turned horizontally (yaw variation >= 0.12) or nodded (pitch variation >= 0.10)
                if yaw_range >= 0.12 or pitch_range >= 0.10:
                    self.head_movement_verified = True
                    logger.info(f"Verified natural 3D head movement (YawRange: {yaw_range:.2f}, PitchRange: {pitch_range:.2f})")
        else:
            yaw = 0.0
            pitch = 0.5

        # -------------------------------------------------------------
        # Factor 4: Static Image / Freeze Replay Attack Detection
        # -------------------------------------------------------------
        # If bounding box coordinates have 0.0 pixel variation for > 15 consecutive frames
        curr_box = np.array([x, y, w, h])
        self.box_history.append(curr_box)
        if len(self.box_history) > 15:
            self.box_history.pop(0)
            diffs = [np.sum(np.abs(self.box_history[i] - self.box_history[0])) for i in range(1, len(self.box_history))]
            if max(diffs) < 1.0: # Absolutely zero movement
                self.static_frame_count += 1
                if self.static_frame_count >= 10:
                    return False, "Spoofing Attempt Detected", {
                        "spoof_type": "Static Image / Video Freeze",
                        "diagnosis": "Artificial static image detected without human micro-movement",
                        "texture_score": texture_res["score"]
                    }
            else:
                self.static_frame_count = 0

        # -------------------------------------------------------------
        # Decision Synthesis
        # -------------------------------------------------------------
        is_live = False
        details = {
            "texture_passed": texture_res["passed"],
            "texture_score": texture_res["score"],
            "blink_count": self.blink_count,
            "blink_verified": self.blink_verified,
            "eye_state": self.eye_state,
            "movement_verified": self.head_movement_verified,
            "frames": self.frames_analyzed
        }
        
        # A person is certified real living person if:
        # 1. Texture check passes (no screen Moiré or paper print)
        # 2. AND at least ONE active physiological liveness check passed:
        #    - Verified natural blink cycle, OR
        #    - Verified natural 3D head turn/movement
        if texture_res["passed"] and (self.blink_verified or self.head_movement_verified):
            is_live = True
            status = "Real Person Verified"
            details["method"] = "Blink verified" if self.blink_verified else "3D Head Movement verified"
        else:
            is_live = False
            status = "Verifying Liveness..."
            details["instruction"] = "Please blink naturally or turn your head slightly to complete verification."
            
        return is_live, status, details
