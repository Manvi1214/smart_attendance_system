import logging
import time
from services.anti_spoofing.face_mesh_ear import FaceMeshEarDetector
from services.anti_spoofing.blink_tracker import AdaptiveBlinkTracker
from services.anti_spoofing.pad_model import FacePresentationAttackDetector

logger = logging.getLogger(__name__)

class SingleBlinkLivenessSession:
    """
    Streamlined Single-Blink Active Challenge + AI Passive Presentation Attack Detection (PAD).
    Enforces the new verification workflow:
    Face Detection -> Eye Blink Verification -> Passive Anti-Spoofing Verification -> Face Recognition -> Attendance Marked.
    """
    def __init__(self, client_id="default"):
        self.client_id = client_id
        self.ear_detector = FaceMeshEarDetector.get_instance()
        self.blink_tracker = AdaptiveBlinkTracker(timeout_seconds=8.0)
        self.face_history = []
        self.reset()
        
    def reset(self):
        """Resets verification buffers for the next student or retry."""
        self.blink_tracker.reset()
        self.face_history = []
        self.blink_passed = False
        self.pad_passed = False
        self.is_completed = False
        self.is_spoof = False
        self.spoof_reason = ""
        self.failure_start_time = None
        self.spoof_start_time = None
        
    def process_frame(self, frame_bgr, detection):
        """
        Processes a single webcam frame through the streamlined verification pipeline.
        Returns:
            is_ready_for_recognition (bool): True ONLY when blink AND passive PAD pass.
            status_text (str): Exact status message matching UI requirements.
            details (dict): Diagnostic metadata, EAR, PAD score, and instructions.
        """
        now = time.time()
        
        if self.is_spoof:
            if self.spoof_start_time is None:
                self.spoof_start_time = now
            elif (now - self.spoof_start_time) > 3.0:
                self.reset()
                return False, "Detecting Face...", {
                    "instruction": "Looking for face",
                    "message": "Scanning for student...",
                    "is_live": False
                }
            return False, "Spoofing Detected", {
                "instruction": "Spoofing Detected",
                "diagnosis": self.spoof_reason,
                "is_live": False
            }
            
        if self.blink_tracker.is_failed:
            if self.failure_start_time is None:
                self.failure_start_time = now
            elif (now - self.failure_start_time) > 1.2:
                self.reset()
                return False, "Please Blink Once", {
                    "instruction": "Please Blink Once",
                    "message": "Looking at camera... Please blink naturally once to verify.",
                    "is_live": False
                }
            return False, "Verification Failed", {
                "instruction": "Verification Failed",
                "message": self.blink_tracker.failure_reason or "Blink not detected. Retrying naturally...",
                "is_live": False
            }
            
        if detection is None or 'box' not in detection:
            return False, "Detecting Face...", {
                "instruction": "Looking for face",
                "message": "Please look directly into camera.",
                "is_live": False
            }
            
        box = detection['box']
        x, y, w, h = box
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face_crop = frame_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0 or w < 40 or h < 40:
            return False, "Face Detected", {
                "instruction": "Face Detected",
                "message": "Please move closer to the camera.",
                "is_live": False
            }
            
        # Keep sliding history of recent frames for temporal anti-replay analysis
        self.face_history.append(face_crop)
        if len(self.face_history) > 6:
            self.face_history.pop(0)

        # -------------------------------------------------------------
        # STEP 1: Eye Blink Verification (Active Liveness Challenge)
        # -------------------------------------------------------------
        if not self.blink_passed:
            left_ear, right_ear, avg_ear = self.ear_detector.calculate_ear(frame_bgr, box)
            
            if avg_ear is None or avg_ear == 0.0:
                return False, "Face Detected", {
                    "instruction": "Face Detected",
                    "message": "Position your face clearly toward camera.",
                    "is_live": False
                }
                
            blink_ok, blink_status, blink_details = self.blink_tracker.update(avg_ear)
            
            if self.blink_tracker.is_failed:
                self.failure_start_time = now
                return False, "Verification Failed", {
                    "instruction": "Verification Failed",
                    "message": blink_details.get("message", "Blink timed out. Please retry."),
                    "ear": avg_ear,
                    "is_live": False
                }
                
            if not blink_ok:
                return False, "Please Blink Once", {
                    "instruction": "Please Blink Once",
                    "message": "Please blink naturally once to verify.",
                    "ear": round(avg_ear, 3),
                    "baseline_ear": round(self.blink_tracker.baseline_ear or 0.28, 3),
                    "time_remaining": blink_details.get("time_remaining", 8.0),
                    "is_live": False
                }
                
            # Blink verified!
            self.blink_passed = True
            logger.info(f"Student blink verified! EAR={avg_ear:.3f}")
            return False, "Blink Verified", {
                "instruction": "Blink Verified",
                "message": "Blink verified! Checking anti-spoofing presentation attack detection...",
                "ear": round(avg_ear, 3),
                "is_live": False
            }

        # -------------------------------------------------------------
        # STEP 2: Passive Anti-Spoofing PAD Verification
        # -------------------------------------------------------------
        if not self.pad_passed:
            is_real, pad_score, diagnosis, pad_details = FacePresentationAttackDetector.evaluate(face_crop, self.face_history)
            
            if not is_real:
                self.is_spoof = True
                self.spoof_reason = diagnosis
                self.spoof_start_time = now
                logger.warning(f"Passive Anti-Spoofing PAD REJECTED: {diagnosis} (PAD Score: {pad_score})")
                return False, "Spoofing Detected", {
                    "instruction": "Spoofing Detected",
                    "diagnosis": diagnosis,
                    "pad_score": pad_score,
                    "pad_details": pad_details,
                    "is_live": False
                }
                
            # Passed passive anti-spoofing!
            self.pad_passed = True
            self.is_completed = True
            logger.info(f"Passive PAD verified! Score: {pad_score:.3f}")
            
        # -------------------------------------------------------------
        # STEP 3: Liveness & Anti-Spoofing Both Confirmed!
        # -------------------------------------------------------------
        return True, "Real Person Verified", {
            "instruction": "Real Person Verified",
            "message": "Blink and Anti-Spoofing checks passed.",
            "is_live": True
        }
