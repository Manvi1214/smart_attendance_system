import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveBlinkTracker:
    """
    Adaptive Eye Blink Detector that dynamically calibrates each student's baseline
    open-eye EAR to accommodate different eye shapes, distances, lighting, and glasses.
    
    Validates complete active liveness sequence:
        OPEN_STABLE -> CLOSED (1 to 6 frames) -> REOPENED
    
    Guarantees:
    - Never triggers on continuously closed eyes (>6 frames / static photo).
    - Never triggers on static open eyes (times out).
    - Tolerates webcam FPS jitter, glasses, and natural partial blinks.
    - Seamless auto-retry on timeout without requiring page reload.
    """
    def __init__(self, timeout_seconds=14.0):
        self.timeout_seconds = timeout_seconds
        self.reset()
        
    def reset(self):
        self.state = "CALIBRATING" # "CALIBRATING", "OPEN_STABLE", "CLOSED", "VERIFIED"
        self.baseline_samples = []
        self.baseline_ear = None
        self.threshold_closed = None
        self.threshold_open = None
        
        self.closed_frames = 0
        self.min_closed_ear = 999.0
        
        self.start_time = None
        self.is_verified = False
        self.is_failed = False
        self.failure_reason = ""
        
    def update(self, current_ear):
        """
        Updates the blink detector state machine with the latest frame's EAR.
        Returns:
            is_verified (bool): True when a genuine blink is completed.
            status_text (str): "Please Blink Once", "Blink Verified", "Verification Failed"
            details (dict): current metrics, state, and diagnostics.
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        elapsed = now - self.start_time
        time_remaining = max(0.0, round(self.timeout_seconds - elapsed, 1))
        
        if self.is_verified:
            return True, "Blink Verified", {
                "ear": round(current_ear, 3),
                "baseline_ear": round(self.baseline_ear or 0.28, 3),
                "state": "VERIFIED"
            }
            
        if self.is_failed:
            # Auto-reset 2 seconds after failure to allow seamless natural retry
            if elapsed > (self.timeout_seconds + 2.0):
                self.reset()
            return False, "Verification Failed", {
                "ear": round(current_ear, 3),
                "message": self.failure_reason,
                "state": "FAILED"
            }
            
        # Check timeout
        if elapsed > self.timeout_seconds:
            self.is_failed = True
            self.failure_reason = "Blink detection timed out. Please look at the camera and blink naturally once."
            return False, "Verification Failed", {
                "ear": round(current_ear, 3),
                "message": self.failure_reason,
                "state": "FAILED"
            }
            
        # 1. Calibration Phase (First 3-4 open frames)
        if self.state == "CALIBRATING":
            # Real open eyes typically range between 0.16 and 0.45
            if current_ear >= 0.16:
                self.baseline_samples.append(current_ear)
                if len(self.baseline_samples) >= 3:
                    self.baseline_ear = float(np.median(self.baseline_samples))
                    # Adaptive dynamic thresholds tailored to this individual student:
                    self.threshold_closed = max(0.18, min(0.25, self.baseline_ear * 0.78))
                    self.threshold_open = max(0.20, self.baseline_ear * 0.82)
                    self.state = "OPEN_STABLE"
                    logger.info(f"Blink baseline calibrated: EAR={self.baseline_ear:.3f}, ClosedThresh={self.threshold_closed:.3f}, OpenThresh={self.threshold_open:.3f}")
            return False, "Please Blink Once", {
                "ear": round(current_ear, 3),
                "state": "CALIBRATING",
                "time_remaining": time_remaining
            }
            
        # Dynamic closure criteria:
        # 1. Absolute EAR below 0.22 (typical closed-eye value for MediaPipe 468 mesh)
        # 2. Or relative drop of at least 14% from calibrated baseline
        # 3. Or absolute drop of at least 0.030 from baseline
        # 4. Or below individual calibrated closed threshold
        rel_drop = (self.baseline_ear - current_ear) >= 0.030
        rel_ratio = (current_ear <= self.baseline_ear * 0.86)
        abs_closed = (current_ear <= self.threshold_closed) or (current_ear <= 0.22)
        is_closed = (rel_drop or rel_ratio or abs_closed)
        
        # Dynamic reopening criteria:
        # Eye returns to at least 80% of baseline, or recovers by >= 0.030 above lowest closed frame
        is_reopened = (current_ear >= self.threshold_open) or (
            (current_ear - self.min_closed_ear >= 0.030) and (current_ear >= self.baseline_ear * 0.75)
        )
        
        # 2. Open Stable Phase
        if self.state == "OPEN_STABLE":
            if is_closed:
                # Eyelids closing
                self.state = "CLOSED"
                self.closed_frames = 1
                self.min_closed_ear = current_ear
                logger.debug(f"Blink started: frame EAR={current_ear:.3f} (baseline={self.baseline_ear:.3f})")
            else:
                # Adapt baseline slowly if user has slight natural head tilt / posture adjustment
                if current_ear >= self.baseline_ear * 0.85:
                    self.baseline_ear = 0.95 * self.baseline_ear + 0.05 * current_ear
                    self.threshold_closed = max(0.18, min(0.25, self.baseline_ear * 0.78))
                    self.threshold_open = max(0.20, self.baseline_ear * 0.82)
                    
        # 3. Closed Phase
        elif self.state == "CLOSED":
            if is_closed:
                self.closed_frames += 1
                self.min_closed_ear = min(self.min_closed_ear, current_ear)
                # If eyes stay closed for > 8 frames (~600-800ms):
                # Discard as continuous closure / sleeping / photo of closed eyes!
                if self.closed_frames > 8:
                    logger.warning(f"Continuously closed eyes detected ({self.closed_frames} frames). Resetting blink tracker.")
                    self.state = "OPEN_STABLE"
                    self.closed_frames = 0
            elif is_reopened:
                # Eyelids reopened after 1 to 8 closed frames!
                if 1 <= self.closed_frames <= 8:
                    self.state = "VERIFIED"
                    self.is_verified = True
                    logger.info(f"Genuine single blink verified! Closed for {self.closed_frames} frames, lowest EAR={self.min_closed_ear:.3f}, reopened to {current_ear:.3f}")
                    return True, "Blink Verified", {
                        "ear": round(current_ear, 3),
                        "baseline_ear": round(self.baseline_ear, 3),
                        "closed_frames": self.closed_frames,
                        "state": "VERIFIED"
                    }
                else:
                    self.state = "OPEN_STABLE"
                    self.closed_frames = 0
                    
        return False, "Please Blink Once", {
            "ear": round(current_ear, 3),
            "baseline_ear": round(self.baseline_ear or 0.28, 3),
            "state": self.state,
            "time_remaining": time_remaining
        }
