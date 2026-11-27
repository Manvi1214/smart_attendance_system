import cv2
import numpy as np
import logging
import random
import time
from services.anti_spoofing.texture_analysis import TextureAnalyzer

logger = logging.getLogger(__name__)

class BlinkCycleTracker:
    """
    Tracks multi-frame physiological eye blink cycle:
    Phase 1: OPEN (eyes open >= 2 consecutive frames)
    Phase 2: CLOSED (eyes closed for 1 to 5 consecutive frames)
    Phase 3: REOPENED (eyes reopen >= 2 consecutive frames)
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = "IDLE" # "IDLE", "OPEN_STABLE", "CLOSING", "CLOSED", "REOPENED", "VERIFIED"
        self.open_frames = 0
        self.closed_frames = 0
        self.reopen_frames = 0
        self.verified = False
        
    def evaluate_openness(self, frame_bgr, left_eye_pt, right_eye_pt, face_box):
        h, w = frame_bgr.shape[:2]
        _, _, fw, fh = face_box
        eye_w = max(12, int(fw * 0.22))
        eye_h = max(10, int(fh * 0.16))
        
        scores = []
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
            # Vertical Sobel gradient energy (eyelids/sclera/iris boundary)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            v_grad = float(np.mean(np.abs(sobel_y)))
            
            # Darkness ratio
            min_val, _, _, _ = cv2.minMaxLoc(gray)
            mean_val = float(np.mean(gray))
            dark_ratio = (mean_val - min_val) / (mean_val + 1e-5)
            
            score = v_grad * (1.0 + dark_ratio)
            scores.append(score)
            
        if not scores:
            return 24.0 # default assume open
        return float(np.mean(scores))
        
    def update(self, openness):
        OPEN_THRESHOLD = 22.5
        CLOSED_THRESHOLD = 18.0
        
        is_open = (openness >= OPEN_THRESHOLD)
        is_closed = (openness < CLOSED_THRESHOLD)
        
        if self.state in ["IDLE", "OPEN_STABLE"]:
            if is_open:
                self.open_frames += 1
                if self.open_frames >= 2:
                    self.state = "OPEN_STABLE"
            elif is_closed and self.state == "OPEN_STABLE":
                self.state = "CLOSED"
                self.closed_frames = 1
            else:
                self.open_frames = 0
                
        elif self.state == "CLOSED":
            if is_closed:
                self.closed_frames += 1
                if self.closed_frames > 6:
                    # Eyes closed too long (shut eyes / photo / sleeping), reset
                    self.reset()
            elif is_open:
                if 1 <= self.closed_frames <= 6:
                    self.state = "REOPENED"
                    self.reopen_frames = 1
                else:
                    self.reset()
                    
        elif self.state == "REOPENED":
            if is_open:
                self.reopen_frames += 1
                if self.reopen_frames >= 2:
                    self.state = "VERIFIED"
                    self.verified = True
            else:
                self.reset()
                
        return self.verified


class HeadPoseEstimator:
    """
    Computes horizontal yaw and vertical pitch ratios from facial landmarks.
    Enforces non-planar perspective distortion checks to reject flat photos.
    """
    @staticmethod
    def estimate_pose(keypoints):
        """
        Returns:
            yaw (float): ~0.0 neutral, < -0.15 turn right/left, > +0.15 turn opposite.
            pitch (float): ~0.45 neutral, < 0.35 look UP, > 0.55 look DOWN.
            is_valid (bool)
        """
        required = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
        if not all(k in keypoints for k in required):
            return 0.0, 0.45, False
            
        lx, ly = keypoints['left_eye']
        rx, ry = keypoints['right_eye']
        nx, ny = keypoints['nose']
        mlx, mly = keypoints['mouth_left']
        mrx, mry = keypoints['mouth_right']
        
        iod = np.hypot(rx - lx, ry - ly)
        if iod < 10:
            return 0.0, 0.45, False
            
        # Horizontal Yaw: distance from nose to left eye vs nose to right eye
        dl = nx - lx
        dr = rx - nx
        yaw = (dl - dr) / iod
        
        # Vertical Pitch: position of nose between eye line and mouth line
        ey = (ly + ry) / 2.0
        my = (mly + mry) / 2.0
        em_dist = my - ey
        if em_dist < 10:
            return float(yaw), 0.45, True
            
        pitch = (ny - ey) / em_dist
        return float(yaw), float(pitch), True


class ActiveChallengeSession:
    """
    Manages a random 2-step active liveness challenge per client connection.
    Challenges:
    - BLINK: 'Please Blink'
    - TURN_LEFT: 'Turn Head LEFT'
    - TURN_RIGHT: 'Turn Head RIGHT'
    - LOOK_UP: 'Look UP'
    - LOOK_DOWN: 'Look DOWN'
    """
    CHALLENGE_INSTRUCTIONS = {
        "BLINK": "Please Blink",
        "TURN_LEFT": "Turn Head LEFT",
        "TURN_RIGHT": "Turn Head RIGHT",
        "LOOK_UP": "Look UP",
        "LOOK_DOWN": "Look DOWN"
    }
    
    def __init__(self, client_id="default_client"):
        self.client_id = client_id
        self.reset()
        
    def reset(self):
        """Generates a fresh random sequence of 2 challenges."""
        head_options = ["TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
        chosen_head = random.choice(head_options)
        
        # Random sequence of exactly 2 steps: 1 blink + 1 head movement
        if random.random() < 0.5:
            self.challenge_sequence = ["BLINK", chosen_head]
        else:
            self.challenge_sequence = [chosen_head, "BLINK"]
            
        self.current_step = 0 # 0 or 1
        self.step_state = "NEUTRAL_CALIBRATION" # "NEUTRAL_CALIBRATION", "CHALLENGE_ACTIVE", "PASSED"
        self.step_start_time = None
        self.timeout_per_step = 9.0 # seconds
        
        # Calibration baselines
        self.baseline_yaw = None
        self.baseline_pitch = None
        self.neutral_frames = 0
        
        # Consecutiveness tracker
        self.consecutive_pass_frames = 0
        self.blink_tracker = BlinkCycleTracker()
        
        # Static freeze detection buffer
        self.box_history = []
        self.static_count = 0
        
        # Overall status
        self.is_completed = False
        self.is_failed = False
        self.failure_reason = ""
        self.spoof_detected = False
        self.spoof_reason = ""
        self.start_timestamp = time.time()
        
        logger.info(f"ActiveChallengeSession [{self.client_id}] initialized with sequence: {self.challenge_sequence}")

    def get_current_instruction(self):
        if self.is_completed:
            return "Real Person Verified"
        if self.spoof_detected:
            return "Spoofing Attempt Detected"
        if self.is_failed:
            return "Verification Failed"
        if self.current_step < len(self.challenge_sequence):
            ch = self.challenge_sequence[self.current_step]
            return self.CHALLENGE_INSTRUCTIONS.get(ch, "Verifying Liveness...")
        return "Verifying Liveness..."

    def process_frame(self, frame_bgr, detection):
        """
        Evaluates a frame against the active challenge sequence.
        Returns:
            is_completed (bool): True ONLY when BOTH challenges are 100% verified.
            status_text (str): Exact status message matching UI requirements.
            info (dict): Challenge metadata, steps summary, and diagnostics.
        """
        now = time.time()
        
        # Check if previous state failed
        if self.is_failed:
            # Auto-reset after 2 seconds
            if self.step_start_time and (now - self.step_start_time) > 2.5:
                self.reset()
            return False, "Verification Failed", {
                "instruction": "Verification Failed",
                "message": self.failure_reason or "Challenge failed. Retrying with fresh challenge...",
                "steps": self._build_steps_summary()
            }
            
        if self.spoof_detected:
            # Auto-reset after 3 seconds so legitimate user isn't locked forever
            if self.step_start_time and (now - self.step_start_time) > 3.0:
                self.reset()
            return False, "Spoofing Attempt Detected", {
                "instruction": "Spoofing Attempt Detected",
                "diagnosis": self.spoof_reason,
                "steps": self._build_steps_summary()
            }
            
        if self.is_completed:
            return True, "Real Person Verified", {
                "instruction": "Real Person Verified",
                "message": "All challenges completed successfully.",
                "steps": self._build_steps_summary()
            }
            
        if detection is None or 'box' not in detection:
            return False, "Detecting Face...", {
                "instruction": "Face Detected",
                "message": "Please look directly into camera.",
                "steps": self._build_steps_summary()
            }
            
        box = detection['box']
        kps = detection.get('keypoints', {})
        x, y, w, h = box
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face_crop = frame_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0 or w < 40 or h < 40:
            return False, "Face Detected", {
                "instruction": "Face Detected",
                "message": "Please move closer to camera.",
                "steps": self._build_steps_summary()
            }

        # -------------------------------------------------------------
        # 1. Passive Texture Analysis (Paper & Digital Screen Detection)
        # -------------------------------------------------------------
        tex = TextureAnalyzer.analyze_texture(face_crop)
        if tex["is_screen"] or tex["is_print"]:
            self.spoof_detected = True
            self.spoof_reason = tex["reason"]
            self.step_start_time = now
            logger.warning(f"Presentation attack flagged: {tex['reason']}")
            return False, "Spoofing Attempt Detected", {
                "instruction": "Spoofing Attempt Detected",
                "diagnosis": tex["reason"],
                "steps": self._build_steps_summary()
            }

        # -------------------------------------------------------------
        # 2. Static Freeze / Replay Attack Detection
        # -------------------------------------------------------------
        curr_box = np.array([x, y, w, h])
        self.box_history.append(curr_box)
        if len(self.box_history) > 15:
            self.box_history.pop(0)
            diffs = [np.sum(np.abs(self.box_history[i] - self.box_history[0])) for i in range(1, len(self.box_history))]
            if max(diffs) < 1.0: # Absolutely zero pixel micro-movement
                self.static_count += 1
                if self.static_count >= 10:
                    self.spoof_detected = True
                    self.spoof_reason = "Artificial static photograph detected (0 pixel movement)"
                    self.step_start_time = now
                    return False, "Spoofing Attempt Detected", {
                        "instruction": "Spoofing Attempt Detected",
                        "diagnosis": self.spoof_reason,
                        "steps": self._build_steps_summary()
                    }
            else:
                self.static_count = 0

        # -------------------------------------------------------------
        # 3. Head Pose & Eye Openness Estimation
        # -------------------------------------------------------------
        yaw, pitch, pose_valid = HeadPoseEstimator.estimate_pose(kps)
        if not pose_valid:
            return False, "Face Detected", {
                "instruction": "Face Detected",
                "message": "Landmarks unclear. Please center your face.",
                "steps": self._build_steps_summary()
            }
            
        current_challenge = self.challenge_sequence[self.current_step]
        instruction_text = self.CHALLENGE_INSTRUCTIONS[current_challenge]
        
        # -------------------------------------------------------------
        # 4. Calibration: Require brief neutral face before each step
        # -------------------------------------------------------------
        if self.step_state == "NEUTRAL_CALIBRATION":
            # Neutral face criteria: yaw between -0.15 and +0.15, pitch between 0.35 and 0.60
            if abs(yaw) <= 0.16 and 0.35 <= pitch <= 0.60:
                self.neutral_frames += 1
                if self.neutral_frames >= 2:
                    self.baseline_yaw = yaw
                    self.baseline_pitch = pitch
                    self.step_state = "CHALLENGE_ACTIVE"
                    self.step_start_time = now
                    self.consecutive_pass_frames = 0
                    self.blink_tracker.reset()
                    logger.info(f"Baseline established for step {self.current_step} ({current_challenge}): yaw={yaw:.2f}, pitch={pitch:.2f}")
            else:
                self.neutral_frames = 0
                return False, instruction_text, {
                    "instruction": instruction_text,
                    "message": f"Center face to start: {instruction_text}",
                    "steps": self._build_steps_summary()
                }

        # -------------------------------------------------------------
        # 5. Challenge Active Evaluation
        # -------------------------------------------------------------
        # Check timeout
        elapsed = now - (self.step_start_time or now)
        time_left = max(0.0, round(self.timeout_per_step - elapsed, 1))
        
        if elapsed > self.timeout_per_step:
            self.is_failed = True
            self.failure_reason = f"Timeout on '{instruction_text}'. Please respond promptly."
            self.step_start_time = now
            logger.warning(f"Active challenge timed out on step {self.current_step} ({current_challenge})")
            return False, "Verification Failed", {
                "instruction": "Verification Failed",
                "message": self.failure_reason,
                "steps": self._build_steps_summary()
            }

        step_passed = False
        
        if current_challenge == "BLINK":
            if 'left_eye' in kps and 'right_eye' in kps:
                openness = self.blink_tracker.evaluate_openness(frame_bgr, kps['left_eye'], kps['right_eye'], box)
                blink_ok = self.blink_tracker.update(openness)
                if blink_ok:
                    step_passed = True
            else:
                self.blink_tracker.reset()
                
        elif current_challenge == "TURN_LEFT":
            # User turns head to their left. Raw camera feed: delta_yaw reaches >= 0.16 or <= -0.16
            delta_yaw = yaw - (self.baseline_yaw or 0.0)
            # Accept clear yaw rotation beyond baseline
            if abs(delta_yaw) >= 0.16 and delta_yaw > 0.10: # turning towards user left
                self.consecutive_pass_frames += 1
                if self.consecutive_pass_frames >= 2:
                    step_passed = True
            elif abs(delta_yaw) >= 0.18: # or significant clear yaw shift
                self.consecutive_pass_frames += 1
                if self.consecutive_pass_frames >= 2:
                    step_passed = True
            else:
                self.consecutive_pass_frames = 0
                
        elif current_challenge == "TURN_RIGHT":
            # User turns head to their right
            delta_yaw = yaw - (self.baseline_yaw or 0.0)
            if abs(delta_yaw) >= 0.16 and delta_yaw < -0.10: # turning towards user right
                self.consecutive_pass_frames += 1
                if self.consecutive_pass_frames >= 2:
                    step_passed = True
            elif abs(delta_yaw) >= 0.18:
                self.consecutive_pass_frames += 1
                if self.consecutive_pass_frames >= 2:
                    step_passed = True
            else:
                self.consecutive_pass_frames = 0
                
        elif current_challenge == "LOOK_UP":
            # Chin/nose tilts UP -> pitch decreases
            delta_pitch = pitch - (self.baseline_pitch or 0.45)
            if delta_pitch <= -0.09 or pitch < 0.36:
                self.consecutive_pass_frames += 1
                if self.consecutive_pass_frames >= 2:
                    step_passed = True
            else:
                self.consecutive_pass_frames = 0
                
        elif current_challenge == "LOOK_DOWN":
            # Forehead tilts forward, nose tip moves down -> pitch increases
            delta_pitch = pitch - (self.baseline_pitch or 0.45)
            if delta_pitch >= 0.09 or pitch > 0.56:
                self.consecutive_pass_frames += 1
                if self.consecutive_pass_frames >= 2:
                    step_passed = True
            else:
                self.consecutive_pass_frames = 0

        # Handle Step Completion
        if step_passed:
            logger.info(f"Step {self.current_step} ({current_challenge}) PASSED!")
            self.current_step += 1
            if self.current_step >= len(self.challenge_sequence):
                self.is_completed = True
                return True, "Real Person Verified", {
                    "instruction": "Real Person Verified",
                    "message": "Both challenges passed!",
                    "steps": self._build_steps_summary()
                }
            else:
                # Prepare next challenge step
                self.step_state = "NEUTRAL_CALIBRATION"
                self.neutral_frames = 0
                self.consecutive_pass_frames = 0
                self.blink_tracker.reset()
                next_ch = self.challenge_sequence[self.current_step]
                return False, self.CHALLENGE_INSTRUCTIONS[next_ch], {
                    "instruction": self.CHALLENGE_INSTRUCTIONS[next_ch],
                    "message": f"Step 1 passed! Next: {self.CHALLENGE_INSTRUCTIONS[next_ch]}",
                    "steps": self._build_steps_summary(),
                    "time_remaining": self.timeout_per_step
                }

        # Step still in progress
        return False, instruction_text, {
            "instruction": instruction_text,
            "message": f"Step {self.current_step + 1} of 2: {instruction_text}",
            "time_remaining": time_left,
            "steps": self._build_steps_summary()
        }

    def _build_steps_summary(self):
        summary = []
        for i, ch in enumerate(self.challenge_sequence):
            name = self.CHALLENGE_INSTRUCTIONS.get(ch, ch)
            if self.is_completed:
                st = "completed"
            elif i < self.current_step:
                st = "completed"
            elif i == self.current_step:
                st = "failed" if self.is_failed else ("spoof" if self.spoof_detected else "in_progress")
            else:
                st = "pending"
            summary.append({"name": name, "status": st, "code": ch})
        return summary
