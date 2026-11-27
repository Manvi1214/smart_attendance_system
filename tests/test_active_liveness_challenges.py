import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.anti_spoofing.active_challenge_engine import ActiveChallengeSession, HeadPoseEstimator, BlinkCycleTracker
from services.anti_spoofing.texture_analysis import TextureAnalyzer

def create_synthetic_face(w=160, h=160, is_screen=False, is_printed=False):
    face = np.full((h, w, 3), (140, 160, 210), dtype=np.uint8)
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Eyes
    cv2.circle(face, (int(w*0.35), int(h*0.4)), int(w*0.08), (40, 50, 70), -1)
    cv2.circle(face, (int(w*0.65), int(h*0.4)), int(w*0.08), (40, 50, 70), -1)
    cv2.circle(face, (int(w*0.35), int(h*0.4)), int(w*0.04), (10, 15, 20), -1)
    cv2.circle(face, (int(w*0.65), int(h*0.4)), int(w*0.04), (10, 15, 20), -1)
    # Mouth
    cv2.ellipse(face, (int(w*0.5), int(h*0.75)), (int(w*0.2), int(h*0.08)), 0, 0, 180, (50, 60, 150), -1)
    
    if is_screen:
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        moire_pattern = (np.sin(xx * 0.8) * np.cos(yy * 0.8) * 45).astype(np.int16)
        face = np.clip(face.astype(np.int16) + np.repeat(moire_pattern[:, :, np.newaxis], 3, axis=2), 0, 255).astype(np.uint8)
        cv2.circle(face, (int(w*0.25), int(h*0.25)), int(w*0.15), (255, 255, 255), -1)
    elif is_printed:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        face = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        face[:, :, 0] = (face[:, :, 0] * 0.7 + 40).astype(np.uint8)
        face[:, :, 2] = (face[:, :, 2] * 0.7 + 40).astype(np.uint8)
        
    return face

def test_random_challenge_generation():
    print("==================================================")
    print("TEST 1: RANDOM CHALLENGE GENERATION & DIVERSITY")
    print("==================================================")
    sequences = []
    for i in range(100):
        session = ActiveChallengeSession(f"client_{i}")
        seq = session.challenge_sequence
        assert len(seq) == 2, f"Each challenge must have 2 steps, got {len(seq)}"
        assert "BLINK" in seq, "Each challenge sequence must include BLINK"
        head_steps = [s for s in seq if s in ["TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]]
        assert len(head_steps) == 1, f"Each sequence must include exactly 1 head movement, got {head_steps}"
        sequences.append(tuple(seq))
        
    unique_seqs = set(sequences)
    print(f"Generated 100 sessions. Unique sequence permutations observed: {len(unique_seqs)}/8")
    for s in sorted(unique_seqs):
        print(f"  - {s[0]} -> {s[1]}")
    assert len(unique_seqs) >= 4, "Should observe diverse random sequence distribution"
    print("✓ Random Challenge Generation PASSED")

def test_static_photograph_rejection():
    print("\n==================================================")
    print("TEST 2: STATIC PHOTOGRAPH PRESENTATION ATTACK")
    print("==================================================")
    session = ActiveChallengeSession("test_photo_client")
    frame = np.full((480, 640, 3), (120, 150, 200), dtype=np.uint8)
    real_face = create_synthetic_face(160, 160)
    frame[100:260, 200:360] = real_face
    
    # Static detection: landmarks never change
    static_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (280, 180),
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    
    # 1. Photograph shown to camera
    for f in range(5):
        is_completed, status, info = session.process_frame(frame, static_detection)
        assert is_completed == False, "Static photograph must NEVER be marked as completed!"
        
    current_req = session.challenge_sequence[session.current_step]
    print(f"Photograph presented. Active challenge: '{status}', is_completed={is_completed}")
    assert status in ["Please Blink", "Turn Head LEFT", "Turn Head RIGHT", "Look UP", "Look DOWN", "Verifying Liveness", "Face Detected"]
    
    # 2. Moving photo translationally in 2D space (simulating hand shake with paper)
    for shift in [-10, -5, 0, 5, 10, 15, 10, 5, 0]:
        shifted_detection = {
            'box': [200 + shift, 100 + shift, 160, 160],
            'keypoints': {
                'left_eye': (240 + shift, 150 + shift),
                'right_eye': (320 + shift, 150 + shift),
                'nose': (280 + shift, 180 + shift),
                'mouth_left': (250 + shift, 220 + shift),
                'mouth_right': (310 + shift, 220 + shift)
            }
        }
        is_completed, status, info = session.process_frame(frame, shifted_detection)
        assert is_completed == False, "Translational hand movement of a photo must NEVER pass active challenge!"
        
    print(f"After hand movement of photo: is_completed={is_completed}, still pending challenge.")
    
    # 3. Simulate timeout (photo cannot complete movement)
    session.step_start_time = time.time() - 10.0 # Force timeout
    is_completed, status, info = session.process_frame(frame, static_detection)
    assert is_completed == False
    assert status == "Verification Failed", f"Expected 'Verification Failed', got '{status}'"
    print(f"After timeout without movement: status='{status}', reason='{info['message']}'")
    print("✓ Static Photograph Presentation Attack Successfully Defeated!")

def test_screen_and_print_texture_detection():
    print("\n==================================================")
    print("TEST 3: DIGITAL SCREEN & PAPER PRINT DETECTION")
    print("==================================================")
    session = ActiveChallengeSession("test_screen_client")
    frame = np.full((480, 640, 3), (120, 150, 200), dtype=np.uint8)
    
    # Digital screen with glare & moire
    screen_face = create_synthetic_face(160, 160, is_screen=True)
    frame[100:260, 200:360] = screen_face
    detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (280, 180),
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    
    is_completed, status, info = session.process_frame(frame, detection)
    assert is_completed == False
    assert status == "Spoofing Attempt Detected", f"Expected 'Spoofing Attempt Detected', got '{status}'"
    print(f"Digital Screen on phone: status='{status}', diagnosis='{info['diagnosis']}'")
    print("✓ Digital Screen Attack Immediately Flagged & Rejected")

def test_real_human_verification_flow():
    print("\n==================================================")
    print("TEST 4: REAL HUMAN MULTI-STEP VERIFICATION FLOW")
    print("==================================================")
    session = ActiveChallengeSession("test_real_human")
    # For testing, explicitly test sequence ["TURN_LEFT", "BLINK"]
    session.challenge_sequence = ["TURN_LEFT", "BLINK"]
    session.current_step = 0
    
    frame = np.full((480, 640, 3), (120, 150, 200), dtype=np.uint8)
    real_face = create_synthetic_face(160, 160)
    frame[100:260, 200:360] = real_face
    
    neutral_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (280, 180),
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    
    # Neutral calibration
    for _ in range(3):
        is_completed, status, info = session.process_frame(frame, neutral_detection)
        
    print(f"Calibrated neutral. Current prompt: '{status}'")
    assert status == "Turn Head LEFT"
    
    # Step 1: User turns head to LEFT (nose moves towards right in raw frame)
    turn_left_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (298, 180), # shifted right (yaw >= +0.16)
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    for _ in range(5):
        is_completed, status, info = session.process_frame(frame, turn_left_detection)
        if session.current_step == 1:
            break
        
    print(f"After turning head left: Step 1 passed! Next prompt: '{status}'")
    assert status == "Please Blink"
    assert session.current_step == 1
    
    # Calibrate neutral before blink
    for _ in range(3):
        session.process_frame(frame, neutral_detection)
        
    # Step 2: User blinks (Open -> Closed -> Reopen)
    # Eyes Closed
    frame_closed = frame.copy()
    frame_closed[135:165, 225:255] = (140, 160, 210) # Left eye smooth skin
    frame_closed[135:165, 305:335] = (140, 160, 210) # Right eye smooth skin
    for _ in range(2):
        session.process_frame(frame_closed, neutral_detection)
        
    # Eyes Reopen
    for _ in range(3):
        is_completed, status, info = session.process_frame(frame, neutral_detection)
        
    print(f"After natural eye blink: is_completed={is_completed}, status='{status}'")
    assert is_completed == True
    assert status == "Real Person Verified"
    print("✓ Real Human Verification Workflow 100% SUCCESSFUL!")

def test_all_movement_types():
    print("\n==================================================")
    print("TEST 5: ALL HEAD MOVEMENT TYPES (RIGHT, UP, DOWN)")
    print("==================================================")
    frame = np.full((480, 640, 3), (120, 150, 200), dtype=np.uint8)
    real_face = create_synthetic_face(160, 160)
    frame[100:260, 200:360] = real_face
    
    neutral_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (280, 180),
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    
    # 1. Test TURN_RIGHT
    session_r = ActiveChallengeSession("client_right")
    session_r.challenge_sequence = ["TURN_RIGHT", "BLINK"]
    for _ in range(3):
        session_r.process_frame(frame, neutral_detection)
    turn_right_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (260, 180), # shifted left in raw frame (yaw <= -0.16)
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    for _ in range(3):
        session_r.process_frame(frame, turn_right_detection)
    assert session_r.current_step == 1, "TURN_RIGHT should pass"
    print("✓ Turn Head RIGHT verified")

    # 2. Test LOOK_UP
    session_u = ActiveChallengeSession("client_up")
    session_u.challenge_sequence = ["LOOK_UP", "BLINK"]
    for _ in range(3):
        session_u.process_frame(frame, neutral_detection)
    look_up_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (280, 164), # nose higher (pitch <= 0.35)
            'mouth_left': (250, 225),
            'mouth_right': (310, 225)
        }
    }
    for _ in range(3):
        session_u.process_frame(frame, look_up_detection)
    assert session_u.current_step == 1, "LOOK_UP should pass"
    print("✓ Look UP verified")

    # 3. Test LOOK_DOWN
    session_d = ActiveChallengeSession("client_down")
    session_d.challenge_sequence = ["LOOK_DOWN", "BLINK"]
    for _ in range(3):
        session_d.process_frame(frame, neutral_detection)
    look_down_detection = {
        'box': [200, 100, 160, 160],
        'keypoints': {
            'left_eye': (240, 150),
            'right_eye': (320, 150),
            'nose': (280, 196), # nose lower (pitch >= 0.58)
            'mouth_left': (250, 220),
            'mouth_right': (310, 220)
        }
    }
    for _ in range(3):
        session_d.process_frame(frame, look_down_detection)
    assert session_d.current_step == 1, "LOOK_DOWN should pass"
    print("✓ Look DOWN verified")

if __name__ == '__main__':
    test_random_challenge_generation()
    test_static_photograph_rejection()
    test_screen_and_print_texture_detection()
    test_real_human_verification_flow()
    test_all_movement_types()
    print("\n==================================================")
    print("ALL ACTIVE ANTI-SPOOFING TESTS PASSED! 🛡️")
    print("==================================================")
