import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.anti_spoofing.face_mesh_ear import FaceMeshEarDetector
from services.anti_spoofing.blink_tracker import AdaptiveBlinkTracker
from services.anti_spoofing.pad_model import FacePresentationAttackDetector
from services.anti_spoofing.liveness_verifier import SingleBlinkLivenessSession

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

def test_real_person_natural_blink():
    print("==================================================")
    print("TEST 1: REAL PERSON NATURAL EYE BLINK")
    print("==================================================")
    tracker = AdaptiveBlinkTracker(timeout_seconds=6.0)
    
    # 1. Open eyes calibration (EAR ~ 0.34)
    for _ in range(4):
        ok, status, _ = tracker.update(0.34)
        assert not ok
        assert status == "Please Blink Once"
    assert tracker.state == "OPEN_STABLE"
    assert tracker.baseline_ear is not None
    print(f"Calibration successful: Baseline EAR={tracker.baseline_ear:.3f}, ClosedThresh={tracker.threshold_closed:.3f}")

    # 2. Eyes closed during blink (EAR drops to 0.12 for 2 frames)
    for _ in range(2):
        ok, status, _ = tracker.update(0.12)
        assert not ok
        assert status == "Please Blink Once"
    assert tracker.state == "CLOSED"
    print("Eyes closed phase registered (2 frames).")

    # 3. Eyes reopen (EAR rises back to 0.34)
    ok, status, details = tracker.update(0.34)
    assert ok == True
    assert status == "Blink Verified"
    print(f"Blink verification complete: status='{status}', details={details}")
    print("✓ Real Person Natural Blink Test PASSED")

def test_continuously_closed_eyes_rejection():
    print("\n==================================================")
    print("TEST 2: CONTINUOUSLY CLOSED EYES (NOT A BLINK)")
    print("==================================================")
    tracker = AdaptiveBlinkTracker(timeout_seconds=5.0)
    # Calibrate open
    for _ in range(4):
        tracker.update(0.32)
        
    # Eyes closed for 10 consecutive frames (sleeping / photo of closed eyes)
    for frame_idx in range(10):
        ok, status, _ = tracker.update(0.10)
        assert not ok, "Continuously closed eyes must NEVER be verified as a blink!"
        
    print(f"After 10 closed frames: is_verified={tracker.is_verified}, state='{tracker.state}'")
    assert tracker.is_verified == False
    print("✓ Continuously Closed Eyes Correctly Rejected")

def test_static_photograph_rejection():
    print("\n==================================================")
    print("TEST 3: STATIC PRINTED PHOTOGRAPH")
    print("==================================================")
    session = SingleBlinkLivenessSession("test_photo")
    frame = np.full((480, 640, 3), (120, 150, 200), dtype=np.uint8)
    real_face = create_synthetic_face(160, 160)
    frame[100:260, 200:360] = real_face
    
    detection = {'box': [200, 100, 160, 160]}
    
    # Static photo presented: constant open eyes, no blink
    for _ in range(5):
        ok, status, details = session.process_frame(frame, detection)
        assert ok == False
        assert status == "Please Blink Once"
        
    # Simulate timeout
    session.blink_tracker.start_time = time.time() - 10.0
    ok, status, details = session.process_frame(frame, detection)
    assert ok == False
    assert status == "Verification Failed"
    print(f"Static photo result: is_ready={ok}, status='{status}', message='{details.get('message')}'")
    print("✓ Static Printed Photograph Rejected by Blink Timeout")

def test_mobile_screen_photo_and_video():
    print("\n==================================================")
    print("TEST 4: MOBILE SCREEN ATTACK & REPLAY VIDEO")
    print("==================================================")
    # Screen face with Moiré pattern and specular glare
    screen_crop = create_synthetic_face(160, 160, is_screen=True)
    is_real, pad_score, diagnosis, details = FacePresentationAttackDetector.evaluate(screen_crop)
    
    print(f"Screen PAD evaluation: is_real={is_real}, pad_score={pad_score}, diagnosis='{diagnosis}'")
    assert is_real == False, "Mobile screen presentation must be rejected!"
    assert details["is_screen"] == True or details["glare_ratio"] > 0.04
    print("✓ Mobile Screen & Glare Attack Immediately Flagged")

    # Paper print flat blur
    print_crop = create_synthetic_face(160, 160, is_printed=True)
    is_real_p, pad_score_p, diag_p, details_p = FacePresentationAttackDetector.evaluate(print_crop)
    print(f"Paper print PAD evaluation: is_real={is_real_p}, pad_score={pad_score_p}, diagnosis='{diag_p}'")
    assert is_real_p == False, "Paper printout must be rejected!"
    print("✓ Paper Printout PAD Attack Immediately Flagged")

def test_adaptive_calibration_narrow_eyes_glasses():
    print("\n==================================================")
    print("TEST 5: ADAPTIVE CALIBRATION (NARROW EYES / GLASSES)")
    print("==================================================")
    # User with naturally narrow open eyes (EAR ~ 0.23)
    tracker = AdaptiveBlinkTracker(timeout_seconds=6.0)
    for _ in range(4):
        tracker.update(0.23)
        
    print(f"Narrow eyes baseline: {tracker.baseline_ear:.3f}, Closed Threshold: {tracker.threshold_closed:.3f}")
    assert tracker.threshold_closed < 0.20
    
    # Blink occurs: EAR drops to 0.11
    for _ in range(2):
        tracker.update(0.11)
    for _ in range(2):
        ok, status, _ = tracker.update(0.23)
        
    assert ok == True
    assert status == "Blink Verified"
    print("✓ Adaptive Calibration for Narrow Eyes & Glasses PASSED")

if __name__ == '__main__':
    test_real_person_natural_blink()
    test_continuously_closed_eyes_rejection()
    test_static_photograph_rejection()
    test_mobile_screen_photo_and_video()
    test_adaptive_calibration_narrow_eyes_glasses()
    print("\n==================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS! 🎯")
    print("==================================================")
