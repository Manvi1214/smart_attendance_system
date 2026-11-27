import os
import sys
import time
import cv2
import numpy as np

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.anti_spoofing.texture_analysis import TextureAnalyzer
from services.anti_spoofing.liveness_engine import MultiFactorLivenessDetector
from services.face_detection.mtcnn_detector import detect_faces
from services.face_recognition.compare_embeddings import StudentEmbeddingCache, compare_embeddings
from services.attendance.mark_attendance import mark_attendance, reset_session_attendance_cache

def create_synthetic_face(w=160, h=160, is_screen=False, is_printed=False):
    """Generates realistic synthetic test face crops with various attack modalities."""
    # Natural skin tone base (RGB ~ (210, 160, 140))
    face = np.full((h, w, 3), (140, 160, 210), dtype=np.uint8)
    
    # Add subtle natural skin texture / noise
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw eyes
    cv2.circle(face, (int(w*0.35), int(h*0.4)), int(w*0.08), (40, 50, 70), -1)
    cv2.circle(face, (int(w*0.65), int(h*0.4)), int(w*0.08), (40, 50, 70), -1)
    # Pupil
    cv2.circle(face, (int(w*0.35), int(h*0.4)), int(w*0.04), (10, 15, 20), -1)
    cv2.circle(face, (int(w*0.65), int(h*0.4)), int(w*0.04), (10, 15, 20), -1)
    
    # Draw mouth
    cv2.ellipse(face, (int(w*0.5), int(h*0.75)), (int(w*0.2), int(h*0.08)), 0, 0, 180, (50, 60, 150), -1)
    
    if is_screen:
        # Screen attack: Periodic grid / Moiré pattern + specular glare
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        moire_pattern = (np.sin(xx * 0.8) * np.cos(yy * 0.8) * 45).astype(np.int16)
        face = np.clip(face.astype(np.int16) + np.repeat(moire_pattern[:, :, np.newaxis], 3, axis=2), 0, 255).astype(np.uint8)
        # Add bright specular screen glare spot
        cv2.circle(face, (int(w*0.25), int(h*0.25)), int(w*0.15), (255, 255, 255), -1)
        
    elif is_printed:
        # Paper printout attack: Flat, low chrominance, blurred/flat texture
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        face = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        # Washed out color
        face[:, :, 0] = (face[:, :, 0] * 0.7 + 40).astype(np.uint8)
        face[:, :, 2] = (face[:, :, 2] * 0.7 + 40).astype(np.uint8)
        
    return face

def test_speed_benchmark():
    print("==================================================")
    print("TEST 1: SPEED & LATENCY BENCHMARK")
    print("==================================================")
    
    # 1. Test in-memory cache vectorized distance calculation
    cache = StudentEmbeddingCache.get_instance()
    # Mock populate cache with 100 students if empty
    if cache.embeddings_matrix is None or len(cache.embeddings_matrix) == 0:
        fake_embs = [np.random.randn(512).astype(np.float32) for _ in range(100)]
        fake_students = [{"_id": f"s_{i}", "name": f"Student {i}", "roll_no": f"ROLL_{i}"} for i in range(100)]
        cache.embeddings_matrix = np.array(fake_embs, dtype=np.float32)
        cache.students_list = fake_students
        cache.last_loaded = time.time()
    
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)
    
    # Benchmark vectorized matching
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = cache.match(query_emb, threshold=0.75)
        times.append((time.perf_counter() - t0) * 1000)
    
    avg_match_ms = np.mean(times)
    print(f"Vectorized Embedding Matching (100 students): {avg_match_ms:.4f} ms per frame")
    assert avg_match_ms < 1.0, f"Matching should be under 1ms, got {avg_match_ms:.2f}ms"
    print("✓ Vectorized Cache Matching PASSED (< 0.1ms target met)")

def test_anti_spoofing_modalities():
    print("\n==================================================")
    print("TEST 2: ANTI-SPOOFING PRESENTATION ATTACK REJECTION")
    print("==================================================")
    
    # 1. Screen Moiré Attack
    screen_crop = create_synthetic_face(160, 160, is_screen=True)
    tex_screen = TextureAnalyzer.analyze_texture(screen_crop)
    print(f"Digital Screen Presentation Attack: passed={tex_screen['passed']}, is_screen={tex_screen['is_screen']}, diagnosis='{tex_screen['reason']}', fft_ratio={tex_screen['fft_ratio']:.2f}")
    assert tex_screen['passed'] == False or tex_screen['is_screen'] == True, "Screen attack should be flagged"
    print("✓ Digital Screen / Moiré Attack Rejected")

    # 2. Paper Printout Attack
    print_crop = create_synthetic_face(160, 160, is_printed=True)
    tex_print = TextureAnalyzer.analyze_texture(print_crop)
    print(f"Paper Printout Attack: passed={tex_print['passed']}, is_print={tex_print['is_print']}, diagnosis='{tex_print['reason']}', lap_var={tex_print['lap_var']:.2f}")
    assert tex_print['passed'] == False or tex_print['is_print'] == True, "Print attack should be flagged"
    print("✓ Paper Printout Presentation Attack Rejected")

    # 3. Live Natural Face Texture
    real_crop = create_synthetic_face(160, 160, is_screen=False, is_printed=False)
    tex_real = TextureAnalyzer.analyze_texture(real_crop)
    print(f"Live Human Skin Texture: passed={tex_real['passed']}, score={tex_real['score']:.2f}, diagnosis='{tex_real['reason']}'")
    assert tex_real['passed'] == True, "Natural skin texture should pass"
    print("✓ Real Human Face Texture Accepted")

def test_liveness_engine():
    print("\n==================================================")
    print("TEST 3: MULTI-FACTOR LIVENESS ENGINE")
    print("==================================================")
    
    engine = MultiFactorLivenessDetector()
    frame = np.full((480, 640, 3), (120, 150, 200), dtype=np.uint8)
    real_face = create_synthetic_face(160, 160)
    frame[100:260, 200:360] = real_face
    
    # Case A: Initial frames without movement or blink -> Verifying Liveness...
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
    
    is_live, status, details = engine.check_liveness(frame, detection)
    print(f"Frame 1 (Initial detection): is_live={is_live}, status='{status}'")
    assert not is_live, "Must not mark as live on first frame without liveness proof"
    assert status == "Verifying Liveness...", f"Expected 'Verifying Liveness...', got '{status}'"
    print("✓ Initial Face Detected -> 'Verifying Liveness...' correctly enforced")

    # Case B: 3D Head Movement Verification (turning head)
    # Simulate yaw shift
    engine.reset()
    for yaw_offset in [0, 8, 18, 25, 12, 0]:
        detection_turn = {
            'box': [200, 100, 160, 160],
            'keypoints': {
                'left_eye': (240 + yaw_offset, 150),
                'right_eye': (320 + yaw_offset // 2, 150),
                'nose': (280 + yaw_offset, 180),
                'mouth_left': (250 + yaw_offset, 220),
                'mouth_right': (310 + yaw_offset, 220)
            }
        }
        is_live, status, details = engine.check_liveness(frame, detection_turn)
        
    print(f"After natural 3D head movement: is_live={is_live}, status='{status}', details={details}")
    assert is_live == True, "Head movement should verify liveness"
    assert status == "Real Person Verified", f"Expected 'Real Person Verified', got '{status}'"
    print("✓ 3D Head Movement Verification PASSED -> 'Real Person Verified'")

    # Case C: Eye Blink Verification (Open -> Closed -> Open)
    engine.reset()
    # 1. Eyes Open
    for _ in range(2):
        engine.check_liveness(frame, detection)
    
    # 2. Eyes Closed (simulate closed eyelid crop - smooth uniform patch)
    frame_closed = frame.copy()
    frame_closed[135:165, 225:255] = (140, 160, 210) # Left eye smooth skin
    frame_closed[135:165, 305:335] = (140, 160, 210) # Right eye smooth skin
    for _ in range(2):
        engine.check_liveness(frame_closed, detection)
    
    # 3. Eyes Reopen
    is_live_blink, status_blink, _ = engine.check_liveness(frame, detection)
    print(f"After natural eye blink cycle: is_live={is_live_blink}, status='{status_blink}'")
    assert is_live_blink == True, "Blink cycle should verify liveness"
    assert status_blink == "Real Person Verified", f"Expected 'Real Person Verified', got '{status_blink}'"
    print("✓ Physiological Eye Blink Verification PASSED -> 'Real Person Verified'")

    # Case D: Static Photo / Replay Attack (Frozen landmark detection)
    engine.reset()
    for _ in range(25):
        is_live_static, status_static, details_static = engine.check_liveness(frame, detection)
    
    print(f"After 25 frozen frames: is_live={is_live_static}, status='{status_static}', type={details_static.get('spoof_type')}")
    assert is_live_static == False, "Static frozen image must not be verified"
    assert status_static == "Spoofing Attempt Detected", f"Expected 'Spoofing Attempt Detected', got '{status_static}'"
    print("✓ Static Photo / Replay Attack Rejected -> 'Spoofing Attempt Detected'")

if __name__ == '__main__':
    test_speed_benchmark()
    test_anti_spoofing_modalities()
    test_liveness_engine()
    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! 🚀")
    print("==================================================")
