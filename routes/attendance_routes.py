import base64
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import logging
from flask import Blueprint, request, jsonify, session
from database.models import SessionModel, StudentModel
from services.face_detection.mtcnn_detector import detect_faces
from services.face_alignment.align_face import align_face
from services.face_recognition.embedding_generator import generate_embedding
from services.face_recognition.compare_embeddings import compare_embeddings
from services.anti_spoofing.blink_detection import BlinkDetector
from services.attendance.mark_attendance import mark_attendance

attendance_bp = Blueprint('attendance', __name__)
logger = logging.getLogger(__name__)

# In-memory blink detector storage mapped by session/client identifier
# Format: { client_session_id: BlinkDetector }
_liveness_detectors = {}

@attendance_bp.route('/attendance/api/detect', methods=['POST'])
def api_detect():
    data = request.get_json()
    frame_data = data.get('image') # Base64 string
    client_id = data.get('client_id', 'default_client')
    
    # Verify active session first
    active_session = SessionModel.get_active()
    if not active_session:
        return jsonify({
            "success": False,
            "status": "No Active Session",
            "message": "There is no active attendance tracking session. Please start a session first."
        }), 200

    if not frame_data:
        return jsonify({"success": False, "message": "No image frame received."}), 400

    try:
        # Decode base64 image
        header, encoded = frame_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Detect faces
        faces = detect_faces(frame)
        if not faces:
            return jsonify({
                "success": True, 
                "faces_count": 0, 
                "message": "Scanning for faces..."
            })
            
        # Get the largest face
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = largest_face['box']
        
        # 2. Track liveness
        if client_id not in _liveness_detectors:
            _liveness_detectors[client_id] = BlinkDetector()
            
        detector = _liveness_detectors[client_id]
        is_live, liveness_details = detector.check_liveness(frame, largest_face)
        
        # 3. Align face crop
        aligned_crop = align_face(frame, largest_face)
        
        # 4. Generate FaceNet embedding
        live_embedding = generate_embedding(aligned_crop)
        
        # 5. Recognize student
        match_result = compare_embeddings(live_embedding)
        
        response_data = {
            "success": True,
            "faces_count": len(faces),
            "box": [int(x), int(y), int(w), int(h)],
            "liveness": {
                "is_live": is_live,
                "details": liveness_details
            }
        }
        
        if match_result:
            student = match_result["student"]
            response_data["recognized"] = True
            response_data["name"] = student["name"]
            response_data["roll_no"] = student["roll_no"]
            response_data["confidence"] = match_result["confidence"]
            
            # 6. Mark attendance if and only if liveness is confirmed
            if is_live:
                mark_res = mark_attendance(student["_id"])
                response_data["attendance"] = mark_res
                
                # Reset blink counter after successfully marking
                detector.reset()
            else:
                response_data["attendance"] = {
                    "success": False,
                    "status": "Spoof Alert / Blink Required",
                    "message": "Liveness verification in progress. Please blink or turn your head to confirm."
                }
        else:
            response_data["recognized"] = False
            response_data["name"] = "Unknown"
            
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during frame recognition: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@attendance_bp.route('/attendance/api/reset_liveness', methods=['POST'])
def api_reset_liveness():
    data = request.get_json() or {}
    client_id = data.get('client_id', 'default_client')
    if client_id in _liveness_detectors:
        _liveness_detectors[client_id].reset()
    return jsonify({"success": True})
