import base64
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import logging
from flask import Blueprint, request, jsonify, session
from database.models import db, SessionModel, StudentModel
from services.face_detection.mtcnn_detector import detect_faces
from services.face_alignment.align_face import align_face
from services.face_recognition.embedding_generator import generate_embedding
from services.face_recognition.compare_embeddings import compare_embeddings
from services.anti_spoofing.liveness_verifier import SingleBlinkLivenessSession
from services.attendance.mark_attendance import mark_attendance

attendance_bp = Blueprint('attendance', __name__)
logger = logging.getLogger(__name__)

# Active single-blink + passive PAD liveness sessions mapped by client ID
_liveness_sessions = {}

def get_or_create_liveness_session(client_id):
    if client_id not in _liveness_sessions:
        _liveness_sessions[client_id] = SingleBlinkLivenessSession(client_id)
    return _liveness_sessions[client_id]

@attendance_bp.route('/attendance/api/detect', methods=['POST'])
def api_detect():
    data = request.get_json() or {}
    frame_data = data.get('image') # Base64 string
    client_id = data.get('client_id', 'default_client')
    
    # 1. Verify active session first
    active_session = SessionModel.get_active()
    if not active_session:
        return jsonify({
            "success": False,
            "status": "No Active Session",
            "message": "There is no active attendance tracking session. Please start a session first."
        }), 200

    if not frame_data:
        return jsonify({
            "success": False, 
            "status": "Detecting Face...", 
            "message": "No image frame received."
        }), 400

    try:
        # Decode base64 image
        if "," in frame_data:
            _, encoded = frame_data.split(",", 1)
        else:
            encoded = frame_data
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            return jsonify({
                "success": True, 
                "faces_count": 0, 
                "status": "Detecting Face...", 
                "message": "Invalid frame"
            })
        
        # 2. Fast Face Detection (MTCNN)
        faces = detect_faces(frame)
        if not faces:
            return jsonify({
                "success": True, 
                "faces_count": 0, 
                "status": "Detecting Face...",
                "message": "Detecting Face... Please look directly into the camera."
            })
            
        # Select largest face
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = largest_face['box']
        
        # 3. Streamlined Single-Blink Active Challenge + AI Passive Anti-Spoofing (PAD)
        liveness_session = get_or_create_liveness_session(client_id)
        is_completed, liveness_status, liveness_details = liveness_session.process_frame(frame, largest_face)
        
        response_data = {
            "success": True,
            "faces_count": len(faces),
            "status": liveness_status,
            "box": [int(x), int(y), int(w), int(h)],
            "liveness": {
                "is_live": is_completed,
                "status": liveness_status,
                "details": liveness_details
            }
        }
        
        # 4. STRICT WORKFLOW ENFORCEMENT:
        # Workflow: Face Detection -> Eye Blink Verification -> Passive Anti-Spoofing -> Face Recognition -> Attendance Marked.
        # If liveness verification (Blink + Passive PAD) is NOT completed:
        # DO NOT execute face recognition or return student identity!
        if not is_completed:
            response_data["recognized"] = False
            return jsonify(response_data)
            
        # -------------------------------------------------------------
        # 5. Face Recognition & Attendance Marking (ONLY FOR VERIFIED REAL HUMANS)
        # -------------------------------------------------------------
        aligned_crop = align_face(frame, largest_face)
        live_embedding = generate_embedding(aligned_crop)
        
        # Scope face recognition strictly to students matching this session's branch, semester, and section
        active_branch = active_session.get("branch")
        active_section = active_session.get("section")
        active_semester = active_session.get("semester")
        
        session_students = None
        if (active_branch and active_branch not in ["All", "General"]) or \
           (active_section and active_section not in ["All", "General"]) or \
           (active_semester and active_semester not in ["All", "General"]):
            session_students = StudentModel.filter_students(
                branch=active_branch,
                semester=active_semester,
                section=active_section
            )
        
        # 1. Match against this session's specific enrolled class roster
        match_result = compare_embeddings(live_embedding, students=session_students)
        
        if match_result:
            student = match_result["student"]
            mark_res = mark_attendance(student["_id"])
            
            response_data["recognized"] = True
            response_data["name"] = student["name"]
            response_data["roll_no"] = student["roll_no"]
            response_data["branch"] = student.get("branch", "")
            response_data["section"] = student.get("section", "")
            response_data["semester"] = student.get("semester", "")
            response_data["confidence"] = match_result["confidence"]
            response_data["attendance"] = mark_res
            response_data["status"] = "Attendance Marked Successfully" if not mark_res.get("already_marked") else "Real Person Verified"
            
            # Reset liveness session after successful attendance marking
            liveness_session.reset()
        else:
            # 2. Check if the face is enrolled in a DIFFERENT branch, semester, or section
            global_match = compare_embeddings(live_embedding)
            if global_match:
                other_student = global_match["student"]
                other_grp = f"{other_student.get('branch', '')} Sem {other_student.get('semester', '')} Sec {other_student.get('section', '')}".strip()
                curr_grp = f"{active_branch or ''} Sem {active_semester or ''} Sec {active_section or ''}".strip()
                response_data["recognized"] = False
                response_data["name"] = other_student.get("name", "Unknown")
                response_data["roll_no"] = other_student.get("roll_no", "")
                response_data["status"] = "Real Person Verified"
                response_data["message"] = f"Student {other_student.get('name')} ({other_grp}) does not belong to this active session ({curr_grp}). Attendance not marked."
            else:
                response_data["recognized"] = False
                response_data["name"] = "Unknown"
                response_data["status"] = "Real Person Verified"
                response_data["message"] = "Real Person Verified, but face not enrolled in database."
            liveness_session.reset()
            
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during frame recognition: {e}")
        return jsonify({"success": False, "status": "Error", "message": str(e)}), 500

@attendance_bp.route('/attendance/api/reset_liveness', methods=['POST'])
def api_reset_liveness():
    data = request.get_json() or {}
    client_id = data.get('client_id', 'default_client')
    if client_id in _liveness_sessions:
        _liveness_sessions[client_id].reset()
    return jsonify({"success": True})
