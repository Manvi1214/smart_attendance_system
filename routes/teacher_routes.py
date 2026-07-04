import os
import base64
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import datetime
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from database.models import StudentModel, SessionModel, AttendanceModel, UserModel
from database.db_init import db
from config import Config
from services.face_detection.mtcnn_detector import detect_faces
from services.face_alignment.align_face import align_face
from services.face_recognition.embedding_generator import generate_embedding
from services.attendance.session_manager import SessionManager

teacher_bp = Blueprint('teacher', __name__, template_folder='../templates')

# Registration temp memory cache to accumulate frames per roll number
# Format: { roll_no: { "embeddings": [], "images_count": 0, "name": ... } }
_reg_sessions = {}

def login_required_teacher(f):
    def wrapper(*args, **kwargs):
        if 'role' not in session or session['role'] != 'teacher':
            flash("Unauthorized. Please log in as a teacher.", "danger")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@teacher_bp.route('/teacher/dashboard')
@login_required_teacher
def dashboard():
    # Gather dashboard stats
    active_session = SessionModel.get_active()
    stats = SessionManager.get_session_stats()
    
    # Get recent sessions
    all_sessions = SessionModel.get_all()[:5]
    for s in all_sessions:
        s['id_str'] = str(s['_id'])
        # count attendance
        s['present_count'] = db.attendance.count_documents({"session_id": s["_id"], "status": "Present"})
    
    # Fetch all registered students
    students = StudentModel.get_all()
    for student in students:
        student['id_str'] = str(student['_id'])
        
    return render_template(
        'teacher/dashboard.html', 
        active_session=active_session, 
        stats=stats,
        recent_sessions=all_sessions,
        students=students
    )

@teacher_bp.route('/teacher/register_student')
@login_required_teacher
def register_student():
    return render_template('teacher/register_student.html')

@teacher_bp.route('/teacher/start_attendance')
@login_required_teacher
def start_attendance():
    active_session = SessionModel.get_active()
    stats = SessionManager.get_session_stats()
    return render_template('teacher/start_attendance.html', active_session=active_session, stats=stats)

@teacher_bp.route('/teacher/attendance_history')
@login_required_teacher
def attendance_history():
    sessions_list = SessionModel.get_all()
    history = []
    for s in sessions_list:
        present = db.attendance.count_documents({"session_id": s["_id"], "status": "Present"})
        total = db.students.count_documents({})
        history.append({
            "id": str(s["_id"]),
            "subject": s["subject"],
            "date": s["date"],
            "start_time": s["start_time"].strftime('%I:%M %p') if s["start_time"] else "N/A",
            "end_time": s["end_time"].strftime('%I:%M %p') if s["end_time"] else "Active",
            "present": present,
            "total": total,
            "percentage": round((present / total * 100) if total > 0 else 0, 1)
        })
    return render_template('teacher/attendance_history.html', history=history)

@teacher_bp.route('/teacher/reports')
@login_required_teacher
def reports():
    students = StudentModel.get_all()
    subjects = db.sessions.distinct("subject")
    return render_template('teacher/reports.html', students=students, subjects=subjects)

@teacher_bp.route('/teacher/analytics')
@login_required_teacher
def analytics():
    # Analytics data
    # Calculate attendance percentages per subject
    pipeline = [
        {"$group": {"_id": "$subject", "count": {"$sum": 1}}}
    ]
    subjects = list(db.sessions.aggregate(pipeline))
    
    subject_stats = []
    for sub in subjects:
        sub_name = sub["_id"]
        sub_sessions = list(db.sessions.find({"subject": sub_name}))
        sub_session_ids = [s["_id"] for s in sub_sessions]
        
        # total registered students
        total_students = db.students.count_documents({})
        total_slots = total_students * len(sub_sessions)
        
        # total present records
        total_present = db.attendance.count_documents({
            "session_id": {"$in": sub_session_ids},
            "status": "Present"
        })
        
        percentage = (total_present / total_slots * 100) if total_slots > 0 else 0.0
        subject_stats.append({
            "subject": sub_name,
            "classes": len(sub_sessions),
            "percentage": round(percentage, 1)
        })
        
    return render_template('teacher/analytics.html', subject_stats=subject_stats)

# --- APIs ---

@teacher_bp.route('/teacher/api/start_session', methods=['POST'])
@login_required_teacher
def api_start_session():
    data = request.get_json()
    subject = data.get('subject')
    if not subject:
        return jsonify({"success": False, "message": "Subject name is required."}), 400
        
    try:
        session_record = SessionManager.start_new_session(subject, session['user_id'])
        return jsonify({
            "success": True, 
            "message": f"Session started for {subject}.",
            "session_id": str(session_record["_id"])
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/api/end_session', methods=['POST'])
@login_required_teacher
def api_end_session():
    try:
        session_record = SessionManager.end_active_session()
        if session_record:
            return jsonify({"success": True, "message": "Attendance session ended successfully."})
        return jsonify({"success": False, "message": "No active session found."}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/api/stats', methods=['GET'])
@login_required_teacher
def api_stats():
    stats = SessionManager.get_session_stats()
    return jsonify(stats)

@teacher_bp.route('/teacher/api/register/start', methods=['POST'])
@login_required_teacher
def api_register_start():
    data = request.get_json()
    roll_no = str(data.get('roll_no')).strip()
    name = data.get('name').strip()
    branch = data.get('branch').strip()
    section = data.get('section').strip()
    semester = data.get('semester').strip()
    email = data.get('email').strip()
    password = data.get('password').strip()
    
    if not all([roll_no, name, branch, section, semester, email, password]):
        return jsonify({"success": False, "message": "All fields are required."}), 400
        
    # Check if student/user already exists
    if db.students.find_one({"roll_no": roll_no}):
        return jsonify({"success": False, "message": "Student with this roll number is already registered."}), 400
    if db.users.find_one({"email": email}):
        return jsonify({"success": False, "message": "A user with this email is already registered."}), 400

    # Initialize capturing session
    _reg_sessions[roll_no] = {
        "name": name,
        "branch": branch,
        "section": section,
        "semester": semester,
        "email": email,
        "password": password,
        "embeddings": [],
        "images_count": 0
    }
    
    return jsonify({"success": True, "message": "Student registration sequence initiated. Ready to capture frames."})

@teacher_bp.route('/teacher/api/register/upload_frame', methods=['POST'])
@login_required_teacher
def api_register_upload_frame():
    data = request.get_json()
    roll_no = str(data.get('roll_no')).strip()
    frame_data = data.get('image') # Base64 string
    
    if roll_no not in _reg_sessions:
        return jsonify({"success": False, "message": "Registration session not found or expired."}), 400
        
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
            return jsonify({"success": False, "message": "No face detected in the frame. Please look directly at the webcam."}), 200
            
        # Extract the largest face detected
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        
        # Ensure confidence is high
        if largest_face['confidence'] < 0.85:
            return jsonify({"success": False, "message": "Face match confidence is too low. Stand in well-lit area."}), 200
            
        # 2. Align face
        aligned_crop = align_face(frame, largest_face)
        
        # 3. Generate embedding
        embedding = generate_embedding(aligned_crop)
        if embedding is None:
            return jsonify({"success": False, "message": "Failed to extract facial features."}), 500
            
        # Cache embedding and save raw frame crop
        reg_sess = _reg_sessions[roll_no]
        reg_sess["embeddings"].append(embedding)
        reg_sess["images_count"] += 1
        
        # Save one sample profile image to static upload folder
        if reg_sess["images_count"] == 1:
            student_dir = os.path.join(Config.DATASET_FOLDER, f"{roll_no}_{reg_sess['name']}")
            os.makedirs(student_dir, exist_ok=True)
            
            # Save original frame
            cv2.imwrite(os.path.join(student_dir, "img1.jpg"), frame)
            
            # Save cropped profile photo
            profile_path = os.path.join(Config.UPLOAD_FOLDER, f"{roll_no}_{reg_sess['name']}.jpg")
            cv2.imwrite(profile_path, aligned_crop)
            
        return jsonify({
            "success": True, 
            "message": f"Frame captured.", 
            "count": reg_sess["images_count"]
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/api/register/complete', methods=['POST'])
@login_required_teacher
def api_register_complete():
    data = request.get_json()
    roll_no = str(data.get('roll_no')).strip()
    
    if roll_no not in _reg_sessions:
        return jsonify({"success": False, "message": "Registration session not found or expired."}), 400
        
    reg_sess = _reg_sessions[roll_no]
    
    if reg_sess["images_count"] < 40: # Minimum frames required
        return jsonify({"success": False, "message": f"Insufficient frames. Captured only {reg_sess['images_count']}/40. Please capture more angles."}), 400

    try:
        # 1. Average embeddings to create a single master face fingerprint
        all_embeddings = np.array(reg_sess["embeddings"])
        avg_embedding = np.mean(all_embeddings, axis=0).tolist()
        
        # 2. Create Student User Credentials
        user_record = UserModel.create_user(reg_sess["email"], reg_sess["password"], role="student")
        if "error" in user_record:
            return jsonify({"success": False, "message": user_record["error"]}), 400
            
        # 3. Create Student details record
        photo_rel_path = f"static/uploads/profile_photos/{roll_no}_{reg_sess['name']}.jpg"
        student_record = StudentModel.create_student(
            name=reg_sess["name"],
            roll_no=roll_no,
            branch=reg_sess["branch"],
            section=reg_sess["section"],
            semester=reg_sess["semester"],
            photo_path=photo_rel_path,
            embedding=avg_embedding,
            user_id=user_record["_id"]
        )
        
        # Save embedding to local .npy file as backup as requested in requirements
        npy_path = os.path.join(Config.EMBEDDINGS_FOLDER, f"{roll_no}_{reg_sess['name']}.npy")
        np.save(npy_path, np.array(avg_embedding))
        
        # Remove from temp cache
        del _reg_sessions[roll_no]
        
        return jsonify({
            "success": True, 
            "message": f"Student {student_record['name']} registered and enrolled successfully!"
        })
        
    except Exception as e:
        # Cleanup user record if student registration failed
        db.users.delete_one({"email": reg_sess["email"]})
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/api/student/delete/<student_id>', methods=['POST'])
@login_required_teacher
def api_delete_student(student_id):
    from bson import ObjectId
    try:
        student = db.students.find_one({"_id": ObjectId(student_id)})
        if not student:
            return jsonify({"success": False, "message": "Student not found."}), 404
            
        # Delete student details
        db.students.delete_one({"_id": ObjectId(student_id)})
        
        # Delete student user login credentials
        if student.get("user_id"):
            db.users.delete_one({"_id": student["user_id"]})
            
        # Remove their local .npy backup embedding
        npy_path = os.path.join(Config.EMBEDDINGS_FOLDER, f"{student['roll_no']}_{student['name']}.npy")
        if os.path.exists(npy_path):
            os.remove(npy_path)
            
        # Remove their profile photo file
        photo_path = os.path.join(Config.BASE_DIR, student['photo_path'])
        if os.path.exists(photo_path):
            os.remove(photo_path)
            
        # Delete associated attendance records
        db.attendance.delete_many({"student_id": ObjectId(student_id)})
        
        return jsonify({"success": True, "message": f"Student {student['name']} removed successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
