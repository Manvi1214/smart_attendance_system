import os
import base64
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import datetime
from bson import ObjectId
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from database.models import StudentModel, SessionModel, AttendanceModel, UserModel, AttendanceCorrectionModel
from database.db_init import db
from config import Config
from services.face_detection.mtcnn_detector import detect_faces
from services.face_alignment.align_face import align_face
from services.face_recognition.embedding_generator import generate_embedding
from services.face_recognition.compare_embeddings import invalidate_student_cache
from services.attendance.session_manager import SessionManager
from services.attendance.defaulter_service import DefaulterService

teacher_bp = Blueprint('teacher', __name__, template_folder='../templates')

# Registration temp memory cache to accumulate frames per roll number
# Format: { roll_no: { "embeddings": [], "images_count": 0, "name": ... } }
_reg_sessions = {}

def login_required_teacher(f):
    def wrapper(*args, **kwargs):
        if 'role' not in session or session['role'] != 'teacher':
            if request.is_json or request.path.startswith('/teacher/api/') or 'application/json' in request.headers.get('Accept', ''):
                return jsonify({"success": False, "message": "Unauthorized or session expired. Please log in again."}), 401
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
    all_sessions = SessionModel.get_all()[:6]
    for s in all_sessions:
        s['id_str'] = str(s['_id'])
        s['present_count'] = db.attendance.count_documents({"session_id": s["_id"], "status": "Present"})
        
        roster_q = {}
        if s.get("branch") and s["branch"] not in ["All", "General"]:
            roster_q["branch"] = {"$regex": f"^{s['branch'].strip()}$", "$options": "i"}
        if s.get("section") and s["section"] not in ["All", "General"]:
            roster_q["section"] = {"$regex": f"^{s['section'].strip()}$", "$options": "i"}
        if s.get("semester") and s["semester"] not in ["All", "General"]:
            sem_c = str(s["semester"]).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            roster_q["semester"] = {"$regex": f"^{sem_c}(th|st|nd|rd)?$", "$options": "i"}
            
        total = db.students.count_documents(roster_q) if roster_q else db.students.count_documents({})
        s['total_count'] = total
        s['rate'] = round((s['present_count'] / total * 100) if total > 0 else 0, 1)
    
    # Fetch all registered students
    students = StudentModel.get_all()
    for student in students:
        student['id_str'] = str(student['_id'])
        
    # Low Attendance / Defaulter Students summary
    defaulter_data = DefaulterService.calculate_student_attendance()
    
    # Filter options
    branches = sorted([b for b in db.students.distinct("branch") if b])
    sections = sorted([sec for sec in db.students.distinct("section") if sec])
    semesters = sorted([str(sem) for sem in db.students.distinct("semester") if sem])
    subjects = sorted([subj for subj in db.sessions.distinct("subject") if subj])
        
    return render_template(
        'teacher/dashboard.html', 
        active_session=active_session, 
        stats=stats,
        recent_sessions=all_sessions,
        students=students,
        defaulters=defaulter_data["defaulters"],
        defaulter_count=defaulter_data["defaulter_count"],
        total_students_count=defaulter_data["total_students"],
        branches=branches,
        sections=sections,
        semesters=semesters,
        subjects=subjects
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
    branches = sorted([b for b in db.students.distinct("branch") if b])
    sections = sorted([sec for sec in db.students.distinct("section") if sec])
    semesters = sorted([str(sem) for sem in db.students.distinct("semester") if sem])
    subjects = sorted([subj for subj in db.sessions.distinct("subject") if subj])
    return render_template(
        'teacher/start_attendance.html',
        active_session=active_session,
        stats=stats,
        branches=branches,
        sections=sections,
        semesters=semesters,
        subjects=subjects
    )

@teacher_bp.route('/teacher/attendance_history')
@login_required_teacher
def attendance_history():
    branch = request.args.get('branch', 'all')
    section = request.args.get('section', 'all')
    semester = request.args.get('semester', 'all')
    subject = request.args.get('subject', 'all')
    date_val = request.args.get('date', '').strip()

    sessions_list = SessionModel.get_filtered(
        branch=branch,
        semester=semester,
        section=section,
        subject=subject,
        date=date_val
    )
    history = []
    for s in sessions_list:
        present = db.attendance.count_documents({"session_id": s["_id"], "status": "Present"})
        
        # Calculate total students belonging specifically to this session's branch, sem, sec
        roster_q = {}
        if s.get("branch") and s["branch"] not in ["All", "General"]:
            roster_q["branch"] = {"$regex": f"^{s['branch'].strip()}$", "$options": "i"}
        if s.get("section") and s["section"] not in ["All", "General"]:
            roster_q["section"] = {"$regex": f"^{s['section'].strip()}$", "$options": "i"}
        if s.get("semester") and s["semester"] not in ["All", "General"]:
            sem_c = str(s["semester"]).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            roster_q["semester"] = {"$regex": f"^{sem_c}(th|st|nd|rd)?$", "$options": "i"}
            
        total = db.students.count_documents(roster_q) if roster_q else db.students.count_documents({})
        history.append({
            "id": str(s["_id"]),
            "subject": s["subject"],
            "branch": s.get("branch", "CSE"),
            "section": s.get("section", "A"),
            "semester": s.get("semester", "1"),
            "date": s["date"],
            "start_time": s["start_time"].strftime('%I:%M %p') if s.get("start_time") else "N/A",
            "end_time": s["end_time"].strftime('%I:%M %p') if s.get("end_time") else "Active",
            "present": present,
            "total": total,
            "percentage": round((present / total * 100) if total > 0 else 0, 1)
        })

    branches = sorted([b for b in db.students.distinct("branch") if b])
    sections = sorted([sec for sec in db.students.distinct("section") if sec])
    semesters = sorted([str(sem) for sem in db.students.distinct("semester") if sem])
    subjects = sorted([subj for subj in db.sessions.distinct("subject") if subj])

    return render_template(
        'teacher/attendance_history.html',
        history=history,
        branches=branches,
        sections=sections,
        semesters=semesters,
        subjects=subjects,
        current_filters={
            "branch": branch,
            "section": section,
            "semester": semester,
            "subject": subject,
            "date": date_val
        }
    )

@teacher_bp.route('/teacher/reports')
@login_required_teacher
def reports():
    students = StudentModel.get_all()
    subjects = db.sessions.distinct("subject")
    return render_template('teacher/reports.html', students=students, subjects=subjects)

@teacher_bp.route('/teacher/analytics')
@login_required_teacher
def analytics():
    # Read filter query parameters
    branch = request.args.get('branch', 'all')
    section = request.args.get('section', 'all')
    semester = request.args.get('semester', 'all')
    subject = request.args.get('subject', 'all')
    
    # 1. Student attendance summary with filters
    attendance_data = DefaulterService.calculate_student_attendance(
        branch=branch, section=section, semester=semester, subject=subject
    )
    
    # 2. Subject-wise low attendance analysis
    subject_analysis = DefaulterService.get_subject_low_attendance_analysis(
        branch=branch, section=section, semester=semester
    )
    
    # 3. Attendance distribution data for Chart.js
    distribution = DefaulterService.get_attendance_distribution(attendance_data["students"])
    
    # 4. Filter options
    filter_options = DefaulterService.get_filter_options()
    
    return render_template(
        'teacher/analytics.html',
        students=attendance_data["students"],
        defaulters=attendance_data["defaulters"],
        defaulter_count=attendance_data["defaulter_count"],
        total_students_count=attendance_data["total_students"],
        defaulter_percentage=attendance_data["defaulter_percentage"],
        subject_analysis=subject_analysis,
        distribution=distribution,
        filter_options=filter_options,
        current_filters={
            "branch": branch,
            "section": section,
            "semester": semester,
            "subject": subject
        }
    )

@teacher_bp.route('/teacher/api/analytics/defaulters', methods=['GET'])
@login_required_teacher
def api_analytics_defaulters():
    """API for dynamic filtering on the analytics page without page reload."""
    try:
        branch = request.args.get('branch', 'all')
        section = request.args.get('section', 'all')
        semester = request.args.get('semester', 'all')
        subject = request.args.get('subject', 'all')
        
        attendance_data = DefaulterService.calculate_student_attendance(
            branch=branch, section=section, semester=semester, subject=subject
        )
        subject_analysis = DefaulterService.get_subject_low_attendance_analysis(
            branch=branch, section=section, semester=semester
        )
        distribution = DefaulterService.get_attendance_distribution(attendance_data["students"])
        
        return jsonify({
            "success": True,
            "data": attendance_data,
            "distribution": distribution,
            "subject_analysis": subject_analysis
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/students')
@login_required_teacher
def student_management():
    """Dedicated Student Management page with branch, section, semester, subject filtering."""
    students = StudentModel.get_all()
    for s in students:
        s['id_str'] = str(s['_id'])
        
    branches = sorted([b for b in db.students.distinct("branch") if b])
    sections = sorted([sec for sec in db.students.distinct("section") if sec])
    semesters = sorted([str(sem) for sem in db.students.distinct("semester") if sem])
    subjects = sorted([subj for subj in db.students.distinct("subject") if subj] + [subj for subj in db.sessions.distinct("subject") if subj])
    # deduplicate subjects
    subjects = sorted(list(set(subjects)))
    
    return render_template(
        'teacher/students.html',
        students=students,
        branches=branches,
        sections=sections,
        semesters=semesters,
        subjects=subjects
    )

@teacher_bp.route('/teacher/api/students/filter', methods=['GET'])
@login_required_teacher
def api_filter_students():
    """Returns filtered students based on branch, semester, section, subject, and keyword search."""
    try:
        branch = request.args.get('branch', 'all')
        section = request.args.get('section', 'all')
        semester = request.args.get('semester', 'all')
        subject = request.args.get('subject', 'all')
        search = request.args.get('search', '').strip()

        students = StudentModel.filter_students(
            branch=branch,
            semester=semester,
            section=section,
            subject=subject,
            search=search
        )
        data = []
        for s in students:
            data.append({
                "id": str(s["_id"]),
                "name": s.get("name", "Unknown"),
                "roll_no": s.get("roll_no", ""),
                "branch": s.get("branch", ""),
                "section": s.get("section", ""),
                "semester": s.get("semester", ""),
                "subject": s.get("subject", ""),
                "photo_path": s.get("photo_path", ""),
                "email": s.get("email", ""),
                "phone": s.get("phone", "")
            })
        return jsonify({"success": True, "students": data, "count": len(data)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- APIs ---

@teacher_bp.route('/teacher/api/start_session', methods=['POST'])
@login_required_teacher
def api_start_session():
    data = request.get_json() or {}
    subject = str(data.get('subject', '')).strip()
    branch = str(data.get('branch', '')).strip().upper()
    semester = str(data.get('semester', '')).strip()
    section = str(data.get('section', '')).strip().upper()
    date_val = str(data.get('date', '')).strip()

    if not all([subject, branch, semester, section]):
        return jsonify({
            "success": False,
            "message": "Subject, Branch, Semester, and Section are all required to create an attendance session."
        }), 400

    # Verify that students are registered matching this branch, semester, and section
    sem_clean = semester.lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "").strip()
    roster_query = {
        "branch": {"$regex": f"^{branch}$", "$options": "i"},
        "section": {"$regex": f"^{section}$", "$options": "i"},
        "semester": {"$regex": f"^{sem_clean}(th|st|nd|rd)?$", "$options": "i"}
    }
    matching_students = list(db.students.find(roster_query))
    if len(matching_students) == 0:
        return jsonify({
            "success": False,
            "message": f"No registered students found for Branch: {branch}, Semester: {semester}, Section: {section}. Please register students for this class before creating a session."
        }), 400

    try:
        session_record = SessionManager.start_new_session(
            subject=subject,
            teacher_id=session['user_id'],
            branch=branch,
            semester=semester,
            section=section,
            date=date_val or None
        )
        return jsonify({
            "success": True, 
            "message": f"Session created for {branch} Sem {semester} Sec {section} - {subject} ({len(matching_students)} students enrolled).",
            "session_id": str(session_record["_id"]),
            "students_count": len(matching_students)
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
    roll_no = str(data.get('roll_no', '')).strip()
    name = data.get('name', '').strip()
    branch = data.get('branch', '').strip()
    section = data.get('section', '').strip()
    semester = data.get('semester', '').strip()
    subject = data.get('subject', '').strip()
    email = data.get('email', '').strip()
    phone = data.get('phone', '').strip()
    password = data.get('password', '').strip() or phone or "password123"
    
    if not all([roll_no, name, branch, section, semester, email, phone]):
        return jsonify({"success": False, "message": "All required fields (Name, Roll, Branch, Section, Semester, Email, Phone) must be filled."}), 400
        
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
        "subject": subject,
        "email": email,
        "phone": phone,
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
        default_pwd = reg_sess.get("password") or reg_sess.get("phone") or "password123"
        user_record = UserModel.create_user(reg_sess["email"], default_pwd, role="student")
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
            subject=reg_sess.get("subject", ""),
            photo_path=photo_rel_path,
            embedding=avg_embedding,
            user_id=user_record["_id"],
            phone=reg_sess.get("phone"),
            email=reg_sess.get("email")
        )
        
        # Save embedding to local .npy file as backup as requested in requirements
        npy_path = os.path.join(Config.EMBEDDINGS_FOLDER, f"{roll_no}_{reg_sess['name']}.npy")
        np.save(npy_path, np.array(avg_embedding))
        
        # Remove from temp cache
        del _reg_sessions[roll_no]
        
        # Invalidate student embedding cache so new student is immediately recognized
        invalidate_student_cache()
        
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
    # pyrefly: ignore [missing-import]
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
        
        # Invalidate student embedding cache
        invalidate_student_cache()
        
        return jsonify({"success": True, "message": f"Student {student['name']} removed successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/session/<session_id>/attendance', methods=['GET'])
@login_required_teacher
def view_session_attendance(session_id):
    """Redirect to session history with edit modal triggered."""
    return redirect(url_for('teacher.attendance_history', edit_session=session_id))

@teacher_bp.route('/teacher/session/<session_id>/view', methods=['GET'])
@login_required_teacher
def view_session_full_page(session_id):
    """Render a dedicated full-page manual attendance sheet for this session."""
    try:
        session_oid = ObjectId(session_id)
        session_doc = db.sessions.find_one({"_id": session_oid})
        if not session_doc:
            flash("Session not found.", "danger")
            return redirect(url_for('teacher.attendance_history'))

        # Scope student roster strictly to session's branch, semester, section
        roster_q = {}
        if session_doc.get("branch") and session_doc["branch"] not in ["All", "General"]:
            roster_q["branch"] = {"$regex": f"^{session_doc['branch'].strip()}$", "$options": "i"}
        if session_doc.get("section") and session_doc["section"] not in ["All", "General"]:
            roster_q["section"] = {"$regex": f"^{session_doc['section'].strip()}$", "$options": "i"}
        if session_doc.get("semester") and session_doc["semester"] not in ["All", "General"]:
            sem_c = str(session_doc["semester"]).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            roster_q["semester"] = {"$regex": f"^{sem_c}(th|st|nd|rd)?$", "$options": "i"}

        all_students = list(db.students.find(roster_q).sort("roll_no", 1)) if roster_q else list(db.students.find({}).sort("roll_no", 1))
        attendance_docs = list(db.attendance.find({"session_id": session_oid}))
        att_map = {str(a.get("student_id")): a for a in attendance_docs}

        student_roster = []
        present_count = 0
        for s in all_students:
            s_id_str = str(s["_id"])
            att_record = att_map.get(s_id_str)
            status = att_record.get("status", "Present") if att_record else "Absent"
            if status == "Present":
                present_count += 1
            student_roster.append({
                "id": s_id_str,
                "name": s.get("name", "Unknown"),
                "roll_no": s.get("roll_no", "N/A"),
                "branch": s.get("branch", "N/A"),
                "section": s.get("section", "N/A"),
                "semester": s.get("semester", "N/A"),
                "status": status,
                "time": att_record.get("time", "-") if att_record else "-",
                "manually_corrected": bool(att_record.get("manually_corrected", False)) if att_record else False,
                "photo_path": s.get("photo_path", "")
            })

        total_students = len(all_students)
        absent_count = max(0, total_students - present_count)
        rate = round((present_count / total_students * 100) if total_students > 0 else 0, 1)

        corrections = AttendanceCorrectionModel.get_by_session(session_id)
        formatted_corrections = []
        for c in corrections:
            formatted_corrections.append({
                "id": str(c["_id"]),
                "student_name": c.get("student_name", ""),
                "student_roll": c.get("student_roll", ""),
                "original_status": c.get("original_status", ""),
                "updated_status": c.get("updated_status", ""),
                "reason": c.get("reason", ""),
                "teacher_email": c.get("teacher_email", "Teacher"),
                "time": c.get("formatted_time", c.get("correction_time", "").strftime('%b %d, %Y %I:%M %p') if isinstance(c.get("correction_time"), datetime.datetime) else str(c.get("correction_time", "")))
            })

        session_data = {
            "id": str(session_doc["_id"]),
            "subject": session_doc.get("subject", "General"),
            "branch": session_doc.get("branch", "CSE"),
            "section": session_doc.get("section", "A"),
            "semester": session_doc.get("semester", "1"),
            "date": session_doc.get("date", ""),
            "is_active": session_doc.get("is_active", False),
            "start_time": session_doc["start_time"].strftime('%I:%M %p') if session_doc.get("start_time") else "N/A",
            "end_time": session_doc["end_time"].strftime('%I:%M %p') if session_doc.get("end_time") else "Active"
        }

        stats = {
            "total": total_students,
            "present": present_count,
            "absent": absent_count,
            "rate": rate
        }

        return render_template(
            'teacher/session_view.html',
            session=session_data,
            stats=stats,
            students=student_roster,
            corrections=formatted_corrections
        )
    except Exception as e:
        flash(f"Error loading session: {str(e)}", "danger")
        return redirect(url_for('teacher.attendance_history'))

@teacher_bp.route('/teacher/api/session/<session_id>/attendance', methods=['GET'])
@login_required_teacher
def api_session_attendance(session_id):
    """Fetch all students and their attendance status for a given session."""
    try:
        session_oid = ObjectId(session_id)
        session_doc = db.sessions.find_one({"_id": session_oid})
        if not session_doc:
            return jsonify({"success": False, "message": "Session not found."}), 404

        # Scope student roster strictly to session's branch, semester, section
        roster_q = {}
        if session_doc.get("branch") and session_doc["branch"] not in ["All", "General"]:
            roster_q["branch"] = {"$regex": f"^{session_doc['branch'].strip()}$", "$options": "i"}
        if session_doc.get("section") and session_doc["section"] not in ["All", "General"]:
            roster_q["section"] = {"$regex": f"^{session_doc['section'].strip()}$", "$options": "i"}
        if session_doc.get("semester") and session_doc["semester"] not in ["All", "General"]:
            sem_c = str(session_doc["semester"]).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            roster_q["semester"] = {"$regex": f"^{sem_c}(th|st|nd|rd)?$", "$options": "i"}

        all_students = list(db.students.find(roster_q).sort("roll_no", 1)) if roster_q else list(db.students.find({}).sort("roll_no", 1))
        # Fetch attendance for this session
        attendance_docs = list(db.attendance.find({"session_id": session_oid}))
        att_map = {}
        for a in attendance_docs:
            s_id_str = str(a.get("student_id"))
            att_map[s_id_str] = a

        student_roster = []
        present_count = 0
        for s in all_students:
            s_id_str = str(s["_id"])
            att_record = att_map.get(s_id_str)
            status = att_record.get("status", "Present") if att_record else "Absent"
            if status == "Present":
                present_count += 1
            student_roster.append({
                "id": s_id_str,
                "name": s.get("name", "Unknown"),
                "roll_no": s.get("roll_no", "N/A"),
                "branch": s.get("branch", "N/A"),
                "section": s.get("section", "N/A"),
                "semester": s.get("semester", "N/A"),
                "status": status,
                "time": att_record.get("time", "-") if att_record else "-",
                "manually_corrected": bool(att_record.get("manually_corrected", False)) if att_record else False,
                "photo_path": s.get("photo_path", "")
            })

        total_students = len(all_students)
        absent_count = max(0, total_students - present_count)
        rate = round((present_count / total_students * 100) if total_students > 0 else 0, 1)

        # Fetch corrections for this session
        corrections = AttendanceCorrectionModel.get_by_session(session_id)
        formatted_corrections = []
        for c in corrections:
            formatted_corrections.append({
                "id": str(c["_id"]),
                "student_name": c.get("student_name", ""),
                "student_roll": c.get("student_roll", ""),
                "original_status": c.get("original_status", ""),
                "updated_status": c.get("updated_status", ""),
                "reason": c.get("reason", ""),
                "teacher_email": c.get("teacher_email", "Teacher"),
                "time": c.get("formatted_time", c.get("correction_time", "").strftime('%b %d, %Y %I:%M %p') if isinstance(c.get("correction_time"), datetime.datetime) else str(c.get("correction_time", "")))
            })

        return jsonify({
            "success": True,
            "session": {
                "id": str(session_doc["_id"]),
                "subject": session_doc.get("subject", "General"),
                "branch": session_doc.get("branch", "CSE"),
                "section": session_doc.get("section", "A"),
                "semester": session_doc.get("semester", "1"),
                "date": session_doc.get("date", ""),
                "is_active": session_doc.get("is_active", False),
                "start_time": session_doc["start_time"].strftime('%I:%M %p') if session_doc.get("start_time") else "N/A",
                "end_time": session_doc["end_time"].strftime('%I:%M %p') if session_doc.get("end_time") else "Active"
            },
            "stats": {
                "total": total_students,
                "present": present_count,
                "absent": absent_count,
                "rate": rate
            },
            "students": student_roster,
            "corrections": formatted_corrections
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@teacher_bp.route('/teacher/api/attendance/correct', methods=['POST'])
@teacher_bp.route('/teacher/api/correct_attendance', methods=['POST'])
@login_required_teacher
def api_attendance_correct():
    """Manually update student attendance status with reason and audit log."""
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")
        student_id = data.get("student_id")
        new_status = data.get("new_status") or data.get("status")
        reason = (data.get("reason") or "").strip()

        if not session_id or not student_id or not new_status:
            return jsonify({"success": False, "message": "Session ID, student ID, and new status are required."}), 400

        if new_status not in ["Present", "Absent"]:
            return jsonify({"success": False, "message": "Status must be either 'Present' or 'Absent'."}), 400

        if not reason:
            return jsonify({"success": False, "message": "A reason is mandatory for correcting attendance."}), 400

        teacher_id = session.get("user_id")

        result = AttendanceModel.correct_attendance(
            student_id=student_id,
            session_id=session_id,
            new_status=new_status,
            reason=reason,
            teacher_id=teacher_id
        )

        if "error" in result:
            return jsonify({"success": False, "message": result["error"]}), 400

        now_formatted = datetime.datetime.now().strftime('%b %d, %Y %I:%M %p')
        return jsonify({
            "success": True,
            "message": result.get("message", "Attendance corrected successfully."),
            "data": result,
            "stats": result.get("stats"),
            "correction": {
                "time": now_formatted,
                "student_name": result.get("student_name", ""),
                "student_roll": result.get("student_roll", ""),
                "original_status": result.get("original_status", ""),
                "updated_status": result.get("updated_status", ""),
                "reason": reason,
                "teacher_email": session.get("user_email", "Teacher")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
