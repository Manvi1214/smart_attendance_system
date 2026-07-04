import logging
import datetime
from flask import Blueprint, render_template, session, redirect, url_for, flash, jsonify
from database.models import StudentModel, AttendanceModel
from database.db_init import db

student_bp = Blueprint('student', __name__, template_folder='../templates')
logger = logging.getLogger(__name__)

def login_required_student(f):
    def wrapper(*args, **kwargs):
        if 'role' not in session or session['role'] != 'student':
            flash("Unauthorized. Please log in as a student.", "danger")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@student_bp.route('/student/dashboard')
@login_required_student
def dashboard():
    student_id = session.get('student_id')
    if not student_id:
        flash("Profile not linked. Contact admin.", "warning")
        return redirect(url_for('auth.login'))
        
    student = db.students.find_one({"_id": StudentModel.get_by_user_id(session['user_id'])["_id"]})
    
    # Calculate attendance statistics
    # 1. Total sessions conducted
    total_sessions = db.sessions.count_documents({})
    
    # 2. Sessions student was present
    present_records = AttendanceModel.get_by_student(student["_id"])
    present_count = len(present_records)
    
    # 3. Absent sessions
    absent_count = max(0, total_sessions - present_count)
    
    # 4. Attendance Percentage
    percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0.0
    
    # 5. Calendar history list (recent 10 days for calendar card)
    # Generate calendar checklist of recent sessions
    calendar_checklist = []
    sessions_list = list(db.sessions.find({}).sort("start_time", -1).limit(10))
    
    # Map presence for sessions
    present_session_ids = {str(r["_id"]) for r in present_records} # Wait, present_records maps session fields
    # Let's query directly to check session presence
    present_sessions = db.attendance.find({"student_id": student["_id"], "status": "Present"})
    present_set = {str(p["session_id"]) for p in present_sessions}
    
    for s in sessions_list:
        was_present = str(s["_id"]) in present_set
        calendar_checklist.append({
            "subject": s["subject"],
            "date": s["date"],
            "day": datetime.datetime.strptime(s["date"], '%Y-%m-%d').strftime('%a'),
            "status": "Present" if was_present else "Absent"
        })
        
    # Reverse checklist to show chronological order
    calendar_checklist.reverse()
    
    # Notifications list
    notifications = []
    if percentage < 75.0 and total_sessions > 2:
        notifications.append({
            "type": "warning",
            "message": f"Critical: Your attendance is {round(percentage, 1)}%, which is below the minimum threshold of 75%!"
        })
    else:
        notifications.append({
            "type": "success",
            "message": "Good standing: Your attendance is above 75%. Keep it up!"
        })
        
    if present_count > 0:
        last_present = present_records[0]
        notifications.append({
            "type": "info",
            "message": f"Last marked present for {last_present['subject']} on {last_present['date']} at {last_present['time']}"
        })
        
    return render_template(
        'student/dashboard.html',
        student=student,
        present_count=present_count,
        absent_count=absent_count,
        percentage=round(percentage, 1),
        calendar_checklist=calendar_checklist,
        notifications=notifications
    )

@student_bp.route('/student/profile')
@login_required_student
def profile():
    student = db.students.find_one({"_id": StudentModel.get_by_user_id(session['user_id'])["_id"]})
    return render_template('student/profile.html', student=student)

@student_bp.route('/student/attendance_calendar')
@login_required_student
def attendance_calendar():
    # Returns page to view detailed monthly grid calendar
    return render_template('student/attendance_calendar.html')

@student_bp.route('/student/reports')
@login_required_student
def reports():
    student = db.students.find_one({"_id": StudentModel.get_by_user_id(session['user_id'])["_id"]})
    attendance_records = AttendanceModel.get_by_student(student["_id"])
    return render_template('student/reports.html', attendance_records=attendance_records)

# --- APIs ---

@student_bp.route('/student/api/stats', methods=['GET'])
@login_required_student
def api_stats():
    student = db.students.find_one({"_id": StudentModel.get_by_user_id(session['user_id'])["_id"]})
    if not student:
        return jsonify({"error": "Student record not found"}), 404
        
    total_sessions = db.sessions.count_documents({})
    present_records = AttendanceModel.get_by_student(student["_id"])
    present_count = len(present_records)
    absent_count = max(0, total_sessions - present_count)
    percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0.0
    
    return jsonify({
        "name": student["name"],
        "roll_no": student["roll_no"],
        "branch": student["branch"],
        "section": student["section"],
        "total_sessions": total_sessions,
        "present": present_count,
        "absent": absent_count,
        "percentage": round(percentage, 1)
    })
