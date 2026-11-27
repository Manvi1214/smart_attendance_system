import logging
import datetime
from database.models import AttendanceModel, SessionModel

logger = logging.getLogger(__name__)

# In-memory tracking of students marked in the current active session
# Format: { (session_id_str, student_id_str): { "time": "...", "date": "...", "record_id": "..." } }
_session_marked_cache = {}
_current_active_session_id = None

def reset_session_attendance_cache():
    """Resets the marked students cache when a new session starts."""
    global _session_marked_cache, _current_active_session_id
    _session_marked_cache.clear()
    _current_active_session_id = None

def mark_attendance(student_id):
    """
    Checks if there is an active session and marks the student present.
    Uses in-memory session tracking to prevent duplicate database writes and duplicate logs.
    Args:
        student_id: The student ObjectId or string.
    Returns:
        dict: Result status and details of the marked attendance.
    """
    global _current_active_session_id
    try:
        # 1. Retrieve current active session
        active_session = SessionModel.get_active()
        if not active_session:
            return {
                "success": False,
                "status": "No Active Session",
                "message": "There is currently no active attendance session started by a teacher."
            }
        
        session_id_str = str(active_session["_id"])
        student_id_str = str(student_id)
        
        # If active session changed, clear old session cache
        if _current_active_session_id != session_id_str:
            _session_marked_cache.clear()
            _current_active_session_id = session_id_str
            
        cache_key = (session_id_str, student_id_str)
        if cache_key in _session_marked_cache:
            cached = _session_marked_cache[cache_key]
            return {
                "success": True,
                "status": "Already Marked",
                "already_marked": True,
                "time": cached["time"],
                "date": cached["date"],
                "subject": active_session["subject"],
                "record_id": cached["record_id"],
                "message": "Attendance already recorded for this student in this session."
            }
        
        # 2. Mark attendance in database (upsert/idempotent)
        record = AttendanceModel.mark_present(student_id, active_session["_id"])
        
        time_now = record.get("time", datetime.datetime.now().strftime('%I:%M %p'))
        date_str = record.get("date", datetime.date.today().strftime('%Y-%m-%d'))
        record_id_str = str(record.get("_id", ""))
        
        # Cache marked status
        _session_marked_cache[cache_key] = {
            "time": time_now,
            "date": date_str,
            "record_id": record_id_str
        }
        
        logger.info(f"Student {student_id_str} marked present in session {session_id_str}.")
        
        return {
            "success": True,
            "status": "Marked",
            "already_marked": False,
            "time": time_now,
            "date": date_str,
            "subject": active_session["subject"],
            "record_id": record_id_str
        }
        
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        return {
            "success": False,
            "status": "Error",
            "message": str(e)
        }
