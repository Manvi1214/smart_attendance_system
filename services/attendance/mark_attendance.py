import logging
import datetime
from database.models import AttendanceModel, SessionModel

logger = logging.getLogger(__name__)

def mark_attendance(student_id):
    """
    Checks if there is an active session and marks the student present.
    Args:
        student_id: The student ObjectId or string.
    Returns:
        dict: Result status and details of the marked attendance.
    """
    try:
        # 1. Retrieve current active session
        active_session = SessionModel.get_active()
        if not active_session:
            return {
                "success": False,
                "status": "No Active Session",
                "message": "There is currently no active attendance session started by a teacher."
            }
        
        session_id = active_session["_id"]
        
        # 2. Mark attendance (checks duplicates internally)
        record = AttendanceModel.mark_present(student_id, session_id)
        
        # Check if timestamp equals now or it was created previously
        # If it was just created, it will be returned. If it was already there, it's also returned.
        # We can verify if it was already marked by comparing dates/times or if custom keys match.
        time_now = datetime.datetime.now().strftime('%I:%M %p')
        
        # Log event
        logger.info(f"Student {student_id} attendance checked/marked for session {session_id}.")
        
        return {
            "success": True,
            "status": "Marked" if record["time"] == time_now else "Already Marked",
            "time": record["time"],
            "date": record["date"],
            "subject": active_session["subject"],
            "record_id": str(record["_id"])
        }
        
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        return {
            "success": False,
            "status": "Error",
            "message": str(e)
        }
