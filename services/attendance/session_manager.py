import logging
from database.models import SessionModel, StudentModel, AttendanceModel
from database.db_init import db

logger = logging.getLogger(__name__)

class SessionManager:
    @staticmethod
    def start_new_session(subject, teacher_id):
        """
        Starts a new active attendance session, closing any existing ones.
        """
        try:
            # Close existing sessions
            active = SessionModel.get_active()
            if active:
                SessionModel.end_session(active["_id"])
                logger.info(f"Closed previously active session: {active['_id']}")
                
            session = SessionModel.start_session(subject, teacher_id)
            logger.info(f"Started new session: {session['_id']} for subject: {subject}")
            return session
        except Exception as e:
            logger.error(f"Error starting new session: {e}")
            raise e

    @staticmethod
    def end_active_session():
        """
        Ends the current active session.
        """
        try:
            active = SessionModel.get_active()
            if not active:
                return None
            session = SessionModel.end_session(active["_id"])
            logger.info(f"Ended session: {session['_id']}")
            return session
        except Exception as e:
            logger.error(f"Error ending active session: {e}")
            raise e

    @staticmethod
    def get_session_stats(session_id=None):
        """
        Gathers live stats for a session (Total Students, Present, Absent, Attendance %).
        If session_id is None, uses the current active session.
        """
        import datetime
        date_str = datetime.date.today().strftime('%A, %d %b %Y')

        if db is None:
            return {
                "total": 0,
                "present": 0,
                "absent": 0,
                "percentage": 0.0,
                "date": date_str,
                "subject": "No Active Session"
            }

        try:
            # 1. Resolve session
            session = None
            if session_id:
                session = db.sessions.find_one({"_id": session_id})
            else:
                session = SessionModel.get_active()

            # Global enrolled count
            total_students = db.students.count_documents({})

            if not session:
                # Count unique students present in any session today
                today_db_str = datetime.date.today().strftime('%Y-%m-%d')
                present_students = len(db.attendance.distinct("student_id", {
                    "date": today_db_str,
                    "status": "Present"
                }))
                absent_students = max(0, total_students - present_students)
                percentage = (present_students / total_students * 100) if total_students > 0 else 0.0
                
                return {
                    "total": total_students,
                    "present": present_students,
                    "absent": absent_students,
                    "percentage": round(percentage, 1),
                    "date": date_str,
                    "subject": "No Active Session"
                }

            # 2. Query student metrics
            # Distinct students present in this session
            present_students = db.attendance.count_documents({
                "session_id": session["_id"],
                "status": "Present"
            })
            
            absent_students = max(0, total_students - present_students)
            
            percentage = (present_students / total_students * 100) if total_students > 0 else 0.0
            
            return {
                "session_id": str(session["_id"]),
                "subject": session["subject"],
                "total": total_students,
                "present": present_students,
                "absent": absent_students,
                "percentage": round(percentage, 1),
                "is_active": session.get("is_active", False),
                "date": date_str
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"total": 0, "present": 0, "absent": 0, "percentage": 0.0, "error": str(e)}
