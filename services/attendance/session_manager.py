import logging
from database.models import SessionModel, StudentModel, AttendanceModel
from database.db_init import db

logger = logging.getLogger(__name__)

class SessionManager:
    @staticmethod
    def start_new_session(subject, teacher_id, branch=None, semester=None, section=None, date=None):
        """
        Starts a new active attendance session scoped to a specific branch, semester, section, and subject.
        Closes any previously active session.
        """
        try:
            # Close existing sessions
            active = SessionModel.get_active()
            if active:
                SessionModel.end_session(active["_id"])
                logger.info(f"Closed previously active session: {active['_id']}")
                
            session = SessionModel.start_session(
                subject=subject,
                teacher_id=teacher_id,
                branch=branch,
                semester=semester,
                section=section,
                date=date
            )
            logger.info(f"Started new session: {session['_id']} for {session.get('branch')} Sem {session.get('semester')} Sec {session.get('section')} - {subject}")
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
        Calculations use only students matching this session's branch, semester, and section.
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
                "subject": "No Active Session",
                "branch": "",
                "section": "",
                "semester": ""
            }

        try:
            # 1. Resolve session
            session = None
            if session_id:
                session = db.sessions.find_one({"_id": session_id})
            else:
                session = SessionModel.get_active()

            if not session:
                total_students = db.students.count_documents({})
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
                    "subject": "No Active Session",
                    "branch": "All",
                    "section": "All",
                    "semester": "All"
                }

            # 2. Scope total roster to session's specific branch, semester, and section
            roster_query = {}
            if session.get("branch") and session["branch"] not in ["All", "General"]:
                roster_query["branch"] = {"$regex": f"^{session['branch'].strip()}$", "$options": "i"}
            if session.get("section") and session["section"] not in ["All", "General"]:
                roster_query["section"] = {"$regex": f"^{session['section'].strip()}$", "$options": "i"}
            if session.get("semester") and session["semester"] not in ["All", "General"]:
                sem_clean = str(session["semester"]).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
                roster_query["semester"] = {"$regex": f"^{sem_clean}(th|st|nd|rd)?$", "$options": "i"}

            total_students = db.students.count_documents(roster_query)
            # Fallback only if no criteria specified (e.g. legacy session)
            if total_students == 0 and not roster_query:
                total_students = db.students.count_documents({})

            # 3. Distinct students present in this session
            present_students = db.attendance.count_documents({
                "session_id": session["_id"],
                "status": "Present"
            })
            
            absent_students = max(0, total_students - present_students)
            percentage = (present_students / total_students * 100) if total_students > 0 else 0.0
            
            return {
                "session_id": str(session["_id"]),
                "subject": session.get("subject", "General"),
                "branch": session.get("branch", "CSE"),
                "section": session.get("section", "A"),
                "semester": session.get("semester", "1"),
                "total": total_students,
                "present": present_students,
                "absent": absent_students,
                "percentage": round(percentage, 1),
                "is_active": session.get("is_active", False),
                "date": session.get("date", date_str)
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"total": 0, "present": 0, "absent": 0, "percentage": 0.0, "error": str(e)}
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"total": 0, "present": 0, "absent": 0, "percentage": 0.0, "error": str(e)}
