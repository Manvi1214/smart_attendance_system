from database.db_init import db
from bson import ObjectId
import datetime

class DefaulterService:
    @staticmethod
    def get_finalized_sessions(subject=None, branch=None, section=None, semester=None):
        """
        Fetch all finalized/closed sessions.
        Active or unfinished sessions (is_active=True or end_time is None) are strictly excluded.
        """
        if db is None:
            return []
            
        # A session is finalized if is_active is False and end_time is not None
        query = {
            "is_active": False,
            "end_time": {"$ne": None}
        }
        if subject and subject.strip().lower() != "all":
            query["subject"] = {"$regex": f"^{subject.strip()}$", "$options": "i"}
            
        sessions = list(db.sessions.find(query).sort("date", -1))
        
        # In-memory deduplication by _id (ensuring no duplicate session counts)
        seen_ids = set()
        unique_sessions = []
        for s in sessions:
            sid = str(s["_id"])
            if sid not in seen_ids:
                seen_ids.add(sid)
                unique_sessions.append(s)
        return unique_sessions

    @staticmethod
    def is_session_applicable_to_student(session_doc, student_doc):
        """
        Check if a session applies to a given student.
        Do not count sessions from other branches, sections, or semesters.
        If a session specifies branch/section/semester, it must match the student.
        If not specified on the session, it applies to all students.
        """
        if session_doc.get("branch") and session_doc.get("branch").strip():
            if session_doc["branch"].strip().lower() != (student_doc.get("branch") or "").strip().lower():
                return False
                
        if session_doc.get("section") and session_doc.get("section").strip():
            if session_doc["section"].strip().upper() != (student_doc.get("section") or "").strip().upper():
                return False
                
        if session_doc.get("semester") and str(session_doc.get("semester")).strip():
            if str(session_doc["semester"]).strip() != str(student_doc.get("semester") or "").strip():
                return False
                
        return True

    @staticmethod
    def calculate_student_attendance(branch=None, section=None, semester=None, subject=None):
        """
        Calculates attendance percentage for all matching students.
        Attendance Percentage = (Present Sessions / Total Applicable Sessions) * 100.
        Defaulter condition: percentage < 75.0 (exactly 75% is NOT a defaulter).
        Shows 'N/A' if total applicable sessions == 0.
        """
        if db is None:
            return {"students": [], "defaulters": [], "defaulter_count": 0, "total_students": 0}

        # 1. Fetch students matching student-level filters
        student_query = {}
        if branch and branch.strip().lower() != "all":
            student_query["branch"] = {"$regex": f"^{branch.strip()}$", "$options": "i"}
        if section and section.strip().lower() != "all":
            student_query["section"] = {"$regex": f"^{section.strip()}$", "$options": "i"}
        if semester and str(semester).strip().lower() != "all":
            student_query["semester"] = str(semester).strip()
            
        all_students = list(db.students.find(student_query).sort("roll_no", 1))
        
        # 2. Fetch all finalized sessions
        finalized_sessions = DefaulterService.get_finalized_sessions(subject=subject)
        
        # 3. For all students, fetch their Present attendance records
        student_summaries = []
        defaulters = []
        
        for student in all_students:
            s_id = student["_id"]
            
            # Find applicable sessions for this student
            applicable_sessions = [
                s for s in finalized_sessions
                if DefaulterService.is_session_applicable_to_student(s, student)
            ]
            
            total_applicable = len(applicable_sessions)
            app_session_ids = {str(s["_id"]) for s in applicable_sessions}
            
            # Fetch student's present attendance records for these sessions
            presences = list(db.attendance.find({
                "$or": [
                    {"student_id": s_id},
                    {"student_id": str(s_id)}
                ],
                "session_id": {"$in": [s["_id"] for s in applicable_sessions] + [str(s["_id"]) for s in applicable_sessions]},
                "status": "Present"
            }))
            
            # Count distinct present sessions to avoid duplicates
            present_session_ids = {str(p["session_id"]) for p in presences if str(p.get("session_id")) in app_session_ids}
            present_count = len(present_session_ids)
            absent_count = max(0, total_applicable - present_count)
            
            if total_applicable > 0:
                pct = round((present_count / total_applicable * 100), 1)
                pct_display = f"{pct}%"
                # Below 75% is defaulter; exactly 75% is NOT a defaulter
                is_defaulter = (pct < 75.0)
            else:
                pct = None
                pct_display = "N/A"
                is_defaulter = False
                
            student_data = {
                "id": str(s_id),
                "name": student.get("name", "Unknown"),
                "roll_no": student.get("roll_no", "N/A"),
                "branch": student.get("branch", "N/A"),
                "section": student.get("section", "N/A"),
                "semester": str(student.get("semester", "N/A")),
                "phone": student.get("phone", "N/A"),
                "email": student.get("email", "N/A"),
                "photo_path": student.get("photo_path", "static/images/default_profile.png"),
                "total_sessions": total_applicable,
                "present_sessions": present_count,
                "absent_sessions": absent_count,
                "percentage": pct,
                "percentage_display": pct_display,
                "is_defaulter": is_defaulter,
                "status": "Defaulter" if is_defaulter else ("Eligible" if pct is not None else "N/A")
            }
            
            student_summaries.append(student_data)
            if is_defaulter:
                defaulters.append(student_data)
                
        # Sort defaulters by lowest percentage first, then roll number
        defaulters.sort(key=lambda s: (s["percentage"] if s["percentage"] is not None else 999, s["roll_no"]))
        
        return {
            "students": student_summaries,
            "defaulters": defaulters,
            "defaulter_count": len(defaulters),
            "total_students": len(student_summaries),
            "defaulter_percentage": round((len(defaulters) / len(student_summaries) * 100), 1) if student_summaries else 0.0
        }

    @staticmethod
    def get_subject_low_attendance_analysis(branch=None, section=None, semester=None):
        """
        Subject-wise low attendance analysis across all subjects.
        Returns subject stats including total finalized sessions, average attendance %,
        and number of defaulter students (< 75%) in each subject.
        """
        if db is None:
            return []
            
        all_finalized = DefaulterService.get_finalized_sessions()
        
        # Group finalized sessions by subject
        subject_map = {}
        for s in all_finalized:
            subj = s.get("subject", "General").strip()
            if not subj:
                continue
            if subj not in subject_map:
                subject_map[subj] = []
            subject_map[subj].append(s)
            
        # Get students matching branch/section/semester
        student_query = {}
        if branch and branch.strip().lower() != "all":
            student_query["branch"] = {"$regex": f"^{branch.strip()}$", "$options": "i"}
        if section and section.strip().lower() != "all":
            student_query["section"] = {"$regex": f"^{section.strip()}$", "$options": "i"}
        if semester and str(semester).strip().lower() != "all":
            student_query["semester"] = str(semester).strip()
            
        students = list(db.students.find(student_query))
        
        analysis = []
        for subj, sessions in subject_map.items():
            total_classes = len(sessions)
            
            # For each student, compute attendance in this subject
            subject_defaulters = 0
            student_pcts = []
            
            for st in students:
                app_s = [s for s in sessions if DefaulterService.is_session_applicable_to_student(s, st)]
                if not app_s:
                    continue
                app_s_ids = {str(s["_id"]) for s in app_s}
                
                presences = db.attendance.find({
                    "$or": [{"student_id": st["_id"]}, {"student_id": str(st["_id"])}],
                    "session_id": {"$in": [s["_id"] for s in app_s] + [str(s["_id"]) for s in app_s]},
                    "status": "Present"
                })
                p_ids = {str(p["session_id"]) for p in presences if str(p.get("session_id")) in app_s_ids}
                present_c = len(p_ids)
                total_c = len(app_s)
                
                pct = (present_c / total_c * 100) if total_c > 0 else 0.0
                student_pcts.append(pct)
                if pct < 75.0:
                    subject_defaulters += 1
                    
            avg_pct = round(sum(student_pcts) / len(student_pcts), 1) if student_pcts else 0.0
            
            analysis.append({
                "subject": subj,
                "classes": total_classes,
                "average_percentage": avg_pct,
                "defaulters_count": subject_defaulters,
                "total_enrolled": len(student_pcts),
                "is_low": avg_pct < 75.0 or subject_defaulters > 0
            })
            
        # Sort by lowest average percentage, then highest defaulters count
        analysis.sort(key=lambda x: (x["average_percentage"], -x["defaulters_count"]))
        return analysis

    @staticmethod
    def get_attendance_distribution(students_summary):
        """
        Group students into attendance distribution brackets for visualization chart.
        - < 50% (Critical)
        - 50% - 74.9% (Defaulter)
        - 75% - 84.9% (Satisfactory)
        - 85% - 100% (Excellent)
        - N/A (No sessions)
        """
        brackets = {
            "< 50%": 0,
            "50% - 74.9%": 0,
            "75% - 84.9%": 0,
            "85% - 100%": 0,
            "N/A": 0
        }
        
        for s in students_summary:
            pct = s.get("percentage")
            if pct is None:
                brackets["N/A"] += 1
            elif pct < 50.0:
                brackets["< 50%"] += 1
            elif pct < 75.0:
                brackets["50% - 74.9%"] += 1
            elif pct < 85.0:
                brackets["75% - 84.9%"] += 1
            else:
                brackets["85% - 100%"] += 1
                
        return {
            "labels": ["< 50% (Critical)", "50% - 74.9% (Defaulters)", "75% - 84.9% (Good)", "85% - 100% (Excellent)", "N/A"],
            "counts": [brackets["< 50%"], brackets["50% - 74.9%"], brackets["75% - 84.9%"], brackets["85% - 100%"], brackets["N/A"]],
            "colors": ["#ef4444", "#f97316", "#0ea5e9", "#10b981", "#64748b"]
        }

    @staticmethod
    def get_filter_options():
        """
        Fetch distinct branches, sections, semesters, and subjects available in the database.
        """
        if db is None:
            return {"branches": [], "sections": [], "semesters": [], "subjects": []}
            
        branches = sorted([b for b in db.students.distinct("branch") if b])
        sections = sorted([s for s in db.students.distinct("section") if s])
        semesters = sorted([str(s) for s in db.students.distinct("semester") if s])
        subjects = sorted([s for s in db.sessions.distinct("subject") if s])
        
        return {
            "branches": branches,
            "sections": sections,
            "semesters": semesters,
            "subjects": subjects
        }
