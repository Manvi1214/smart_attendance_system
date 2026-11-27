import datetime
# pyrefly: ignore [missing-import]
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import database.db_init as db_init

# Dynamic database proxy to support mock database overrides during unit testing
class DatabaseProxy:
    def __getattr__(self, name):
        target = db_init.db
        if target is None:
            raise RuntimeError("Database not connected")
        return getattr(target, name)
        
    def __bool__(self):
        return db_init.db is not None

db = DatabaseProxy()

class UserModel:
    @staticmethod
    def create_user(email, password, role='student'):
        if not db:
            raise RuntimeError("Database not connected")
        
        # Check if user already exists
        if db.users.find_one({"email": email}):
            return {"error": "User with this email already exists."}
            
        hashed_password = generate_password_hash(password)
        user_doc = {
            "email": email,
            "password": hashed_password,
            "role": role,
            "created_at": datetime.datetime.utcnow()
        }
        
        result = db.users.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc

    @staticmethod
    def verify_user(email, password):
        if not db:
            return None
        
        user = db.users.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            return user
        return None

    @staticmethod
    def get_by_id(user_id):
        if not db:
            return None
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return db.users.find_one({"_id": user_id})

class StudentModel:
    @staticmethod
    def create_student(name, roll_no, branch, section, semester, photo_path, embedding, user_id=None, phone=None, email=None, subject=None):
        if not db:
            raise RuntimeError("Database not connected")
            
        # Ensure roll_no is unique
        if db.students.find_one({"roll_no": str(roll_no).strip()}):
            return {"error": "Student with this roll number already registered."}

        norm_branch = str(branch).strip().upper() if branch else "CSE"
        norm_section = str(section).strip().upper() if section else "A"
        norm_sem = str(semester).strip() if semester else "1"
        norm_subj = str(subject).strip() if subject else ""

        student_doc = {
            "name": name.strip(),
            "roll_no": str(roll_no).strip(),
            "branch": norm_branch,
            "section": norm_section,
            "semester": norm_sem,
            "subject": norm_subj,
            "phone": phone.strip() if phone else None,
            "email": email.strip() if email else None,
            "photo_path": photo_path,
            "embedding": embedding, # List of floats
            "user_id": ObjectId(user_id) if user_id else None,
            "registered_at": datetime.datetime.utcnow()
        }
        
        result = db.students.insert_one(student_doc)
        student_doc["_id"] = result.inserted_id
        return student_doc

    @staticmethod
    def get_by_roll(roll_no):
        if not db:
            return None
        return db.students.find_one({"roll_no": str(roll_no)})

    @staticmethod
    def get_by_user_id(user_id):
        if not db:
            return None
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return db.students.find_one({"user_id": user_id})

    @staticmethod
    def get_all():
        if not db:
            return []
        return list(db.students.find({}).sort("roll_no", 1))

    @staticmethod
    def filter_students(branch=None, semester=None, section=None, subject=None, search=None):
        """
        Filters students by branch, semester, section, subject, or keyword search.
        Handles variations like '5' and '5th'.
        """
        if not db:
            return []
            
        query = {}
        if branch and branch != 'all':
            query["branch"] = {"$regex": f"^{branch.strip()}$", "$options": "i"}
            
        if section and section != 'all':
            query["section"] = {"$regex": f"^{section.strip()}$", "$options": "i"}
            
        if semester and semester != 'all':
            sem_clean = str(semester).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            query["semester"] = {"$regex": f"^{sem_clean}(th|st|nd|rd)?$", "$options": "i"}
            
        if subject and subject != 'all':
            query["subject"] = {"$regex": subject.strip(), "$options": "i"}
            
        if search and search.strip():
            s = search.strip()
            query["$or"] = [
                {"name": {"$regex": s, "$options": "i"}},
                {"roll_no": {"$regex": s, "$options": "i"}}
            ]
            
        return list(db.students.find(query).sort("roll_no", 1))

class SessionModel:
    @staticmethod
    def start_session(subject, teacher_id, branch=None, semester=None, section=None, date=None):
        if not db:
            raise RuntimeError("Database not connected")
            
        norm_branch = str(branch).strip().upper() if branch and str(branch).strip() != "all" else "CSE"
        norm_section = str(section).strip().upper() if section and str(section).strip() != "all" else "A"
        norm_sem = str(semester).strip() if semester and str(semester).strip() != "all" else "1"
        session_date = date.strip() if date and str(date).strip() else datetime.date.today().strftime('%Y-%m-%d')

        session_doc = {
            "subject": subject.strip(),
            "branch": norm_branch,
            "section": norm_section,
            "semester": norm_sem,
            "teacher_id": ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id,
            "start_time": datetime.datetime.utcnow(),
            "end_time": None,
            "date": session_date,
            "is_active": True
        }
        
        result = db.sessions.insert_one(session_doc)
        session_doc["_id"] = result.inserted_id
        return session_doc

    @staticmethod
    def end_session(session_id):
        if not db:
            return None
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        return db.sessions.find_one_and_update(
            {"_id": session_id},
            {"$set": {"end_time": datetime.datetime.utcnow(), "is_active": False}},
            return_document=True
        )

    @staticmethod
    def get_active():
        if not db:
            return None
        return db.sessions.find_one({"is_active": True})

    @staticmethod
    def get_all():
        if not db:
            return []
        return list(db.sessions.find({}).sort("start_time", -1))

    @staticmethod
    def get_filtered(branch=None, semester=None, section=None, subject=None, date=None):
        if not db:
            return []
        query = {}
        if branch and branch != 'all':
            query["branch"] = {"$regex": f"^{branch.strip()}$", "$options": "i"}
        if section and section != 'all':
            query["section"] = {"$regex": f"^{section.strip()}$", "$options": "i"}
        if semester and semester != 'all':
            sem_clean = str(semester).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            query["semester"] = {"$regex": f"^{sem_clean}(th|st|nd|rd)?$", "$options": "i"}
        if subject and subject != 'all':
            query["subject"] = {"$regex": subject.strip(), "$options": "i"}
        if date and date.strip():
            query["date"] = date.strip()
        return list(db.sessions.find(query).sort("start_time", -1))

    @staticmethod
    def get_all():
        if not db:
            return []
        return list(db.sessions.find({}).sort("start_time", -1))

class AttendanceModel:
    @staticmethod
    def mark_present(student_id, session_id, status="Present"):
        if not db:
            raise RuntimeError("Database not connected")
            
        if isinstance(student_id, str):
            student_id = ObjectId(student_id)
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        date_str = datetime.date.today().strftime('%Y-%m-%d')
        time_str = datetime.datetime.now().strftime('%I:%M %p')
        
        # Check if already marked for this session/date
        existing = db.attendance.find_one({
            "student_id": student_id,
            "session_id": session_id,
            "date": date_str
        })
        
        if existing:
            return existing
            
        attendance_doc = {
            "student_id": student_id,
            "session_id": session_id,
            "date": date_str,
            "time": time_str,
            "status": status,
            "timestamp": datetime.datetime.utcnow()
        }
        
        result = db.attendance.insert_one(attendance_doc)
        attendance_doc["_id"] = result.inserted_id
        return attendance_doc

    @staticmethod
    def get_by_session(session_id):
        if not db:
            return []
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        pipeline = [
            {"$match": {"session_id": session_id}},
            {"$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student"
            }},
            {"$unwind": "$student"},
            {"$project": {
                "_id": 1,
                "date": 1,
                "time": 1,
                "status": 1,
                "student_name": "$student.name",
                "student_roll": "$student.roll_no",
                "student_branch": "$student.branch",
                "student_section": "$student.section"
            }}
        ]
        
        return list(db.attendance.aggregate(pipeline))

    @staticmethod
    def get_by_student(student_id, status="Present"):
        if not db:
            return []
        if isinstance(student_id, str):
            student_id = ObjectId(student_id)
            
        match_query = {"student_id": student_id}
        if status:
            match_query["status"] = status
            
        pipeline = [
            {"$match": match_query},
            {"$lookup": {
                "from": "sessions",
                "localField": "session_id",
                "foreignField": "_id",
                "as": "session"
            }},
            {"$unwind": "$session"},
            {"$project": {
                "_id": 1,
                "date": 1,
                "time": 1,
                "status": 1,
                "subject": "$session.subject"
            }},
            {"$sort": {"date": -1, "time": -1}}
        ]
        return list(db.attendance.aggregate(pipeline))

    @staticmethod
    def correct_attendance(student_id, session_id, new_status, reason, teacher_id):
        if not db:
            raise RuntimeError("Database not connected")
            
        student_oid = ObjectId(student_id) if isinstance(student_id, str) else student_id
        session_oid = ObjectId(session_id) if isinstance(session_id, str) else session_id
        teacher_oid = ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id
        
        session_doc = db.sessions.find_one({"_id": session_oid})
        if not session_doc:
            return {"error": "Session not found."}
            
        student_doc = db.students.find_one({"_id": student_oid})
        if not student_doc:
            return {"error": "Student not found."}
            
        # Determine existing attendance status
        existing = db.attendance.find_one({
            "student_id": student_oid,
            "session_id": session_oid
        })
        
        original_status = existing.get("status", "Present") if existing else "Absent"
        if original_status == new_status:
            return {"error": f"Student is already marked as {new_status}."}
            
        now_time = datetime.datetime.now().strftime('%I:%M %p')
        session_date = session_doc.get("date", datetime.date.today().strftime('%Y-%m-%d'))
        
        # Upsert attendance record to guarantee NO duplicate records
        att_record = db.attendance.find_one_and_update(
            {"student_id": student_oid, "session_id": session_oid},
            {"$set": {
                "student_id": student_oid,
                "session_id": session_oid,
                "status": new_status,
                "date": session_date,
                "time": now_time,
                "timestamp": datetime.datetime.utcnow(),
                "manually_corrected": True
            }},
            upsert=True,
            return_document=True
        )
        
        # Fetch teacher email for logging
        teacher_user = db.users.find_one({"_id": teacher_oid})
        teacher_email = teacher_user.get("email", "teacher") if teacher_user else "teacher"
        
        # Log correction history
        AttendanceCorrectionModel.log_correction(
            student_id=student_oid,
            student_name=student_doc.get("name", ""),
            student_roll=student_doc.get("roll_no", ""),
            session_id=session_oid,
            session_subject=session_doc.get("subject", ""),
            original_status=original_status,
            updated_status=new_status,
            reason=reason,
            teacher_id=teacher_oid,
            teacher_email=teacher_email,
            attendance_id=att_record.get("_id")
        )
        
        # Calculate updated session statistics
        roster_q = {}
        if session_doc.get("branch") and session_doc["branch"] not in ["All", "General"]:
            roster_q["branch"] = {"$regex": f"^{session_doc['branch'].strip()}$", "$options": "i"}
        if session_doc.get("section") and session_doc["section"] not in ["All", "General"]:
            roster_q["section"] = {"$regex": f"^{session_doc['section'].strip()}$", "$options": "i"}
        if session_doc.get("semester") and session_doc["semester"] not in ["All", "General"]:
            sem_c = str(session_doc["semester"]).strip().lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
            roster_q["semester"] = {"$regex": f"^{sem_c}(th|st|nd|rd)?$", "$options": "i"}

        total_students = db.students.count_documents(roster_q) if roster_q else db.students.count_documents({})
        present_count = db.attendance.count_documents({"session_id": session_oid, "status": "Present"})
        absent_count = max(0, total_students - present_count)
        percentage = round((present_count / total_students * 100), 1) if total_students > 0 else 0.0
        
        return {
            "success": True,
            "message": f"Attendance successfully updated from {original_status} to {new_status}.",
            "original_status": original_status,
            "updated_status": new_status,
            "student_id": str(student_oid),
            "student_name": student_doc.get("name", ""),
            "student_roll": student_doc.get("roll_no", ""),
            "stats": {
                "present": present_count,
                "absent": absent_count,
                "total": total_students,
                "percentage": percentage
            }
        }

class AttendanceCorrectionModel:
    @staticmethod
    def log_correction(student_id, student_name, student_roll, session_id, session_subject, original_status, updated_status, reason, teacher_id, teacher_email, attendance_id=None):
        if not db:
            raise RuntimeError("Database not connected")
            
        student_oid = ObjectId(student_id) if isinstance(student_id, str) else student_id
        session_oid = ObjectId(session_id) if isinstance(session_id, str) else session_id
        att_oid = ObjectId(attendance_id) if (attendance_id and isinstance(attendance_id, str)) else attendance_id
        teacher_oid = ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id
        
        now = datetime.datetime.utcnow()
        formatted_date = datetime.datetime.now().strftime('%b %d, %Y %I:%M %p')
        
        doc = {
            "student_id": student_oid,
            "student_name": student_name,
            "student_roll": str(student_roll),
            "session_id": session_oid,
            "session_subject": session_subject,
            "original_status": original_status,
            "updated_status": updated_status,
            "reason": reason,
            "teacher_id": teacher_oid,
            "teacher_email": teacher_email,
            "attendance_id": att_oid,
            "correction_time": now,
            "formatted_time": formatted_date
        }
        
        result = db.attendance_corrections.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    @staticmethod
    def get_by_session(session_id):
        if not db:
            return []
        session_oid = ObjectId(session_id) if isinstance(session_id, str) else session_id
        return list(db.attendance_corrections.find({"session_id": session_oid}).sort("correction_time", -1))

    @staticmethod
    def get_all(limit=50):
        if not db:
            return []
        return list(db.attendance_corrections.find({}).sort("correction_time", -1).limit(limit))
