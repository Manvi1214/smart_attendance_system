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
    def create_student(name, roll_no, branch, section, semester, photo_path, embedding, user_id=None):
        if not db:
            raise RuntimeError("Database not connected")
            
        # Ensure roll_no is unique
        if db.students.find_one({"roll_no": roll_no}):
            return {"error": "Student with this roll number already registered."}

        student_doc = {
            "name": name,
            "roll_no": str(roll_no),
            "branch": branch,
            "section": section,
            "semester": semester,
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
        return list(db.students.find({}))

class SessionModel:
    @staticmethod
    def start_session(subject, teacher_id):
        if not db:
            raise RuntimeError("Database not connected")
            
        session_doc = {
            "subject": subject,
            "teacher_id": ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id,
            "start_time": datetime.datetime.utcnow(),
            "end_time": None,
            "date": datetime.date.today().strftime('%Y-%m-%d'),
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
    def get_by_student(student_id):
        if not db:
            return []
        if isinstance(student_id, str):
            student_id = ObjectId(student_id)
            
        pipeline = [
            {"$match": {"student_id": student_id}},
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
