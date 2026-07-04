import unittest
# pyrefly: ignore [missing-import]
from bson import ObjectId
# pyrefly: ignore [missing-import]
import mongomock
import database.db_init as db_init
from database.models import StudentModel, UserModel
from app import app

class TestDashboardFlow(unittest.TestCase):
    def setUp(self):
        # Override the db client with a mock client for testing
        self.mock_client = mongomock.MongoClient()
        self.mock_db = self.mock_client.smart_attendance_test
        self.original_db = db_init.db
        db_init.db = self.mock_db
        
        # Setup schema indexes on mock db
        db_init.init_indexes(self.mock_db)
        
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def tearDown(self):
        # Restore original database client
        db_init.db = self.original_db

    def test_dashboard_and_deletion_flow(self):
        """Test teacher login, viewing dashboard stats and student list, and deletion flow."""
        # 1. Create a teacher and a student in mock DB
        teacher_email = "test_teacher@school.edu"
        teacher_password = "password123"
        teacher_user = UserModel.create_user(teacher_email, teacher_password, "teacher")
        
        student_user = UserModel.create_user("student_test@school.edu", "password123", "student")
        student = StudentModel.create_student(
            name="Alice Smith",
            roll_no="202",
            branch="CSE",
            section="B",
            semester="6",
            photo_path="static/uploads/profile_photos/202_Alice Smith.jpg",
            embedding=[0.2]*512,
            user_id=student_user["_id"]
        )
        
        # Ensure they exist in DB
        self.assertEqual(self.mock_db.users.count_documents({}), 2)
        self.assertEqual(self.mock_db.students.count_documents({}), 1)
        
        # 2. Simulate Login
        response = self.client.post('/login', data={
            'email': teacher_email,
            'password': teacher_password
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dashboard", response.data)
        
        # Verify stats and student name are rendered on dashboard
        self.assertIn(b"Alice Smith", response.data)
        self.assertIn(b"202", response.data)
        self.assertIn(b"Total Students", response.data)
        
        # Verify stats show 0 present today initially (since no attendance record today)
        self.assertIn(b'<div class="stat-value" id="stat-present">0</div>', response.data)
        self.assertIn(b'<div class="stat-value" id="stat-absent">1</div>', response.data)
        
        # 2.5 Add a mock attendance record for today (without active session) and check dashboard updates stats
        import datetime
        today_db_str = datetime.date.today().strftime('%Y-%m-%d')
        self.mock_db.attendance.insert_one({
            "student_id": student["_id"],
            "session_id": ObjectId(), # random session id
            "date": today_db_str,
            "time": "10:00 AM",
            "status": "Present",
            "timestamp": datetime.datetime.utcnow()
        })
        
        dashboard_res = self.client.get('/teacher/dashboard')
        self.assertEqual(dashboard_res.status_code, 200)
        self.assertIn(b'<div class="stat-value" id="stat-present">1</div>', dashboard_res.data)
        self.assertIn(b'<div class="stat-value" id="stat-absent">0</div>', dashboard_res.data)
        
        # 3. Simulate Student Deletion API
        student_id_str = str(student["_id"])
        delete_response = self.client.post(f'/teacher/api/student/delete/{student_id_str}')
        self.assertEqual(delete_response.status_code, 200)
        
        # Assert database cascade deletion was successful
        self.assertEqual(self.mock_db.students.count_documents({}), 0)
        self.assertEqual(self.mock_db.users.count_documents({"role": "student"}), 0) # student login is deleted
        self.assertEqual(self.mock_db.users.count_documents({"role": "teacher"}), 1) # teacher is preserved

        # 4. Check dashboard page again (should not contain student name)
        dashboard_response = self.client.get('/teacher/dashboard')
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertNotIn(b"Alice Smith", dashboard_response.data)
        self.assertIn(b"No students registered in the database yet.", dashboard_response.data)

if __name__ == '__main__':
    unittest.main()
