import unittest
# pyrefly: ignore [missing-import]
from bson import ObjectId
# pyrefly: ignore [missing-import]
import mongomock
import database.db_init as db_init
from database.models import AttendanceModel, SessionModel, StudentModel, UserModel

class TestAttendanceOperations(unittest.TestCase):
    def setUp(self):
        # Override the db client with a mock client for testing
        self.mock_client = mongomock.MongoClient()
        self.mock_db = self.mock_client.smart_attendance_test
        self.original_db = db_init.db
        db_init.db = self.mock_db
        
        # Setup schema indexes on mock db
        db_init.init_indexes(self.mock_db)

    def tearDown(self):
        # Restore original database client
        db_init.db = self.original_db

    def test_create_user(self):
        """Test UserModel creation in mock DB."""
        user = UserModel.create_user("test_user@school.edu", "password123", "student")
        self.assertIsNotNone(user)
        self.assertEqual(user["email"], "test_user@school.edu")
        self.assertEqual(user["role"], "student")
        
        # Verify hash match
        verified = UserModel.verify_user("test_user@school.edu", "password123")
        self.assertIsNotNone(verified)

    def test_mark_attendance(self):
        """Test Attendance marking and duplicate checking in mock DB."""
        # Create user and student
        user = UserModel.create_user("krish@school.edu", "password123", "student")
        student = StudentModel.create_student(
            name="Krish",
            roll_no="101",
            branch="CSE",
            section="A",
            semester="4",
            photo_path="static/uploads/profile_photos/101_Krish.jpg",
            embedding=[0.1]*512,
            user_id=user["_id"]
        )
        
        # Start session
        session = SessionModel.start_session("Machine Learning", ObjectId())
        
        # Mark present
        record = AttendanceModel.mark_present(student["_id"], session["_id"])
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "Present")
        
        # Mark duplicate (should return original record without crash)
        duplicate_record = AttendanceModel.mark_present(student["_id"], session["_id"])
        self.assertEqual(str(record["_id"]), str(duplicate_record["_id"]))

if __name__ == '__main__':
    unittest.main()
