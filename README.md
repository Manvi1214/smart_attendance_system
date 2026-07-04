# Smart Attendance System using Face Recognition 🎓🤖

A premium Web-based Smart Attendance System combining Computer Vision, Machine Learning, real-time webcam rendering, anti-spoofing (liveness checks), and MongoDB Atlas cloud storage.

## 🚀 Features

- **Modern Glassmorphic UI**: Sleek dark mode visual dashboard using vanilla CSS, layout grids, backdrop blurs, and neon accents.
- **Dynamic Face ID Registration**: Interactive enrollment flow showing on-screen guidance challenges (e.g. look straight, slowly turn left/right, tilt up/down) to collect multiple capture angles.
- **Real-Time Facial Processing**: 
  - **MTCNN**: Used for robust face detection and keypoints/landmarks localization.
  - **Landmark Alignment**: Rotates facial tilt based on coordinates of both eyes to align horizontally, boosting recognition accuracy.
  - **FaceNet**: Extracts 512-dimensional biometric embeddings.
  - **Euclidean Vector Comparisons**: Evaluates distance matches against database records.
- **Liveness Verification (Anti-Spoofing)**: Uses **MediaPipe Face Mesh** to calculate eye-aspect-ratio (EAR) and verify genuine human blinks before logging attendance. Fallbacks to bounding-box variance if MediaPipe is not installed.
- **MongoDB Atlas Integration**: Relational mappings using MongoDB Atlas collections for users, students, sessions, and attendance history logs.
- **Reporting Analytics**: Exporter endpoints to download daily attendance logs to CSV/Excel formats, and individual student reports to formatted PDFs.

---

## 📁 Directory Structure

```
Smart-Attendance-System/
├── app.py                      # Main Flask application entry point
├── config.py                   # App parameters & folder loaders
├── requirements.txt            # Python dependencies
├── README.md                   # Setup guide
│
├── database/
│   ├── models.py               # MongoDB Document query helpers
│   └── db_init.py              # MongoDB Atlas client connections
│
├── routes/
│   ├── auth_routes.py          # Session auth routes (Login/Logout/Register)
│   ├── teacher_routes.py       # Admin controls & registration sequence
│   ├── student_routes.py       # Student profiles & notifications
│   ├── attendance_routes.py    # Webcam detection and marking endpoints
│   └── report_routes.py        # CSV/Excel/PDF exporters
│
├── services/
│   ├── face_detection/         # MTCNN detector loader
│   ├── face_alignment/         # Face rotation correction
│   ├── face_recognition/       # FaceNet embedder & vector comparisons
│   ├── anti_spoofing/          # Blink tracking & EAR checks
│   └── attendance/             # Session logs aggregator
│
├── static/
│   ├── css/                    # Glassmorphic dark styling (Style, Teacher, Student)
│   ├── js/                     # Webcam loops, detection overlays, progress meters
│   └── uploads/                # Cached student profile photos
│
├── templates/                  # Auth, Teacher & Student dashboard HTML views
│
└── tests/                      # Unit test checkers (detection, embedding, DB models)
```

---

## 🛠️ Installation & Setup

### 1. Pre-requisites
Make sure you have Python 3.10+ installed on your system.

### 2. Clone and Install Dependencies
Navigate to the directory and run:
```bash
pip install -r requirements.txt
```

*(Note: `mongomock` is required only for running local model unit tests).*

### 3. Connect MongoDB Database
By default, the application checks for a local MongoDB database connection (`mongodb://localhost:27017/smart_attendance`).

To use **MongoDB Atlas**, create a `.env` file in the root directory and add your connection string:
```env
MONGO_URI="mongodb+srv://<username>:<password>@cluster.mongodb.net/smart_attendance?retryWrites=true&w=majority"
SECRET_KEY="your_custom_secret_session_key"
```

---

## 🚦 How to Run the App

1. Run the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5001/
   ```

### 👤 Default Credentials (Seeded Automatically)
- **Teacher Login**:
  - **Email**: `teacher@smartattendance.com`
  - **Password**: `password123`
- **Student Login**:
  - Set up by registering a student through the teacher dashboard. An email and password will be defined during registration which can then be used to log in as that student.

---

## 🧪 Running Unit Tests
To verify individual sub-modules and core ML features, run:
```bash
python -m unittest discover -s tests
```
