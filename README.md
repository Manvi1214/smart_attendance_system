# Smart Attendance System

A web-based attendance management system using Computer Vision, Deep Learning, and academic group segregation. The system identifies students using facial recognition, validates liveness through blink detection and passive anti-spoofing, and manages records by Branch, Semester, Section, and Subject.

---

## Features

- **Face Recognition**: MTCNN for face detection and 5-point landmark localization; FaceNet (Inception-ResNet) for generating 512-dimensional biometric embeddings.
- **Liveness & Anti-Spoofing**: Eye Aspect Ratio (EAR) single-blink detection via MediaPipe Face Mesh, combined with frequency spectrum and texture analysis to prevent photo and video spoofing.
- **Academic Hierarchy**: Students and sessions are categorized by Branch (CSE, IT, ECE, ME, CE, EE), Semester (1 to 8), Section (A to D), and Subject. Attendance sessions recognize only students enrolled in the selected group.
- **Manual Attendance Correction**: Teachers can modify attendance status (Present / Absent) with mandatory reason audit logging.
- **Reports & Analytics**: Instant export of attendance data to CSV, Excel (.xlsx), and student PDF reports. Interactive dashboard analytics for attendance rates and defaulter tracking.
- **Database Fallback**: Multi-tier connection handling supporting MongoDB Atlas, local MongoDB, and an in-memory MongoMock fallback for instant zero-configuration local runs.

---

## System Architecture & ML Pipeline

```
  +------------------+
  | Live Webcam Feed |
  +--------+---------+
           | (RGB Video Stream)
           v
  +------------------+
  |  MTCNN Detector  | ----> [No Face] ----> Continue Scanning
  +--------+---------+
           | (Bounding Box + 5 Landmarks)
           v
  +------------------+
  | Affine Alignment | ----> Normalizes head tilt & rotation
  +--------+---------+
           |
           +-----------------------------+
           |                             |
           v                             v
  +------------------+          +------------------+
  | MediaPipe Mesh   |          | Passive PAD /    |
  | (EAR Eye Blink)  |          | Texture Analysis |
  +--------+---------+          +--------+---------+
           |                             |
           +--------------+--------------+
                          |
             [Liveness Challenge Passed]
                          |
                          v
               +--------------------+
               |  FaceNet Embedder  |
               |  (512-D Vectors)   |
               +----------+---------+
                          |
                          v
               +--------------------+
               | Vector Comparison  | <--- Filtered Session Roster
               | (Distance < 0.85)  |      (Target Branch/Sec/Sem)
               +----------+---------+
                          |
                  [Match Identified]
                          |
                          v
               +--------------------+
               | MongoDB Attendance |
               | Logged & Confirmed |
               +--------------------+
```

---

## Project Structure

```
Smart_Attendance/
├── app.py                      # Flask application entry point
├── config.py                   # Configuration and model thresholds
├── requirements.txt            # Python dependencies
├── database/                   # MongoDB connection and data models
├── routes/                     # Auth, Teacher, Student, Attendance, and Report routes
├── services/                   # MTCNN, FaceNet, MediaPipe blink, and anti-spoofing logic
├── static/                     # CSS stylesheets, JavaScript handlers, profile photos
├── templates/                  # HTML templates (Auth, Teacher, Student views)
└── tests/                      # Unit test suites
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10 or 3.11
- Webcam

### 2. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5001/
```

---

## Default Credentials

The system automatically initializes an administrator teacher account on first launch:

- **Email**: `teacher@smartattendance.com`
- **Password**: `password123`

---

## Configuration

Optional environment variables can be set in a `.env` file:

```env
SECRET_KEY="your-secret-key"
MONGO_URI="mongodb+srv://<username>:<password>@cluster0.mongodb.net/smart_attendance"
PORT=5001
DEBUG=True
```

---

## Testing

Run unit tests:
```bash
python -m unittest discover -s tests -v
```

---

## License

This project is licensed under the MIT License.
