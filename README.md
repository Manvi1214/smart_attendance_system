# Smart Attendance System with Face Recognition & Anti-Spoofing 🎓👁️⚡

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV%20%7C%20MediaPipe-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Deep Learning](https://img.shields.io/badge/DL-PyTorch%20%7C%20FaceNet%20%7C%20MTCNN-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Database](https://img.shields.io/badge/Database-MongoDB%20Atlas%20%7C%20MongoMock-47A248?style=flat&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![UI Design](https://img.shields.io/badge/UI-Premium%20Dark%20SaaS-6366F1?style=flat)](https://github.com/Manvi1214/smart_attendance_system)

An enterprise-grade, privacy-conscious **Automated Smart Attendance System** engineered for educational institutions and universities. Combines real-time Computer Vision, Deep Learning biometric recognition (**MTCNN** + **FaceNet**), active eye-blink liveness checks (**MediaPipe Face Mesh**), presentation attack detection (**Passive Anti-Spoofing**), and academic group segregation (**Branch, Semester, Section, and Subject**), presented through a **Premium Dark SaaS Dashboard**.

---

## 📌 Table of Contents

- [Key Features](#-key-features)
- [System Architecture & ML Pipeline](#-system-architecture--ml-pipeline)
- [Academic Hierarchy & Organization](#-academic-hierarchy--organization)
- [Teacher Portal & UI Suite](#-teacher-portal--ui-suite)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [API Reference](#-api-reference)
- [Database & Fallback Engine](#-database--fallback-engine)
- [Testing](#-testing)
- [License](#-license)

---

## 🚀 Key Features

### 1. Robust Biometric Verification & Liveness Protection
- **Multi-Task Cascaded Convolutional Networks (MTCNN)**: Fast, high-confidence bounding box detection and 5-point facial landmark localisation (eyes, nose, mouth corners).
- **2D Affine Pose Alignment**: Automatically corrects head roll and angular tilt based on interpupillary eye coordinates before feature extraction, ensuring pose invariance.
- **Inception-ResNet FaceNet**: Computes 512-dimensional normalized biometric embeddings per face with Euclidean/Cosine vector matching.
- **Active Single-Blink Liveness**: Real-time Eye Aspect Ratio (EAR) computation via 468-point 3D MediaPipe Face Mesh. Detects natural blinks to prevent static photo spoofing.
- **Passive Presentation Attack Detection (PAD)**: Frequency spectrum and color texture analysis to intercept printed photos, tablet screens, and replay video streams.

### 2. Branch, Section, Semester & Subject Segregation
- **Structured Student Enrollment**: Students are registered under specific academic departments (e.g. `CSE`, `IT`, `ECE`, `ME`, `CE`, `EE`), Semesters (`1st` to `8th`), and Sections (`A`, `B`, `C`, `D`).
- **Targeted Attendance Sessions**: Attendance sessions are created for a designated Branch + Semester + Section + Subject.
- **Zero Cross-Contamination**: Face recognition matches live camera feeds **only against students enrolled in that specific class session**. Unrelated students or students from other sections/branches are excluded.

### 3. Manual Attendance Correction & Audit Trail
- **In-Session & Post-Session Modification**: Teachers can manually adjust any student's status between **Present** and **Absent**.
- **Audit Compliance**: Mandatory reason tracking with timestamped audit logs recording who made the change, the original status, and the rationale.

### 4. Multi-Format Analytics & Report Exports
- **Instant CSV Export**: Fast raw tabular session records.
- **Microsoft Excel (.xlsx) Export**: Formatted spreadsheets via Pandas and OpenPyXL.
- **Official PDF Dossiers**: Publication-quality, letterhead-styled individual student attendance reports via ReportLab.
- **Visual Analytics**: Interactive Chart.js attendance rate distribution, subject-specific attendance health, and high-risk defaulter alerts.

### 5. Premium Dark SaaS Interface
- Modern dark charcoal and deep navy theme (`#070b14` / `#0f172a`), translucent glassmorphic surfaces, indigo/violet primary accents, emerald green success badges, and Inter typography.
- Mobile-responsive sidebar navigation with collapsible drawer controls.

---

## 🧠 System Architecture & ML Pipeline

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

## 🏫 Academic Hierarchy & Organization

The system enforces strict academic group isolation across all workflows:

```
University
 └── Branch (CSE, IT, ECE, ME, CE, EE)
      └── Semester (Semester 1 to Semester 8)
           └── Section (Section A, B, C, D)
                └── Subject (DBMS, OS, CN, DSA, AI/ML, etc.)
                     └── Eligible Students Roster
```

- When a teacher creates a session for **CSE • 5th Semester • Section A • DBMS**, the facial recognition engine queries the database specifically for students enrolled in **CSE / 5 / A**.
- Students belonging to Section B or other branches will **not** be recognized or logged into this session, ensuring 100% data integrity.

---

## 🖥️ Teacher Portal & UI Suite

| Page | Route | Description |
| :--- | :--- | :--- |
| **Teacher Dashboard** | `/teacher/dashboard` | Central command center with 4 live KPI cards (*Total Students, Present Today, Absent Today, Attendance Rate*), compact 3-row session creator, high-risk defaulters list, and recent sessions table with one-click manual view modal. |
| **Register Student** | `/teacher/register_student` | 2-step onboarding wizard: Form details entry followed by live camera face enrollment with circular viewport guide and progress fill meter. |
| **Student Management** | `/teacher/students` | Filterable directory by Branch, Semester, Section, and Subject with real-time text search, profile avatars, contact details, and student record management. |
| **Start Live Attendance** | `/teacher/start_attendance` | Full-screen live attendance HUD with real-time facial bounding box, eye-blink verification badge, anti-spoofing indicators, attendee counters, and real-time marking logs. |
| **Session History** | `/teacher/attendance_history` | Historical session records with filter toolbar, attendee totals, percentage badges, direct CSV/Excel export buttons, and inline attendance editor. |
| **Dedicated Session View** | `/teacher/session/<id>/view` | Printable session sheet with full student roster, inline Present/Absent status editor, and correction audit log. |
| **Reports & Export Hub** | `/teacher/reports` | Multi-filter student dossiers directory with instant generation of official letterhead PDF attendance records. |
| **Analytics** | `/teacher/analytics` | Interactive Chart.js doughnut chart, subject health breakdown, and defaulter tracking (< 75% attendance threshold). |

---

## 📁 Project Directory Structure

```
Smart_Attendance/
├── app.py                         # Application factory, blueprint registration & socket binding
├── config.py                      # System configuration, paths, model thresholds
├── requirements.txt               # Production Python dependencies
├── run.sh                         # Quick startup shell script
├── README.md                      # Project documentation
│
├── database/
│   ├── db_init.py                 # Multi-tier database connector (Atlas -> Local -> MongoMock)
│   └── models.py                  # Data models (User, Student, Session, Attendance, Correction)
│
├── routes/
│   ├── auth_routes.py             # Login, registration, session management, and API auth
│   ├── teacher_routes.py          # Teacher portal views, session APIs, and attendance correction
│   ├── student_routes.py          # Student dashboard & self-service calendar views
│   ├── attendance_routes.py       # Live stream ingestion, face recognition & marking pipeline
│   └── report_routes.py           # Multi-format report generators (CSV, Excel, PDF)
│
├── services/
│   ├── face_detection/            # MTCNN face detection service
│   ├── face_alignment/            # Affine landmark horizontal transformation
│   ├── face_recognition/          # FaceNet embedding generation & cosine/Euclidean distance matching
│   ├── anti_spoofing/             # MediaPipe EAR blink detection & passive texture analysis
│   └── attendance/                # Attendance aggregation & session state tracking
│
├── static/
│   ├── css/
│   │   ├── style.css              # Global design tokens, typography, glassmorphism, buttons, inputs
│   │   └── teacher.css            # Dashboard layout, sidebar drawer, stat cards, tables, camera HUD
│   ├── js/
│   │   ├── webcam.js              # Camera stream capture, canvas frame extraction
│   │   ├── attendance.js          # Live recognition loop, HUD challenge updates, toast alerts
│   │   └── teacher_dashboard.js   # Dynamic filtering, session management, and modal controls
│   └── uploads/
│       └── profile_photos/        # Enrolled student face crop storage
│
├── templates/
│   ├── base.html                  # Base HTML layout, toast container, font definitions
│   ├── auth/
│   │   ├── login.html             # SaaS login page with password toggle
│   │   └── register.html          # Teacher registration page
│   └── teacher/
│       ├── dashboard.html         # Main overview dashboard
│       ├── register_student.html  # Two-step student enrollment
│       ├── students.html          # Filterable student management roster
│       ├── start_attendance.html  # Live webcam attendance HUD
│       ├── attendance_history.html# Session history table & export triggers
│       ├── session_view.html      # Printable dedicated session attendance sheet
│       ├── reports.html           # Student PDF report dossiers hub
│       ├── analytics.html         # Visual analytics & defaulter tracking
│       └── components/
│           ├── sidebar.html       # Unified reusable navigation sidebar
│           └── session_details_modal.html # Modal for reviewing/editing session attendance
│
├── dataset/                       # Raw frame captures collected during student enrollment
├── embeddings/                    # Serialized NumPy (.npy) FaceNet biometric vectors
└── tests/                         # Test suites for detection, recognition, liveness & DB
```

---

## 🛠️ Installation & Quickstart

### 1. Prerequisites
- **Python**: Version 3.10 or 3.11 recommended.
- **Webcam**: Standard USB or built-in webcam for facial capture.
- **Operating System**: macOS, Linux, or Windows 10/11.

### 2. Clone the Repository
```bash
git clone https://github.com/Manvi1214/smart_attendance_system.git
cd smart_attendance_system
```

### 3. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the Application
```bash
python app.py
```

The application will start on **`http://127.0.0.1:5001`**.

---

## ⚙️ Configuration & Environment Variables

Create an optional `.env` file in the project root to configure cloud database connectivity or custom secret keys:

```env
# Flask Secret Key for secure session cookies
SECRET_KEY="your-strong-random-secret-key"

# MongoDB Atlas Connection URI (optional - system falls back gracefully if omitted)
MONGO_URI="mongodb+srv://<username>:<password>@cluster0.mongodb.net/smart_attendance?retryWrites=true&w=majority"

# Optional Server Port (default: 5001)
PORT=5001
DEBUG=True
```

---

## 👤 Default Credentials

On first run, the database automatically seeds an initial administrator teacher account:

| Role | Email | Password |
| :--- | :--- | :--- |
| **Teacher / Admin** | `teacher@smartattendance.com` | `password123` |

> *Tip: You can change the password or register additional teachers directly from the interface.*

---

## 📡 API Reference

### Teacher & Session Operations
- `POST /teacher/api/start_session` — Create a new session with branch, semester, section, and subject.
- `POST /teacher/api/end_session` — Terminate active session and finalize attendance percentages.
- `GET /teacher/api/stats` — Retrieve live summary metrics for the active teacher.
- `GET /teacher/api/students/filter` — Filter student directory dynamically by branch, semester, section, and subject.
- `GET /teacher/api/session/<session_id>/attendance` — Retrieve detailed attendance roster and audit modifications for a session.
- `POST /teacher/api/attendance/correct` — Manually modify student attendance status (`Present` $\leftrightarrow$ `Absent`) with a mandatory reason.

### Live Face Recognition
- `POST /attendance/process_frame` — Ingest live video frame, execute MTCNN detection, verify eye blink / anti-spoofing, compute FaceNet embedding, and record verified attendance.

### Reports & Exports
- `GET /reports/export/csv/<session_id>` — Download raw session attendance as a CSV file.
- `GET /reports/export/excel/<session_id>` — Download formatted session attendance as an Excel spreadsheet (.xlsx).
- `GET /reports/export/pdf/<student_id>` — Generate and download official letterhead attendance dossier PDF for a student.

---

## 🗄️ Database & Fallback Engine

The system features a **3-tier resilient database connection strategy**:

1. **Tier 1 (MongoDB Atlas)**: If a `MONGO_URI` is provided and reachable, connections are routed directly to the cloud cluster.
2. **Tier 2 (Local MongoDB)**: If Atlas is unreachable, the system automatically attempts to connect to a local MongoDB daemon at `mongodb://localhost:27017/smart_attendance`.
3. **Tier 3 (In-Memory MongoMock Fallback)**: If neither remote nor local MongoDB is running, the system transparently spins up an in-memory `mongomock` instance. This allows instant testing, evaluation, and offline operation with **zero setup configuration required**.

---

## 🧪 Testing

Comprehensive unit tests are provided in the `tests/` directory:

```bash
# Run all unit tests
python -m unittest discover -s tests -v

# Run individual test modules
python -m unittest tests/test_face_detection.py
python -m unittest tests/test_recognition.py
python -m unittest tests/test_single_blink_and_pad.py
python -m unittest tests/test_dashboard_flow.py
```

---

## 🛡️ Anti-Spoofing & Security Measures

- **Active Blink Validation**: Eye Aspect Ratio (EAR) metric tracks eye openness across sequential frames ($\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}$). Attendance is recorded only after genuine human blinks are recorded.
- **Passive Frequency Spectrum Check**: FFT (Fast Fourier Transform) analysis identifies the high-frequency moiré patterns and unnatural reflections characteristic of digital screens and printed photos.
- **Session Expiry Protection**: API endpoints enforce strict JSON 401 error handling to eliminate malformed response exceptions.

---

## 👥 Contributors

- **Manvi Agrawal** — [GitHub Profile](https://github.com/Manvi1214)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
