import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smart_attendance_super_secret_key_12345')
    
    # MongoDB configuration
    # By default, use local MongoDB or mock if MONGO_URI is not provided
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/smart_attendance')
    
    # Base directory configurations
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads', 'profile_photos')
    DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
    EMBEDDINGS_FOLDER = os.path.join(BASE_DIR, 'embeddings')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    # Session configurations
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Face recognition parameters
    # Lower value means stricter recognition. FaceNet recommended Euclidean distance threshold is 0.6.
    RECOGNITION_THRESHOLD = float(os.environ.get('RECOGNITION_THRESHOLD', 0.6))
    
    # Anti-spoofing parameters
    EYE_BLINK_THRESHOLD = 0.25  # EAR (Eye Aspect Ratio) under this is considered closed
    CONSECUTIVE_FRAMES_BLINK = 2 # Number of frames eye must remain closed to count as blink
    
# Make sure directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.DATASET_FOLDER, exist_ok=True)
os.makedirs(Config.EMBEDDINGS_FOLDER, exist_ok=True)
os.makedirs(Config.REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.REPORTS_DIR, 'daily'), exist_ok=True)
os.makedirs(os.path.join(Config.REPORTS_DIR, 'monthly'), exist_ok=True)
os.makedirs(os.path.join(Config.REPORTS_DIR, 'pdf'), exist_ok=True)
