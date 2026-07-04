import os
# Force TensorFlow to run on CPU to avoid Metal Performance Shaders (MPS) Graph crashes on Apple Silicon
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import logging
from flask import Flask, render_template, redirect, url_for, session
from flask_cors import CORS
from config import Config
from database.db_init import db
from database.models import UserModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.BASE_DIR, 'logs', 'errors.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable Cross-Origin Resource Sharing
CORS(app)

# Register Blueprints
from routes.auth_routes import auth_bp
from routes.teacher_routes import teacher_bp
from routes.student_routes import student_bp
from routes.attendance_routes import attendance_bp
from routes.report_routes import report_bp

app.register_blueprint(auth_bp)
app.register_blueprint(teacher_bp)
app.register_blueprint(student_bp)
app.register_blueprint(attendance_bp)
app.register_blueprint(report_bp)

# Root route logic
@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'teacher':
            return redirect(url_for('teacher.dashboard'))
        elif session.get('role') == 'student':
            return redirect(url_for('student.dashboard'))
    return redirect(url_for('auth.login'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', error_title="404 - Page Not Found", error_message="The page you are looking for does not exist."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('base.html', error_title="500 - Internal Server Error", error_message="An internal server error occurred. Please check the logs."), 500

# Setup Seed Data on startup
@app.before_request
def setup_seed_data():
    # Only run once on startup
    if getattr(app, '_seed_done', False):
        return
    
    app._seed_done = True
    if db is not None:
        try:
            # Check if any teacher user exists
            teacher = db.users.find_one({"role": "teacher"})
            if not teacher:
                logger.info("No teacher user found. Creating default admin teacher...")
                # Seed a default teacher login
                UserModel.create_user(
                    email="teacher@smartattendance.com",
                    password="password123",
                    role="teacher"
                )
                logger.info("Default teacher seeded: teacher@smartattendance.com / password123")
        except Exception as e:
            logger.error(f"Error seeding database data: {e}")

if __name__ == '__main__':
    logger.info("Starting Smart Attendance Application Server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
