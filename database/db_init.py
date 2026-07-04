import logging
# pyrefly: ignore [missing-import]
from pymongo import MongoClient
# pyrefly: ignore [missing-import]
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global mongo instances
mongo_client = None
_db_instance = None

def get_db():
    global mongo_client, _db_instance
    if _db_instance is not None:
        return _db_instance
    
    try:
        # Establish Connection
        logger.info(f"Connecting to MongoDB with URI: {Config.MONGO_URI.split('@')[-1]}")  # Logs only host/db info for privacy
        mongo_client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test Connection (force select server)
        mongo_client.admin.command('ping')
        
        # Select database (extract db name from URI or use default)
        # URI format: mongodb+srv://username:password@cluster.mongodb.net/databaseName
        db_name = 'smart_attendance'
        try:
            _db_instance = mongo_client.get_default_database()
        except Exception:
            _db_instance = mongo_client[db_name]
            
        logger.info(f"Successfully connected to MongoDB database: {_db_instance.name}")
        
        # Create indexes to ensure uniqueness
        init_indexes(_db_instance)
        
        return _db_instance
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        # Try fallback to local MongoDB if Atlas connection fails
        logger.info("Attempting connection to local MongoDB fallback: mongodb://localhost:27017/smart_attendance")
        try:
            mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            mongo_client.admin.command('ping')
            _db_instance = mongo_client['smart_attendance']
            logger.info("Connected to fallback local MongoDB successfully.")
            init_indexes(_db_instance)
            return _db_instance
        except Exception as local_err:
            logger.error(f"Local fallback connection also failed: {local_err}")
            # Do not raise immediately to prevent app crash if DB is not ready during startup
            # The app will handle None db gracefully or throw on operations
            return None

def init_indexes(database):
    try:
        # Users indexes
        database.users.create_index("email", unique=True)
        # Students indexes
        database.students.create_index("roll_no", unique=True)
        # Attendance compound index to prevent duplicate daily markings per student per session
        database.attendance.create_index([("student_id", 1), ("session_id", 1), ("date", 1)], unique=True)
        logger.info("MongoDB database collections and indexes verified.")
    except Exception as e:
        logger.warning(f"Error creating indexes: {e}")

class DatabaseProxy:
    def __getattr__(self, name):
        import sys
        module = sys.modules.get('database.db_init')
        if module and hasattr(module, 'db') and module.db is not self:
            target = module.db
        else:
            global _db_instance
            if _db_instance is None:
                _db_instance = get_db()
            target = _db_instance
            
        if target is None:
            raise RuntimeError("Database not connected")
        return getattr(target, name)
        
    def __bool__(self):
        import sys
        module = sys.modules.get('database.db_init')
        if module and hasattr(module, 'db') and module.db is not self:
            return module.db is not None
        global _db_instance
        if _db_instance is None:
            _db_instance = get_db()
        return _db_instance is not None

# The proxy object imported by other files
db = DatabaseProxy()

