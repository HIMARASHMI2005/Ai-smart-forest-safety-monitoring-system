import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for the smart-forest backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "backend" / ".env")
load_dotenv(BASE_DIR / ".env")

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.getenv("SECRET_KEY", "smart-forest-tactical-secret-key-2026-secure")
    JWT_SECRET = os.getenv("JWT_SECRET", "smart-forest-jwt-hmac-sha256-key-998811")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # Database
    DB_PATH = BASE_DIR / "database" / "smart_forest.db"
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

    # File uploads & chunk storage
    UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads")
    SOS_VIDEO_FOLDER = BASE_DIR / os.getenv("SOS_VIDEO_FOLDER", "uploads/sos_videos")
    FACE_CROPS_FOLDER = BASE_DIR / os.getenv("FACE_CROPS_FOLDER", "uploads/face_crops")

    # ML Checkpoint Paths
    PERSON_DETECTION_CHECKPOINT = BASE_DIR / os.getenv(
        "PERSON_DETECTION_CHECKPOINT", "ml/checkpoints/person_detection/forest_person_net_best.pt"
    )
    FACE_RECOGNITION_CHECKPOINT = BASE_DIR / os.getenv(
        "FACE_RECOGNITION_CHECKPOINT", "ml/checkpoints/face_recognition/forest_face_net_best.pt"
    )

    # ML Decision Thresholds
    FACE_VERIFY_THRESHOLD = float(os.getenv("FACE_VERIFY_THRESHOLD", "0.68"))
    PERSON_DETECTION_CONF_THRESHOLD = float(os.getenv("PERSON_DETECTION_CONF_THRESHOLD", "0.35"))
    PERSON_DETECTION_IOU_THRESHOLD = float(os.getenv("PERSON_DETECTION_IOU_THRESHOLD", "0.45"))

    # Web App Port
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Static / Template folders for 3 web apps
    ADMIN_WEB_DIR = BASE_DIR / "admin-web"
    VISITOR_WEB_DIR = BASE_DIR / "visitor-web"
    OFFICER_WEB_DIR = BASE_DIR / "officer-web"
