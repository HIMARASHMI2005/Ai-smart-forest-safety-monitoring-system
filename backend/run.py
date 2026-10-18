import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import Config
from backend.app import create_app
from backend.app.websocket.events import socketio
from backend.app.database.connection import SessionLocal
from sqlalchemy import text
from backend.app.database.models import Zone
from backend.app.geofence.geometry_engine import GeometryEngine
from backend.app.geofence.zone_manager import ZoneManager

def bootstrap_system():
    """Initializes directories, seeds default geofence zones, and ensures evaluation metrics exist."""
    print("=" * 70)
    print("SMART FOREST GEO-FENCING & EMERGENCY MONITORING SYSTEM")
    print("Command & Control Gateway Bootstrapping...")
    print("=" * 70)

    # 1. Create sandboxed directories
    for folder in [Config.UPLOAD_FOLDER, Config.SOS_VIDEO_FOLDER, Config.FACE_CROPS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # 2. Seed Geofence Zones if empty
    db = SessionLocal()
    try:
        existing_zones = db.query(Zone).count()
        if existing_zones == 0:
            print("Seeding initial Bandipur-Mudumalai Reserve geofence zones...")
            seed_sql_path = BASE_DIR / "database" / "seed_zones.sql"
            if os.path.exists(seed_sql_path):
                with open(seed_sql_path, "r") as f:
                    statements = f.read().split(";")
                    for stmt in statements:
                        s = stmt.strip()
                        if s:
                            db.execute(text(s))
                db.commit()
                print("Seeded 4 tactical geofence zones successfully.")
        ZoneManager.load_zones(db)
        print(f"Loaded {len(ZoneManager._zone_cache)} active geofence zones into memory.")
    except Exception as e:
        print(f"Warning during zone bootstrap: {e}")
        db.rollback()
    finally:
        db.close()

    # 3. Ensure official ML evaluation metrics files exist
    person_metrics = BASE_DIR / "ml/checkpoints/person_detection/evaluation_results.json"
    face_metrics = BASE_DIR / "ml/checkpoints/face_recognition/evaluation_results.json"

    if not os.path.exists(person_metrics):
        print("Generating ForestPersonNet baseline evaluation results...")
        try:
            from ml.forest_person_net.evaluate import evaluate_test_set
            evaluate_test_set(output_metrics_file=str(person_metrics))
        except Exception as e:
            print(f"Warning running ForestPersonNet evaluation: {e}")

    if not os.path.exists(face_metrics):
        print("Generating ForestFaceNet baseline evaluation results...")
        try:
            from ml.forest_face_net.evaluate import evaluate_face_verification
            evaluate_face_verification(output_metrics_file=str(face_metrics))
        except Exception as e:
            print(f"Warning running ForestFaceNet evaluation: {e}")

    print("System bootstrap complete. Ready for real-time operations.")
    print("=" * 70)

app = create_app()

if __name__ == "__main__":
    bootstrap_system()
    port = Config.PORT
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"Port {port} is currently in use, falling back to port 5001...")
            port = 5001

    print(f"Starting Tactical Command Gateway on http://{Config.HOST}:{port}...")
    socketio.run(
        app,
        host=Config.HOST,
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True
    )
