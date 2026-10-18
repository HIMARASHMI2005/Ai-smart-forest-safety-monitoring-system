import base64
import json
import os
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from flask import Blueprint, request, jsonify
from backend.app.config import Config
from backend.app.database.connection import SessionLocal
from backend.app.vision.pipeline import VisionPipeline

vision_bp = Blueprint("vision_bp", __name__)

def decode_image(raw_data) -> Optional[np.ndarray]:
    if isinstance(raw_data, str):
        if "base64," in raw_data:
            raw_data = raw_data.split("base64,")[1]
        img_bytes = base64.b64decode(raw_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif hasattr(raw_data, "read"):
        img_bytes = raw_data.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return None

@vision_bp.route("/infer-frame", methods=["POST"])
def infer_frame():
    """
    Tactical frame inspection endpoint.
    Runs frame through ForestPersonNet detector and ForestFaceNet biometric matcher.
    Returns: Bounding Boxes, Centroids, Identities, FPS, Zone, and Latency.
    Enforces 100% detection accuracy filter by default.
    """
    data = request.get_json() or {}
    image_b64 = data.get("frame")
    camera_zone = data.get("camera_zone", "ZONE A: ECO TRAIL")
    require_100 = data.get("require_100_accuracy", True)

    if not image_b64:
        return jsonify({"error": "frame payload is required (base64 string)"}), 400

    frame_bgr = decode_image(image_b64)
    if frame_bgr is None:
        return jsonify({"error": "Failed to decode frame"}), 400

    db = SessionLocal()
    try:
        result = VisionPipeline.process_frame(
            frame_bgr,
            db,
            camera_zone=camera_zone,
            require_100_accuracy=require_100
        )
        return jsonify(result), 200
    finally:
        db.close()

@vision_bp.route("/metrics", methods=["GET"])
@vision_bp.route("/ml/metrics", methods=["GET"])
def get_ml_metrics():
    """
    Returns unedited, measured evaluation metrics directly from actual training outputs.
    """
    base_dir = Config.BASE_DIR
    person_metrics_path = base_dir / "ml/checkpoints/person_detection/evaluation_results.json"
    face_metrics_path = base_dir / "ml/checkpoints/face_recognition/evaluation_results.json"

    person_metrics = {}
    face_metrics = {}

    if os.path.exists(person_metrics_path):
        with open(person_metrics_path, "r") as f:
            person_metrics = json.load(f)

    if os.path.exists(face_metrics_path):
        with open(face_metrics_path, "r") as f:
            face_metrics = json.load(f)

    return jsonify({
        "forest_person_net": person_metrics,
        "forest_face_net": face_metrics,
        "tenets": {
            "zero_pretrained_weights": True,
            "deterministic_geofencing": "Shapely Ray Casting",
            "deterministic_risk": "Multi-variable Weighted Matrix",
            "biometric_embeddings": "128-D Unit Hypersphere (ArcFace)"
        }
    }), 200
