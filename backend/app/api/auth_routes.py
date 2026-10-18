import base64
import uuid
import hashlib
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from flask import Blueprint, request, jsonify, g
from backend.app.database.connection import SessionLocal
from backend.app.database.models import User, Visitor, Officer, BiometricTemplate
from backend.app.api.auth_middleware import generate_jwt_token, token_required, role_required
from backend.app.vision.face_service import FaceService

auth_bp = Blueprint("auth_bp", __name__)

def hash_password(password: str) -> str:
    # Deterministic salted SHA-256 for cross-platform portability
    salt = "smart_forest_salt_2026"
    return hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(plain_password: str, hashed: str) -> bool:
    return hash_password(plain_password) == hashed

def decode_image_payload(raw_data) -> Optional[np.ndarray]:
    """Decodes image from base64 string or multipart file."""
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

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    role = data.get("role", "VISITOR").upper()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if role not in ["ADMIN", "OFFICER", "VISITOR"]:
        return jsonify({"error": "Invalid role specified"}), 400

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return jsonify({"error": "Email already registered"}), 409

        user_id = str(uuid.uuid4())
        user_status = "ACTIVE" if role != "OFFICER" else "PENDING"  # Officers require admin approval

        user = User(
            id=user_id,
            email=email,
            password_hash=hash_password(password),
            role=role,
            status=user_status
        )
        db.add(user)

        profile_data = {}
        if role == "VISITOR":
            visitor_id = data.get("visitor_id") or f"VST-{uuid.uuid4().hex[:6].upper()}"
            visitor = Visitor(
                visitor_id=visitor_id,
                user_id=user_id,
                full_name=data.get("full_name", "Registered Visitor"),
                phone=data.get("phone", "N/A"),
                emergency_contact_name=data.get("emergency_contact_name", "N/A"),
                emergency_contact_phone=data.get("emergency_contact_phone", "N/A"),
                current_zone="OUT_OF_BOUNDS",
                current_risk_level="LOW",
                status="REGISTERED"
            )
            db.add(visitor)
            profile_data = visitor.to_dict()

        elif role == "OFFICER":
            officer_id = data.get("officer_id") or f"OFF-{uuid.uuid4().hex[:6].upper()}"
            badge = data.get("badge_number") or f"BADGE-{uuid.uuid4().hex[:4].upper()}"
            officer = Officer(
                officer_id=officer_id,
                user_id=user_id,
                badge_number=badge,
                full_name=data.get("full_name", "Field Officer"),
                phone=data.get("phone", "N/A"),
                department=data.get("department", "Wildland Rescue & Ranger Division"),
                approval_status="PENDING",
                availability_status="AVAILABLE"
            )
            db.add(officer)
            profile_data = officer.to_dict()

        db.commit()

        token = generate_jwt_token(user)
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "user": user.to_dict(),
            "profile": profile_data
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500
    finally:
        db.close()

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    expected_role = (data.get("expected_role") or data.get("portal") or "").strip().upper()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password_hash):
            return jsonify({"error": "Invalid email or password"}), 401

        if user.status == "SUSPENDED":
            return jsonify({"error": "Account is suspended"}), 403

        # Role isolation enforcement
        if expected_role == "VISITOR" and user.role != "VISITOR":
            return jsonify({
                "error": "Access Denied: Officer accounts cannot log in to the Visitor Portal. Please use the Officer Tactical Portal at /officer/login."
            }), 403

        if expected_role == "OFFICER" and user.role != "OFFICER":
            return jsonify({
                "error": "Access Denied: Visitor accounts cannot log in to the Officer Terminal. Please use the Visitor Portal at /visitor/login."
            }), 403

        token = generate_jwt_token(user)

        profile = None
        if user.role == "VISITOR" and user.visitor:
            profile = user.visitor.to_dict()
        elif user.role == "OFFICER" and user.officer:
            profile = user.officer.to_dict()

        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": user.to_dict(),
            "profile": profile
        }), 200
    finally:
        db.close()

@auth_bp.route("/face-register", methods=["POST"])
@token_required
def face_register():
    """
    Ingests face capture, runs quality gating, executes ForestFaceNet forward pass,
    and commits 128-D L2-normalized embedding to biometric_templates.
    """
    data = request.get_json() or {}
    image_b64 = data.get("image")
    face_pose_index = int(data.get("face_pose_index", 0))  # 0: Front, 1: Left, 2: Right, 3: Up, 4: Down

    if not image_b64:
        return jsonify({"error": "No image payload provided"}), 400

    frame_bgr = decode_image_payload(image_b64)
    if frame_bgr is None:
        return jsonify({"error": "Failed to decode image data"}), 400

    db = SessionLocal()
    try:
        visitor = db.query(Visitor).filter(Visitor.user_id == g.current_user.id).first()
        if not visitor:
            return jsonify({"error": "Visitor profile not found for this account"}), 404

        success, emb_128, quality_score, msg = FaceService.process_and_extract_embedding(frame_bgr)
        if not success or emb_128 is None:
            return jsonify({
                "success": False,
                "error": msg,
                "quality_score": quality_score
            }), 422

        # Check existing pose template
        existing_template = db.query(BiometricTemplate).filter(
            BiometricTemplate.visitor_id == visitor.visitor_id,
            BiometricTemplate.face_pose_index == face_pose_index
        ).first()

        emb_bytes = emb_128.astype(np.float32).tobytes()

        if existing_template:
            existing_template.embedding_vector = emb_bytes
            existing_template.quality_score = quality_score
            existing_template.created_at = datetime.utcnow()
        else:
            new_template = BiometricTemplate(
                visitor_id=visitor.visitor_id,
                embedding_vector=emb_bytes,
                model_version="ForestFaceNet_v1",
                quality_score=quality_score,
                face_pose_index=face_pose_index
            )
            db.add(new_template)

        db.commit()

        # Count total registered poses for this visitor
        total_poses = db.query(BiometricTemplate).filter(
            BiometricTemplate.visitor_id == visitor.visitor_id
        ).count()

        return jsonify({
            "success": True,
            "message": "Biometric face template registered successfully",
            "visitor_id": visitor.visitor_id,
            "face_pose_index": face_pose_index,
            "quality_score": quality_score,
            "total_registered_poses": total_poses,
            "is_registration_complete": total_poses >= 5
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Face registration failed: {str(e)}"}), 500
    finally:
        db.close()

@auth_bp.route("/face-verify", methods=["POST"])
def face_verify():
    """
    Biometric verification login.
    Compares captured face against candidate template via Cosine Distance:
    Similarity = (u . v) / (||u|| * ||v||).
    Accepts if Similarity >= tau_val (0.68).
    """
    data = request.get_json() or {}
    image_b64 = data.get("image")
    visitor_id = data.get("visitor_id")

    if not image_b64:
        return jsonify({"error": "Image payload is required"}), 400

    frame_bgr = decode_image_payload(image_b64)
    if frame_bgr is None:
        return jsonify({"error": "Failed to decode image data"}), 400

    db = SessionLocal()
    try:
        success, emb_128, quality_score, msg = FaceService.process_and_extract_embedding(frame_bgr)
        if not success or emb_128 is None:
            return jsonify({
                "verified": False,
                "error": msg,
                "quality_score": quality_score
            }), 422

        if visitor_id:
            # 1:1 Verification
            is_match, similarity, status_msg = FaceService.verify_visitor_face(db, visitor_id, emb_128)
            matched_visitor = db.query(Visitor).filter(Visitor.visitor_id == visitor_id).first() if is_match else None
        else:
            # 1:N Identification
            matched_id, similarity, status_msg = FaceService.identify_face(db, emb_128)
            is_match = matched_id is not None
            matched_visitor = db.query(Visitor).filter(Visitor.visitor_id == matched_id).first() if is_match else None

        if not is_match or not matched_visitor:
            return jsonify({
                "verified": False,
                "message": status_msg,
                "similarity": similarity,
                "threshold": 0.68
            }), 401

        # Generate session token for verified visitor
        token = generate_jwt_token(matched_visitor.user)

        return jsonify({
            "verified": True,
            "message": "Biometric face verification successful",
            "similarity": similarity,
            "threshold": 0.68,
            "token": token,
            "visitor": matched_visitor.to_dict(),
            "user": matched_visitor.user.to_dict()
        }), 200

    finally:
        db.close()

@auth_bp.route("/me", methods=["GET"])
@token_required
def get_me():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == g.current_user.id).first()
        if not user:
            return jsonify({"error": "User account no longer exists"}), 404
        profile = None
        if user.role == "VISITOR" and user.visitor:
            profile = user.visitor.to_dict()
        elif user.role == "OFFICER" and user.officer:
            profile = user.officer.to_dict()

        res = {
            "user": user.to_dict(),
            "profile": profile,
            "role": user.role
        }
        if user.role == "OFFICER":
            res["officer"] = profile
        elif user.role == "VISITOR":
            res["visitor"] = profile

        return jsonify(res), 200
    finally:
        db.close()
