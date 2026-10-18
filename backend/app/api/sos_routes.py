import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from backend.app.database.connection import SessionLocal
from backend.app.database.models import SosEvent, Visitor, Alert, RiskEvent, SosVideoChunk, Officer
from backend.app.sos.video_handler import VideoHandler
from backend.app.api.auth_middleware import token_required, role_required
from backend.app.websocket.events import (
    broadcast_emergency_alert,
    broadcast_sos_video_chunk,
    broadcast_incident_state_changed
)

sos_bp = Blueprint("sos_bp", __name__)

@sos_bp.route("/start", methods=["POST"])
def start_sos():
    """
    Initiates emergency protocol.
    Creates SOS_EVENTS record (Risk Score = 100, CRITICAL), emits high-priority event.
    """
    data = request.get_json() or {}
    visitor_id = data.get("visitor_id")
    lat = float(data.get("latitude", 0.0))
    lng = float(data.get("longitude", 0.0))
    zone_code = data.get("zone", "UNKNOWN")

    if not visitor_id:
        return jsonify({"error": "visitor_id is required"}), 400

    db = SessionLocal()
    try:
        visitor = db.query(Visitor).filter(Visitor.visitor_id == visitor_id).first()
        if not visitor:
            return jsonify({"error": "Visitor not found"}), 404

        sos_id = f"SOS-{uuid.uuid4().hex[:8].upper()}"

        sos = SosEvent(
            sos_id=sos_id,
            visitor_id=visitor_id,
            triggered_at=datetime.utcnow(),
            initial_lat=lat or visitor.last_latitude,
            initial_lng=lng or visitor.last_longitude,
            initial_zone=zone_code or visitor.current_zone,
            status="TRIGGERED"
        )
        db.add(sos)

        # Escalate visitor risk state
        visitor.current_risk_level = "CRITICAL"
        visitor.status = "ACTIVE"
        if lat and lng:
            visitor.last_latitude = lat
            visitor.last_longitude = lng

        # Add Risk Event record
        risk_event = RiskEvent(
            visitor_id=visitor_id,
            score=100,
            severity="CRITICAL",
            reasons_json='["SOS_EMERGENCY_TRIGGERED (+100)"]',
            timestamp=datetime.utcnow()
        )
        db.add(risk_event)

        # Create System Alert
        alert = Alert(
            alert_type="SOS_TRIGGERED",
            severity="CRITICAL",
            visitor_id=visitor_id,
            title=f"MAYDAY: Emergency SOS from {visitor.full_name}",
            description=f"Critical emergency beacon triggered by {visitor.full_name} ({visitor_id}) in {visitor.current_zone}. Phone: {visitor.phone}, Emergency Contact: {visitor.emergency_contact_name} ({visitor.emergency_contact_phone}).",
            status="UNREAD"
        )
        db.add(alert)
        db.commit()

        sos_payload = sos.to_dict()
        sos_payload["visitor"] = visitor.to_dict()
        sos_payload["alert"] = alert.to_dict()

        # Broadcast emergency to Admin and Rescuers
        broadcast_emergency_alert(sos_payload)

        return jsonify({
            "success": True,
            "sos_id": sos_id,
            "message": "Emergency protocol activated. Distress beacon dispatched to SAR Command.",
            "sos_event": sos_payload
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Failed to activate SOS: {str(e)}"}), 500
    finally:
        db.close()

@sos_bp.route("/video/chunk", methods=["POST"])
def upload_video_chunk():
    """
    Multipart upload endpoint for 30s/60s chunked SOS video stream.
    Validates SHA-256 checksum and saves chunk to sandboxed storage.
    """
    sos_id = request.form.get("sos_id")
    seq_str = request.form.get("sequence_number", "0")
    provided_sha256 = request.form.get("file_hash_sha256")
    duration = int(request.form.get("duration_seconds", "30"))

    if not sos_id:
        return jsonify({"error": "sos_id is required"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No video file stream provided"}), 400

    video_file = request.files["file"]
    file_bytes = video_file.read()

    db = SessionLocal()
    try:
        success, msg, chunk = VideoHandler.save_chunk(
            db=db,
            sos_id=sos_id,
            sequence_number=int(seq_str),
            file_bytes=file_bytes,
            provided_sha256=provided_sha256,
            duration_seconds=duration
        )

        if not success:
            return jsonify({"error": msg}), 400

        chunk_dict = chunk.to_dict()
        broadcast_sos_video_chunk(chunk_dict)

        return jsonify({
            "success": True,
            "message": msg,
            "chunk": chunk_dict
        }), 200

    finally:
        db.close()

@sos_bp.route("/resolve", methods=["POST"])
def resolve_sos():
    """
    Resolves active SOS incident with mandatory action report.
    Can be invoked by Officer or Admin.
    """
    data = request.get_json() or {}
    sos_id = data.get("sos_id")
    resolution_notes = data.get("resolution_notes", "Resolved by field SAR unit.")
    officer_id = data.get("officer_id")

    if not sos_id:
        return jsonify({"error": "sos_id is required"}), 400

    db = SessionLocal()
    try:
        sos = db.query(SosEvent).filter(SosEvent.sos_id == sos_id).first()
        if not sos:
            return jsonify({"error": "SOS incident not found"}), 404

        sos.status = "RESOLVED"
        sos.resolved_at = datetime.utcnow()
        sos.resolution_notes = resolution_notes
        if officer_id:
            sos.acknowledged_by_officer_id = officer_id

        # Normalize visitor risk
        visitor = db.query(Visitor).filter(Visitor.visitor_id == sos.visitor_id).first()
        if visitor:
            visitor.current_risk_level = "LOW"
            visitor.status = "ACTIVE"

        # Resolve related alerts
        alerts = db.query(Alert).filter(Alert.visitor_id == sos.visitor_id, Alert.status != "RESOLVED").all()
        for a in alerts:
            a.status = "RESOLVED"
            a.resolved_at = datetime.utcnow()

        db.commit()

        resolved_payload = sos.to_dict()
        broadcast_incident_state_changed(resolved_payload)

        return jsonify({
            "success": True,
            "message": "Incident marked as RESOLVED",
            "sos_event": resolved_payload
        }), 200

    finally:
        db.close()

@sos_bp.route("/active", methods=["GET"])
def get_active_sos():
    db = SessionLocal()
    try:
        active_list = db.query(SosEvent).filter(
            SosEvent.status.in_(["TRIGGERED", "ACKNOWLEDGED", "DISPATCHED"])
        ).order_by(SosEvent.triggered_at.desc()).all()

        results = []
        for s in active_list:
            d = s.to_dict()
            if s.visitor:
                d["visitor"] = s.visitor.to_dict()
            results.append(d)

        return jsonify(results), 200
    finally:
        db.close()

@sos_bp.route("/<sos_id>/videos", methods=["GET"])
def get_sos_videos(sos_id):
    db = SessionLocal()
    try:
        chunks = db.query(SosVideoChunk).filter(
            SosVideoChunk.sos_id == sos_id
        ).order_by(SosVideoChunk.sequence_number.asc()).all()

        return jsonify([c.to_dict() for c in chunks]), 200
    finally:
        db.close()

@sos_bp.route("/video/stream/<chunk_id>", methods=["GET"])
def stream_video_chunk(chunk_id):
    db = SessionLocal()
    try:
        chunk = db.query(SosVideoChunk).filter(SosVideoChunk.id == chunk_id).first()
        if not chunk or not os.path.exists(chunk.storage_path):
            return jsonify({"error": "Video chunk not found"}), 404

        mime = "video/webm" if chunk.storage_path.endswith(".webm") else "video/mp4"
        return send_file(chunk.storage_path, mimetype=mime)
    finally:
        db.close()
