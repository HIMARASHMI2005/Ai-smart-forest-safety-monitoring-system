from datetime import datetime
from flask import Blueprint, request, jsonify
from backend.app.database.connection import SessionLocal
from backend.app.database.models import Officer, SosEvent, User
from backend.app.websocket.events import (
    broadcast_officer_location,
    broadcast_incident_state_changed
)
from backend.app.api.auth_middleware import token_required, role_required

officer_bp = Blueprint("officer_bp", __name__)

@officer_bp.route("/update-location", methods=["POST"])
def update_officer_location():
    data = request.get_json() or {}
    officer_id = data.get("officer_id")
    try:
        lat = float(data.get("latitude"))
        lng = float(data.get("longitude"))
    except (TypeError, ValueError):
        return jsonify({"error": "Valid latitude and longitude are required"}), 400

    db = SessionLocal()
    try:
        officer = db.query(Officer).filter(Officer.officer_id == officer_id).first()
        if not officer:
            return jsonify({"error": "Officer not found"}), 404

        officer.current_latitude = lat
        officer.current_longitude = lng
        officer.last_ping = datetime.utcnow()
        db.commit()

        payload = officer.to_dict()
        broadcast_officer_location(payload)

        return jsonify({"success": True, "officer": payload}), 200
    finally:
        db.close()

@officer_bp.route("/list", methods=["GET"])
def list_officers():
    status = request.args.get("status")
    db = SessionLocal()
    try:
        query = db.query(Officer)
        if status:
            query = query.filter(Officer.approval_status == status.upper())
        officers = query.all()
        return jsonify([o.to_dict() for o in officers]), 200
    finally:
        db.close()

@officer_bp.route("/pending", methods=["GET"])
def list_pending_officers():
    db = SessionLocal()
    try:
        officers = db.query(Officer).filter(Officer.approval_status == "PENDING").all()
        return jsonify([o.to_dict() for o in officers]), 200
    finally:
        db.close()

@officer_bp.route("/acknowledge", methods=["POST"])
def acknowledge_incident():
    data = request.get_json() or {}
    sos_id = data.get("sos_id")
    officer_id = data.get("officer_id")

    if not sos_id or not officer_id:
        return jsonify({"error": "sos_id and officer_id are required"}), 400

    db = SessionLocal()
    try:
        sos = db.query(SosEvent).filter(SosEvent.sos_id == sos_id).first()
        if not sos:
            return jsonify({"error": "SOS incident not found"}), 404

        officer = db.query(Officer).filter(Officer.officer_id == officer_id).first()
        if not officer:
            return jsonify({"error": "Officer not found"}), 404

        sos.status = "ACKNOWLEDGED"
        sos.acknowledged_by_officer_id = officer_id
        officer.availability_status = "DISPATCHED"
        db.commit()

        incident_data = sos.to_dict()
        incident_data["officer_badge"] = officer.badge_number
        incident_data["officer_name"] = officer.full_name
        broadcast_incident_state_changed(incident_data)

        return jsonify({
            "success": True,
            "message": f"Incident acknowledged by Officer {officer.full_name}",
            "incident": incident_data
        }), 200
    finally:
        db.close()

@officer_bp.route("/availability", methods=["POST"])
def update_availability():
    data = request.get_json() or {}
    officer_id = data.get("officer_id")
    status = data.get("status", "AVAILABLE").upper()

    if status not in ["AVAILABLE", "DISPATCHED", "OFF_DUTY"]:
        return jsonify({"error": "Invalid status"}), 400

    db = SessionLocal()
    try:
        officer = db.query(Officer).filter(Officer.officer_id == officer_id).first()
        if not officer:
            return jsonify({"error": "Officer not found"}), 404

        officer.availability_status = status
        db.commit()
        return jsonify({"success": True, "officer": officer.to_dict()}), 200
    finally:
        db.close()

@officer_bp.route("/approve/<officer_id>", methods=["POST"])
def approve_officer(officer_id):
    data = request.get_json() or {}
    action = data.get("action", "APPROVE").upper()

    db = SessionLocal()
    try:
        officer = db.query(Officer).filter(Officer.officer_id == officer_id).first()
        if not officer:
            return jsonify({"error": "Officer not found"}), 404

        officer.approval_status = "APPROVED" if action == "APPROVE" else "REJECTED"
        if officer.user:
            officer.user.status = "ACTIVE" if action == "APPROVE" else "SUSPENDED"

        db.commit()
        return jsonify({"success": True, "officer": officer.to_dict()}), 200
    finally:
        db.close()
