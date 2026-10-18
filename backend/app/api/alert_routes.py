from datetime import datetime
from flask import Blueprint, request, jsonify
from backend.app.database.connection import SessionLocal
from backend.app.database.models import Alert

alert_bp = Blueprint("alert_bp", __name__)

@alert_bp.route("", methods=["GET"])
def get_alerts():
    status = request.args.get("status")
    limit = int(request.args.get("limit", 50))
    db = SessionLocal()
    try:
        query = db.query(Alert)
        if status:
            query = query.filter(Alert.status == status.upper())
        alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
        return jsonify([a.to_dict() for a in alerts]), 200
    finally:
        db.close()

@alert_bp.route("/<int:alert_id>/ack", methods=["POST"])
def acknowledge_alert(alert_id):
    db = SessionLocal()
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            return jsonify({"error": "Alert not found"}), 404
        alert.status = "ACKNOWLEDGED"
        db.commit()
        return jsonify({"success": True, "alert": alert.to_dict()}), 200
    finally:
        db.close()

@alert_bp.route("/<int:alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    db = SessionLocal()
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            return jsonify({"error": "Alert not found"}), 404
        alert.status = "RESOLVED"
        alert.resolved_at = datetime.utcnow()
        db.commit()
        return jsonify({"success": True, "alert": alert.to_dict()}), 200
    finally:
        db.close()
