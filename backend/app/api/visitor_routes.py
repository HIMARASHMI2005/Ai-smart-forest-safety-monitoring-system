from datetime import datetime
from flask import Blueprint, request, jsonify
from backend.app.database.connection import SessionLocal
from backend.app.database.models import Visitor, SosEvent, RiskEvent, ZoneEvent

visitor_bp = Blueprint("visitor_bp", __name__)

@visitor_bp.route("/list", methods=["GET"])
def list_visitors():
    db = SessionLocal()
    try:
        visitors = db.query(Visitor).all()
        return jsonify([v.to_dict() for v in visitors]), 200
    finally:
        db.close()

@visitor_bp.route("/<visitor_id>/profile", methods=["GET"])
def get_visitor_profile(visitor_id):
    db = SessionLocal()
    try:
        visitor = db.query(Visitor).filter(Visitor.visitor_id == visitor_id).first()
        if not visitor:
            return jsonify({"error": "Visitor not found"}), 404

        latest_risk = db.query(RiskEvent).filter(
            RiskEvent.visitor_id == visitor_id
        ).order_by(RiskEvent.timestamp.desc()).first()

        data = visitor.to_dict()
        data["latest_risk_score"] = latest_risk.score if latest_risk else 0
        data["risk_reasons"] = latest_risk.reasons_json if latest_risk else "[]"
        return jsonify(data), 200
    finally:
        db.close()

@visitor_bp.route("/<visitor_id>/check-in", methods=["POST"])
def check_in_visitor(visitor_id):
    db = SessionLocal()
    try:
        visitor = db.query(Visitor).filter(Visitor.visitor_id == visitor_id).first()
        if not visitor:
            return jsonify({"error": "Visitor not found"}), 404

        visitor.status = "CHECKED_IN"
        visitor.last_seen = datetime.utcnow()
        db.commit()
        return jsonify({"success": True, "visitor": visitor.to_dict()}), 200
    finally:
        db.close()

@visitor_bp.route("/<visitor_id>/sos-history", methods=["GET"])
def get_visitor_sos_history(visitor_id):
    db = SessionLocal()
    try:
        events = db.query(SosEvent).filter(
            SosEvent.visitor_id == visitor_id
        ).order_by(SosEvent.triggered_at.desc()).all()
        return jsonify([e.to_dict() for e in events]), 200
    finally:
        db.close()
