from datetime import datetime
from flask import Blueprint, request, jsonify
from backend.app.database.connection import SessionLocal
from backend.app.database.models import Visitor, Location, ZoneEvent, RiskEvent, Alert
from backend.app.geofence.zone_manager import ZoneManager
from backend.app.geofence.geometry_engine import GeometryEngine
from backend.app.risk_engine.evaluator import RiskEvaluator
from backend.app.websocket.events import (
    broadcast_visitor_location,
    broadcast_zone_transition,
    broadcast_emergency_alert
)

location_bp = Blueprint("location_bp", __name__)

@location_bp.route("/update", methods=["POST"])
def update_location():
    """
    Ingests continuous telemetry payload:
    { visitor_id, latitude, longitude, accuracy, speed, bearing, timestamp }
    Executes Shapely ray-casting containment; calculates risk delta; broadcasts changes via Socket.IO.
    """
    data = request.get_json() or {}
    visitor_id = data.get("visitor_id")
    try:
        lat = float(data.get("latitude"))
        lng = float(data.get("longitude"))
    except (TypeError, ValueError):
        return jsonify({"error": "Valid latitude and longitude are required"}), 400

    accuracy = float(data.get("accuracy", 5.0))
    speed = float(data.get("speed", 0.0))
    bearing = float(data.get("bearing", 0.0))
    altitude = float(data.get("altitude", 0.0))

    db = SessionLocal()
    try:
        visitor = db.query(Visitor).filter(Visitor.visitor_id == visitor_id).first()
        if not visitor:
            return jsonify({"error": f"Visitor {visitor_id} not found"}), 404

        previous_zone = visitor.current_zone

        # 1. Deterministic Geofencing via Shapely Ray Casting
        zone_id, zone_severity, zone_obj = ZoneManager.find_containing_zone(lat, lng, db)

        # 2. Check for Topological Transition: Delta Z = (Z_{t-1}, Z_t)
        transition = GeometryEngine.detect_transition(previous_zone, zone_id)
        zone_event_record = None

        if transition:
            zone_event_record = ZoneEvent(
                visitor_id=visitor_id,
                zone_id=zone_id if zone_id != "OUT_OF_BOUNDS" else (previous_zone or "ZONE_B"),
                previous_zone=previous_zone,
                current_zone=zone_id,
                event_type=transition["event_type"],
                severity=zone_severity,
                timestamp=datetime.utcnow()
            )
            db.add(zone_event_record)

            # Trigger critical alert if entered Restricted or Critical wildlife area
            if zone_id in ["ZONE_C", "ZONE_D"]:
                alert = Alert(
                    alert_type="CRITICAL_ZONE" if zone_id == "ZONE_D" else "ZONE_VIOLATION",
                    severity="CRITICAL" if zone_id == "ZONE_D" else "DANGER",
                    visitor_id=visitor_id,
                    title=f"Restricted Zone Breach: {zone_id}",
                    description=f"Visitor {visitor.full_name} ({visitor_id}) crossed into {zone_id} ({zone_severity}) at [{lat:.5f}, {lng:.5f}].",
                    status="UNREAD"
                )
                db.add(alert)
                db.flush()
                broadcast_emergency_alert(alert.to_dict())

            # Broadcast zone transition event
            broadcast_zone_transition({
                "visitor_id": visitor_id,
                "full_name": visitor.full_name,
                "previous_zone": previous_zone,
                "current_zone": zone_id,
                "event_type": transition["event_type"],
                "severity": zone_severity,
                "latitude": lat,
                "longitude": lng,
                "timestamp": datetime.utcnow().isoformat()
            })

        # 3. Deterministic Risk Assessment Calculation
        is_sos_active = (visitor.current_risk_level == "CRITICAL" and visitor.status == "ACTIVE")
        risk_result = RiskEvaluator.evaluate(
            zone_code=zone_id,
            is_sos=is_sos_active,
            timestamp=datetime.utcnow()
        )

        risk_score = risk_result["score"]
        risk_level = risk_result["severity"]
        reasons_list = risk_result["reasons"]

        # Commit Risk Event
        risk_event = RiskEvent(
            visitor_id=visitor_id,
            score=risk_score,
            severity=risk_level,
            reasons_json=str(reasons_list),
            timestamp=datetime.utcnow()
        )
        db.add(risk_event)

        # 4. Save Location Breadcrumb
        loc = Location(
            visitor_id=visitor_id,
            latitude=lat,
            longitude=lng,
            altitude=altitude,
            accuracy=accuracy,
            speed=speed,
            bearing=bearing,
            timestamp=datetime.utcnow()
        )
        db.add(loc)

        # 5. Update Visitor Profile
        visitor.last_latitude = lat
        visitor.last_longitude = lng
        visitor.current_zone = zone_id
        visitor.current_risk_level = risk_level
        visitor.last_seen = datetime.utcnow()
        visitor.status = "ACTIVE"

        db.commit()

        # 6. Broadcast Real-Time Location Update via Socket.IO
        telemetry_payload = {
            "visitor_id": visitor_id,
            "full_name": visitor.full_name,
            "latitude": lat,
            "longitude": lng,
            "accuracy": accuracy,
            "speed": speed,
            "bearing": bearing,
            "current_zone": zone_id,
            "current_risk_level": risk_level,
            "risk_score": risk_score,
            "risk_reasons": reasons_list,
            "last_seen": visitor.last_seen.isoformat()
        }
        broadcast_visitor_location(telemetry_payload)

        return jsonify({
            "success": True,
            "visitor_id": visitor_id,
            "current_zone": zone_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_reasons": reasons_list,
            "accuracy_warning": accuracy > 35.0,
            "transition": transition is not None
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Failed to ingest location: {str(e)}"}), 500
    finally:
        db.close()

@location_bp.route("/visitor/<visitor_id>/history", methods=["GET"])
def get_visitor_history(visitor_id):
    limit = int(request.args.get("limit", 100))
    db = SessionLocal()
    try:
        records = db.query(Location).filter(
            Location.visitor_id == visitor_id
        ).order_by(Location.timestamp.desc()).limit(limit).all()

        return jsonify([r.to_dict() for r in reversed(records)]), 200
    finally:
        db.close()

@location_bp.route("/live-all", methods=["GET"])
def get_live_all():
    db = SessionLocal()
    try:
        visitors = db.query(Visitor).all()
        return jsonify([v.to_dict() for v in visitors]), 200
    finally:
        db.close()
