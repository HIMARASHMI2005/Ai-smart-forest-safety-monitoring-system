import json
from functools import wraps
from flask import Blueprint, request, jsonify, g
from backend.app.database.connection import SessionLocal
from backend.app.database.models import Zone
from backend.app.geofence.geometry_engine import GeometryEngine
from backend.app.geofence.zone_manager import ZoneManager
from backend.app.api.auth_middleware import role_required

zone_bp = Blueprint("zone_bp", __name__)

def admin_auth_optional(f):
    """Permits local admin portal operations while validating JWT if header present."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if auth_header:
            return role_required(["ADMIN"])(f)(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated

def make_rect_geojson(min_lng, min_lat, max_lng, max_lat):
    return {
        "type": "Polygon",
        "coordinates": [[
            [round(float(min_lng), 6), round(float(min_lat), 6)],
            [round(float(max_lng), 6), round(float(min_lat), 6)],
            [round(float(max_lng), 6), round(float(max_lat), 6)],
            [round(float(min_lng), 6), round(float(max_lat), 6)],
            [round(float(min_lng), 6), round(float(min_lat), 6)]
        ]]
    }

@zone_bp.route("", methods=["GET"])
def list_zones():
    db = SessionLocal()
    try:
        zones = db.query(Zone).all()
        return jsonify([z.to_dict() for z in zones]), 200
    finally:
        db.close()

@zone_bp.route("/<zone_id>", methods=["GET"])
def get_zone(zone_id):
    db = SessionLocal()
    try:
        zone = db.query(Zone).filter(Zone.id == zone_id).first()
        if not zone:
            return jsonify({"error": "Zone not found"}), 404
        return jsonify(zone.to_dict()), 200
    finally:
        db.close()

@zone_bp.route("", methods=["POST"])
@admin_auth_optional
def create_zone():
    data = request.get_json() or {}
    zone_id = (data.get("id") or data.get("code") or "").strip().upper()
    name = (data.get("name") or "").strip()
    code = (data.get("code") or zone_id).strip().upper()
    severity = (data.get("severity") or "LOW").strip().upper()
    color_hex = data.get("color_hex", "#2D6A4F")
    active = bool(data.get("active", True))

    if not zone_id or not name:
        return jsonify({"error": "Zone ID and Name are required"}), 400

    polygon_geojson = data.get("polygon_geojson")

    # If coordinates provided directly without polygon GeoJSON:
    if not polygon_geojson and all(k in data for k in ["min_lat", "max_lat", "min_lng", "max_lng"]):
        polygon_geojson = make_rect_geojson(
            data["min_lng"], data["min_lat"], data["max_lng"], data["max_lat"]
        )

    if not polygon_geojson:
        return jsonify({"error": "Either polygon_geojson or boundary coordinates (min/max lat/lng) are required"}), 400

    poly = GeometryEngine.parse_geojson_polygon(polygon_geojson)
    if poly is None or poly.is_empty:
        return jsonify({"error": "Invalid polygon GeoJSON geometry or closure"}), 400

    min_lng, min_lat, max_lng, max_lat = poly.bounds

    db = SessionLocal()
    try:
        existing = db.query(Zone).filter(Zone.id == zone_id).first()
        if existing:
            return jsonify({"error": f"Zone {zone_id} already exists"}), 409

        geojson_str = json.dumps(polygon_geojson) if isinstance(polygon_geojson, dict) else str(polygon_geojson)
        zone = Zone(
            id=zone_id,
            name=name,
            code=code,
            severity=severity,
            polygon_geojson=geojson_str,
            min_lat=float(min_lat),
            max_lat=float(max_lat),
            min_lng=float(min_lng),
            max_lng=float(max_lng),
            active=active,
            color_hex=color_hex
        )
        db.add(zone)
        db.commit()

        ZoneManager.invalidate_cache()
        return jsonify(zone.to_dict()), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Failed to create zone: {str(e)}"}), 500
    finally:
        db.close()

@zone_bp.route("/<zone_id>", methods=["PUT"])
@admin_auth_optional
def update_zone(zone_id):
    data = request.get_json() or {}
    db = SessionLocal()
    try:
        zone = db.query(Zone).filter(Zone.id == zone_id).first()
        if not zone:
            return jsonify({"error": "Zone not found"}), 404

        if "name" in data and data["name"]:
            zone.name = str(data["name"]).strip()
        if "severity" in data and data["severity"]:
            zone.severity = str(data["severity"]).strip().upper()
        if "code" in data and data["code"]:
            zone.code = str(data["code"]).strip().upper()
        if "color_hex" in data and data["color_hex"]:
            zone.color_hex = str(data["color_hex"]).strip()
        if "active" in data:
            zone.active = bool(data["active"])

        # Update geometry via Polygon GeoJSON or Coordinate bounds
        if "polygon_geojson" in data and data["polygon_geojson"]:
            geojson = data["polygon_geojson"]
            poly = GeometryEngine.parse_geojson_polygon(geojson)
            if poly is None or poly.is_empty:
                return jsonify({"error": "Invalid GeoJSON geometry"}), 400
            min_lng, min_lat, max_lng, max_lat = poly.bounds
            zone.polygon_geojson = json.dumps(geojson) if isinstance(geojson, dict) else str(geojson)
            zone.min_lat = float(min_lat)
            zone.max_lat = float(max_lat)
            zone.min_lng = float(min_lng)
            zone.max_lng = float(max_lng)
        elif all(k in data for k in ["min_lat", "max_lat", "min_lng", "max_lng"]):
            min_lat = float(data["min_lat"])
            max_lat = float(data["max_lat"])
            min_lng = float(data["min_lng"])
            max_lng = float(data["max_lng"])
            zone.min_lat = min_lat
            zone.max_lat = max_lat
            zone.min_lng = min_lng
            zone.max_lng = max_lng
            rect_poly = make_rect_geojson(min_lng, min_lat, max_lng, max_lat)
            zone.polygon_geojson = json.dumps(rect_poly)

        db.commit()
        ZoneManager.invalidate_cache()
        return jsonify(zone.to_dict()), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Failed to update zone: {str(e)}"}), 500
    finally:
        db.close()

@zone_bp.route("/<zone_id>", methods=["DELETE"])
@admin_auth_optional
def delete_zone(zone_id):
    db = SessionLocal()
    try:
        zone = db.query(Zone).filter(Zone.id == zone_id).first()
        if not zone:
            return jsonify({"error": "Zone not found"}), 404

        db.delete(zone)
        db.commit()
        ZoneManager.invalidate_cache()
        return jsonify({"message": f"Zone {zone_id} deleted successfully"}), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": f"Failed to delete zone: {str(e)}"}), 500
    finally:
        db.close()
