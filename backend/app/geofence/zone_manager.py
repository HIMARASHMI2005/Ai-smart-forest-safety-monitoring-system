from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from backend.app.database.models import Zone, ZoneEvent
from backend.app.geofence.geometry_engine import GeometryEngine

class ZoneManager:
    """
    Manages geofence definitions, spatial caching, and containment queries.
    """
    _zone_cache = {}
    _cache_valid = False

    @classmethod
    def invalidate_cache(cls):
        cls._cache_valid = False
        cls._zone_cache.clear()

    @classmethod
    def load_zones(cls, db: Session) -> Dict[str, Any]:
        """Loads and parses all active geofence zones into memory."""
        if cls._cache_valid and cls._zone_cache:
            return cls._zone_cache

        zones = db.query(Zone).filter(Zone.active == True).all()
        cache = {}
        for z in zones:
            poly = GeometryEngine.parse_geojson_polygon(z.polygon_geojson)
            if poly:
                cache[z.id] = {
                    "zone": z,
                    "polygon": poly,
                    "min_lat": z.min_lat,
                    "max_lat": z.max_lat,
                    "min_lng": z.min_lng,
                    "max_lng": z.max_lng,
                    "severity": z.severity,
                    "code": z.code
                }
        cls._zone_cache = cache
        cls._cache_valid = True
        return cls._zone_cache

    @classmethod
    def find_containing_zone(cls, lat: float, lng: float, db: Session) -> Tuple[str, str, Optional[Zone]]:
        """
        Determines which geofence zone contains point (lat, lng).
        Returns (zone_id, severity, Zone_object).
        If point is not in any zone, returns ("OUT_OF_BOUNDS", "LOW", None).
        
        Priority resolution when multiple zones match:
        - Critical / Restricted zones take priority for safety:
          ZONE_D (Critical) > ZONE_C (High) > ZONE_A (Designated Trail) > ZONE_B (Buffer)
        """
        zones_data = cls.load_zones(db)
        
        matched_candidates = []
        for zone_id, data in zones_data.items():
            # 1. Bounding box pre-filter
            if not GeometryEngine.bbox_prefilter(
                lat, lng, data["min_lat"], data["max_lat"], data["min_lng"], data["max_lng"]
            ):
                continue

            # 2. Exact ray-casting containment
            if GeometryEngine.contains_point(data["polygon"], lat, lng):
                matched_candidates.append(data)

        if not matched_candidates:
            return "OUT_OF_BOUNDS", "LOW", None

        # Priority map:
        priority_order = {
            "ZONE_D": 4, # Critical Wildlife Area
            "ZONE_C": 3, # Restricted Corridor
            "ZONE_A": 2, # Eco Trail
            "ZONE_B": 1  # Monitored Buffer
        }

        # Sort candidate zones by priority descending
        matched_candidates.sort(
            key=lambda item: priority_order.get(item["code"], 0),
            reverse=True
        )

        selected = matched_candidates[0]
        return selected["zone"].id, selected["severity"], selected["zone"]
