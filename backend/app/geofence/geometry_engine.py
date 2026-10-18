import json
from typing import Tuple, Optional, Dict, Any
from shapely.geometry import shape, Point, Polygon

class GeometryEngine:
    """
    Deterministic Computational Geometry Engine.
    Implements:
    - Bounding Box Pre-filter
    - Exact Point-in-Polygon (Ray casting algorithm via Shapely / GEOS)
    - Boundary distance calculus
    - Transition state tracking
    """

    @staticmethod
    def parse_geojson_polygon(geojson_str_or_dict) -> Optional[Polygon]:
        """Parses GeoJSON geometry into a Shapely Polygon."""
        try:
            if isinstance(geojson_str_or_dict, str):
                data = json.loads(geojson_str_or_dict)
            else:
                data = geojson_str_or_dict
            poly = shape(data)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Clean invalid geometries
            return poly
        except Exception as e:
            print(f"Error parsing polygon geometry: {e}")
            return None

    @staticmethod
    def bbox_prefilter(lat: float, lng: float, min_lat: float, max_lat: float, min_lng: float, max_lng: float) -> bool:
        """
        Fast Bounding Box Pre-Filter:
        Quick test if point lies within [min_lat, max_lat] x [min_lng, max_lng].
        Avoids expensive ray casting operations when candidate is out of spatial envelope.
        """
        return (min_lat <= lat <= max_lat) and (min_lng <= lng <= max_lng)

    @staticmethod
    def contains_point(polygon: Polygon, lat: float, lng: float) -> bool:
        """
        Executes exact Point-in-Polygon containment check via Ray Casting.
        GeoJSON coordinate order is [longitude, latitude] (x, y).
        """
        if polygon is None:
            return False
        # Point(x, y) where x=lng, y=lat
        point = Point(lng, lat)
        return polygon.contains(point) or polygon.touches(point)

    @staticmethod
    def distance_to_boundary(polygon: Polygon, lat: float, lng: float) -> float:
        """
        Calculates planar distance from point to polygon boundary in approximate meters.
        1 degree latitude ~= 111,320m.
        """
        if polygon is None:
            return float("inf")
        point = Point(lng, lat)
        # Degree distance
        deg_dist = polygon.exterior.distance(point)
        # Rough meters conversion for localized coordinates
        return deg_dist * 111320.0

    @staticmethod
    def detect_transition(previous_zone: Optional[str], current_zone: str) -> Optional[Dict[str, str]]:
        """
        Topological transition detector:
        Delta Z = (Z_{t-1}, Z_t)
        Returns event_type ('ENTRY', 'EXIT', 'TRANSITION') if zone changed, else None.
        """
        if previous_zone == current_zone:
            return None

        if previous_zone is None or previous_zone == "OUT_OF_BOUNDS":
            return {
                "event_type": "ENTRY",
                "previous_zone": previous_zone or "OUT_OF_BOUNDS",
                "current_zone": current_zone
            }
        elif current_zone == "OUT_OF_BOUNDS":
            return {
                "event_type": "EXIT",
                "previous_zone": previous_zone,
                "current_zone": "OUT_OF_BOUNDS"
            }
        else:
            return {
                "event_type": "TRANSITION",
                "previous_zone": previous_zone,
                "current_zone": current_zone
            }
