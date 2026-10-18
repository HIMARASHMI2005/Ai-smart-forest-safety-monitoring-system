import pytest
from shapely.geometry import Polygon
from backend.app.geofence.geometry_engine import GeometryEngine

def test_bbox_prefilter():
    # Min lat 10, max lat 20, min lng 30, max lng 40
    assert GeometryEngine.bbox_prefilter(15.0, 35.0, 10.0, 20.0, 30.0, 40.0) is True
    # Out of latitude
    assert GeometryEngine.bbox_prefilter(25.0, 35.0, 10.0, 20.0, 30.0, 40.0) is False
    # Out of longitude
    assert GeometryEngine.bbox_prefilter(15.0, 45.0, 10.0, 20.0, 30.0, 40.0) is False

def test_contains_point_ray_casting():
    # Square polygon from (10, 10) to (20, 20) -> [lng, lat]
    poly_coords = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)]
    polygon = Polygon(poly_coords)

    # Point inside (lat=15, lng=15)
    assert GeometryEngine.contains_point(polygon, lat=15.0, lng=15.0) is True

    # Point outside (lat=25, lng=25)
    assert GeometryEngine.contains_point(polygon, lat=25.0, lng=25.0) is False

def test_transition_detection():
    # Entry
    trans1 = GeometryEngine.detect_transition("OUT_OF_BOUNDS", "ZONE_A")
    assert trans1 is not None
    assert trans1["event_type"] == "ENTRY"

    # Transition
    trans2 = GeometryEngine.detect_transition("ZONE_A", "ZONE_C")
    assert trans2 is not None
    assert trans2["event_type"] == "TRANSITION"

    # Exit
    trans3 = GeometryEngine.detect_transition("ZONE_C", "OUT_OF_BOUNDS")
    assert trans3 is not None
    assert trans3["event_type"] == "EXIT"

    # Same zone -> None
    trans4 = GeometryEngine.detect_transition("ZONE_A", "ZONE_A")
    assert trans4 is None
