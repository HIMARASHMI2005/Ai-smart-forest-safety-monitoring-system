-- Seed Geofence Zones for Smart Forest Operational Reserve (Bandipur-Mudumalai Wildland Sector)
-- Coordinate Datum: WGS 84 (EPSG:4326)

INSERT OR REPLACE INTO zones (id, name, code, severity, polygon_geojson, min_lat, max_lat, min_lng, max_lng, active, color_hex)
VALUES 
(
    'ZONE_A',
    'Eco Trail Corridor',
    'ZONE_A',
    'LOW',
    '{"type":"Polygon","coordinates":[[[76.6150,11.6550],[76.6350,11.6550],[76.6350,11.6700],[76.6150,11.6700],[76.6150,11.6550]]]}',
    11.6550,
    11.6700,
    76.6150,
    76.6350,
    1,
    '#2D6A4F'
),
(
    'ZONE_B',
    'Monitored Forest Buffer',
    'ZONE_B',
    'MEDIUM',
    '{"type":"Polygon","coordinates":[[[76.6050,11.6500],[76.6450,11.6500],[76.6450,11.6750],[76.6050,11.6750],[76.6050,11.6500]]]}',
    11.6500,
    11.6750,
    76.6050,
    76.6450,
    1,
    '#E09F3E'
),
(
    'ZONE_C',
    'Restricted Ridge & Gorge Corridor',
    'ZONE_C',
    'HIGH',
    '{"type":"Polygon","coordinates":[[[76.6300,11.6700],[76.6550,11.6700],[76.6550,11.6850],[76.6300,11.6850],[76.6300,11.6700]]]}',
    11.6700,
    11.6850,
    76.6300,
    76.6550,
    1,
    '#C1121F'
),
(
    'ZONE_D',
    'Core Wildlife Sanctuary',
    'ZONE_D',
    'CRITICAL',
    '{"type":"Polygon","coordinates":[[[76.6000,11.6800],[76.6300,11.6800],[76.6300,11.6950],[76.6000,11.6950],[76.6000,11.6800]]]}',
    11.6800,
    11.6950,
    76.6000,
    76.6300,
    1,
    '#780000'
);
