-- ====================================================================
-- SMART FOREST GEO-FENCING & EMERGENCY MONITORING SYSTEM
-- Relational Database Schema Specification
-- Target: SQLite 3 / PostgreSQL 14+ / MySQL 8+
-- ====================================================================

-- 1. USERS TABLE (Core Authentication & Role Management)
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK(role IN ('ADMIN', 'OFFICER', 'VISITOR')),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'PENDING', 'SUSPENDED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. VISITORS PROFILE
CREATE TABLE IF NOT EXISTS visitors (
    visitor_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    full_name VARCHAR(120) NOT NULL,
    phone VARCHAR(30) NOT NULL,
    emergency_contact_name VARCHAR(120) NOT NULL,
    emergency_contact_phone VARCHAR(30) NOT NULL,
    current_zone VARCHAR(30) DEFAULT 'OUT_OF_BOUNDS',
    current_risk_level VARCHAR(20) DEFAULT 'LOW' CHECK(current_risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    last_latitude REAL DEFAULT 0.0,
    last_longitude REAL DEFAULT 0.0,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'REGISTERED' CHECK(status IN ('REGISTERED', 'CHECKED_IN', 'ACTIVE', 'EXITED')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 3. OFFICERS PROFILE
CREATE TABLE IF NOT EXISTS officers (
    officer_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    badge_number VARCHAR(50) UNIQUE NOT NULL,
    full_name VARCHAR(120) NOT NULL,
    phone VARCHAR(30) NOT NULL,
    department VARCHAR(100) NOT NULL,
    approval_status VARCHAR(20) DEFAULT 'PENDING' CHECK(approval_status IN ('PENDING', 'APPROVED', 'REJECTED')),
    availability_status VARCHAR(20) DEFAULT 'AVAILABLE' CHECK(availability_status IN ('AVAILABLE', 'DISPATCHED', 'OFF_DUTY')),
    current_latitude REAL DEFAULT 0.0,
    current_longitude REAL DEFAULT 0.0,
    last_ping TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 4. BIOMETRIC TEMPLATES (ForestFaceNet 128-D Embedding Templates)
CREATE TABLE IF NOT EXISTS biometric_templates (
    id VARCHAR(36) PRIMARY KEY,
    visitor_id VARCHAR(36) NOT NULL,
    embedding_vector BLOB NOT NULL,
    model_version VARCHAR(50) DEFAULT 'ForestFaceNet_v1',
    quality_score REAL DEFAULT 1.0,
    face_pose_index INTEGER DEFAULT 0, -- 0: Front, 1: Left, 2: Right, 3: Up, 4: Down
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE
);

-- 5. LOCATIONS TABLE (Temporal Telemetry Index)
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id VARCHAR(36) NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    altitude REAL DEFAULT 0.0,
    accuracy REAL DEFAULT 0.0,
    speed REAL DEFAULT 0.0,
    bearing REAL DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE
);

-- 6. GEOFENCE ZONES TABLE
CREATE TABLE IF NOT EXISTS zones (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(30) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    polygon_geojson TEXT NOT NULL,
    min_lat REAL NOT NULL,
    max_lat REAL NOT NULL,
    min_lng REAL NOT NULL,
    max_lng REAL NOT NULL,
    active BOOLEAN DEFAULT 1,
    color_hex VARCHAR(10) NOT NULL
);

-- 7. ZONE EVENTS TABLE (Containment Transitions)
CREATE TABLE IF NOT EXISTS zone_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id VARCHAR(36) NOT NULL,
    zone_id VARCHAR(50) NOT NULL,
    previous_zone VARCHAR(50),
    current_zone VARCHAR(50) NOT NULL,
    event_type VARCHAR(20) NOT NULL CHECK(event_type IN ('ENTRY', 'EXIT', 'TRANSITION')),
    severity VARCHAR(20) NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE,
    FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE CASCADE
);

-- 8. DETERMINISTIC RISK EVENTS TABLE
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id VARCHAR(36) NOT NULL,
    score INTEGER NOT NULL CHECK(score >= 0 AND score <= 100),
    severity VARCHAR(20) NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    reasons_json TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE
);

-- 9. EMERGENCY SOS EVENTS TABLE
CREATE TABLE IF NOT EXISTS sos_events (
    sos_id VARCHAR(36) PRIMARY KEY,
    visitor_id VARCHAR(36) NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    initial_lat REAL NOT NULL,
    initial_lng REAL NOT NULL,
    initial_zone VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'TRIGGERED' CHECK(status IN ('TRIGGERED', 'ACKNOWLEDGED', 'DISPATCHED', 'RESOLVED', 'CANCELLED')),
    acknowledged_by_officer_id VARCHAR(36),
    resolution_notes TEXT,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE,
    FOREIGN KEY (acknowledged_by_officer_id) REFERENCES officers(officer_id) ON DELETE SET NULL
);

-- 10. SYSTEM ALERTS TABLE
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type VARCHAR(50) NOT NULL CHECK(alert_type IN ('ZONE_VIOLATION', 'CRITICAL_ZONE', 'UNKNOWN_PERSON', 'SOS_TRIGGERED', 'PROLONGED_STATIONARY')),
    severity VARCHAR(20) NOT NULL CHECK(severity IN ('WARNING', 'DANGER', 'CRITICAL')),
    visitor_id VARCHAR(36),
    officer_id VARCHAR(36),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'UNREAD' CHECK(status IN ('UNREAD', 'ACKNOWLEDGED', 'RESOLVED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id) ON DELETE CASCADE,
    FOREIGN KEY (officer_id) REFERENCES officers(officer_id) ON DELETE SET NULL
);

-- 11. SOS CHUNKED VIDEO RECORDINGS
CREATE TABLE IF NOT EXISTS sos_video_chunks (
    id VARCHAR(36) PRIMARY KEY,
    sos_id VARCHAR(36) NOT NULL,
    sequence_number INTEGER NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    file_hash_sha256 VARCHAR(64) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sos_id) REFERENCES sos_events(sos_id) ON DELETE CASCADE
);

-- INDEXES FOR TEMPORAL AND SPATIAL EFFICIENCY
CREATE INDEX IF NOT EXISTS idx_locations_visitor_ts ON locations(visitor_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_zone_events_visitor ON zone_events(visitor_id);
CREATE INDEX IF NOT EXISTS idx_risk_events_visitor ON risk_events(visitor_id);
CREATE INDEX IF NOT EXISTS idx_sos_events_status ON sos_events(status);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
