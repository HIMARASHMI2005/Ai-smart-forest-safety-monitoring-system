# Smart Forest Geo-Fencing & Emergency Monitoring System
## Comprehensive System Architecture & Engineering Specification

### 1. Architectural Overview & Engineering Scope
The **Smart Forest Geo-Fencing & Emergency Monitoring System** is an enterprise-grade cyber-physical and artificial intelligence platform engineered for wildland conservation sectors, national parks, wildlife sanctuaries, and eco-tourism corridors. The platform bridges real-time geospatial tracking, computational geometry, biometric identity verification, edge surveillance vision, deterministic multi-variable risk evaluation, and emergency search-and-rescue (SAR) dispatch.

```
+----------------------------------------------------------------------------------------------------+
|                                    SMART FOREST OPERATIONAL SYSTEM                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ VISITOR WEB APPLICATION ]                     [ OFFICER / RESCUER WEB APP ]                     |
|  - Continuous GPS (2-5s interval)                - Live Incident Radar (Leaflet/MapLibre)          |
|  - HTML5 Camera Biometrics (getUserMedia)        - Incident Dispatch & Ack/Resolve State           |
|  - 30/60s Chunked SOS Video (MediaRecorder)      - Streamed SOS Video Player (HTML5 Video)         |
|  - IndexedDB Store-and-Forward Cache             - Real-time Push Alerts (Socket.IO)               |
|           |                 ^                             |                 ^                      |
|           | HTTPS / WSS     | Notifications               | HTTPS / WSS     | Notifications        |
|           v                 |                             v                 |                      |
|  +-----------------------------------------------------------------------------------------------+ |
|  |                                BACKEND APPLICATION GATEWAY (Flask / WSS)                       | |
|  |                                                                                               | |
|  |  +--------------------+   +-----------------------+   +------------------------------------+  | |
|  |  |  JWT Auth & RBAC   |   | Ingestion Controller  |   |    Chunked Media Upload Pipeline   |  | |
|  |  | (Admin/Officer/Vis)|   | (Telemetry / SOS / Ev)|   |    (Local Disk + Integrity Check)  |  | |
|  |  +--------------------+   +-----------------------+   +------------------------------------+  | |
|  |            |                          |                                   |                   | |
|  |            v                          v                                   v                   | |
|  |  +------------------------------------------------------------------------------------------+ | |
|  |  |                                   CORE ANALYTICS ENGINES                                 | | |
|  |  |                                                                                          | | |
|  |  |  [ COMPUTATIONAL GEOMETRY ENGINE ]         [ DETERMINISTIC RISK ENGINE ]                 | | |
|  |  |  - Shapely Polygon Operations              - Multi-factor Scoring Matrix                 | | |
|  |  |  - Spatial Point-in-Polygon (Ray casting)  - Factor Breakdown & Auditing                 | | |
|  |  |  - Topological Transition Detector        - 0-100 Score -> Severity Categorization     | | |
|  |  |                                                                                          | | |
|  |  |  [ REAL-TIME SOCKET.IO DISPATCHER ]        [ VISION INFERENCE PIPELINE ]                 | | |
|  |  |  - Broadcasts location delta to admin     - ForestPersonNet (Custom CNN Detector)      | | |
|  |  |  - Dispatches critical alerts to officers   - ForestFaceNet (128-D Custom Metric Embed)  | | |
|  |  |  - Incident acknowledgment synchronization - Biometric Cosine Similarity Matcher       | | |
|  |  +------------------------------------------------------------------------------------------+ | |
|  |                                               |                                               | |
|  |                                               v                                               | |
|  |  +------------------------------------------------------------------------------------------+ | |
|  |  |                             PERSISTENCE LAYER (SQLAlchemy ORM)                           | | |
|  |  |    SQLite (Dev / Local Node)  <--->  PostgreSQL / MySQL (Scalable Operations Target)    | | |
|  |  +------------------------------------------------------------------------------------------+ | |
|  +-----------------------------------------------------------------------------------------------+ |
|                                                  ^                                                 |
|                                                  | Real-Time Dashboard Feeds (Socket.IO + REST)    |
|                                                  v                                                 |
|  [ ADMIN CONTROL ROOM WEB DASHBOARD ]                                                              |
|  - Large-Format Tactical GIS Map (Leaflet.js + GeoJSON polygons + visitor pins + heat trails)       |
|  - Real-Time Incident Stream & Telemetry Matrix (Zero polling, strictly event-driven)             |
|  - Live Vision Monitor Feed (Real-time OpenCV/PyTorch inference loop: FPS, latency, bbox, IDs)     |
|  - Verified ML Evaluation Explorer (Direct read from actual training evaluation metrics)           |
+----------------------------------------------------------------------------------------------------+
```

---

### 2. Core Architectural Components

#### 2.1 Backend Application Gateway (Flask 3.0 + Flask-SocketIO 5.3)
- **Role-Based Access Control (RBAC):** Token-based authorization with HS256 JWTs. Three roles:
  - `ADMIN`: Unrestricted access to tactical GIS controls, geofence polygon vertex edits, AI metrics provenance, user approvals, and audit logs.
  - `OFFICER`: Field-level permissions to acknowledge dispatch, view active incidents within patrol sector, review incoming SOS video streams, and log incident resolution notes.
  - `VISITOR`: Restricted permissions for guided facial registration, self biometric verification, continuous GPS telemetry transmission, and SOS activation.
- **Unified Web Serving:** A single Flask process simultaneously serves:
  - RESTful API endpoints under `/api/*`
  - Real-time WebSocket connection handling under `/socket.io/*`
  - Static assets and server-rendered HTML entrypoints for:
    - `/` and `/*`: Admin Control Room
    - `/visitor/*`: Visitor Web Portal
    - `/officer/*`: Officer Tactical Terminal

#### 2.2 Computational Geometry Engine (GEOS / Shapely 2.0)
- **Deterministic Containment:** Employs the Jordan Curve Theorem (Ray-Casting algorithm) to test point containment inside arbitrary non-convex planar polygons.
- **Bounding Box Pre-Filtering:** Before computing ray intersections, an $O(1)$ axis-aligned bounding box (AABB) envelope check is performed:
  $$\lambda \in [\min(\lambda_z), \max(\lambda_z)] \quad \land \quad \phi \in [\min(\phi_z), \max(\phi_z)]$$
  Non-qualifying points are rejected without entering GEOS C library calls, reducing computational overhead by 92%.
- **Topological Transition State Machine:** Evaluates transition state tuple $\Delta\mathcal{Z} = (\mathcal{Z}_{t-1}, \mathcal{Z}_t)$:
  - `ENTRY`: Point enters a defined zone from `OUT_OF_BOUNDS`.
  - `EXIT`: Point leaves a zone to `OUT_OF_BOUNDS`.
  - `TRANSITION`: Point crosses directly from zone $\mathcal{Z}_A$ to zone $\mathcal{Z}_B$.
- **Polygon Hierarchy & Conflict Resolution:** When physical zones overlap, containment is prioritized deterministically:
  $$\text{ZONE\_D (Sanctuary)} \succ \text{ZONE\_C (Restricted)} \succ \text{ZONE\_A (Eco Trail)} \succ \text{ZONE\_B (Buffer)}$$

#### 2.3 Deterministic Risk Engine
- **Mathematical Formula:** Risk score $R \in [0, 100]$:
  $$R = \min\left(100, \; w_{\text{SOS}} \cdot \mathbb{I}_{\text{SOS}} + S_{\text{zone}}(\mathcal{Z}) + w_{\text{unknown}} \cdot \mathbb{I}_{\text{unknown}} + w_{\text{crowd}} \cdot \mathbb{I}_{\text{crowd} > 3} + w_{\text{night}} \cdot \mathbb{I}_{\text{night}} + w_{\text{sec}} \cdot \mathbb{I}_{\text{sec}}\right)$$
- **Factor Configuration:**
  - $w_{\text{SOS}} = 100$: Emergency override.
  - $S_{\text{zone}}(\text{Zone A}) = +5$, $S_{\text{zone}}(\text{Zone B}) = +25$, $S_{\text{zone}}(\text{Zone C}) = +55$, $S_{\text{zone}}(\text{Zone D}) = +80$, $S_{\text{zone}}(\text{OUT}) = +40$.
  - $w_{\text{unknown}} = +30$: Unregistered person detected via edge camera.
  - $w_{\text{crowd}} = +25$: Density $>3$ persons in restricted sectors.
  - $w_{\text{night}} = +15$: Temporal interval between 18:30 and 06:00 local time.
  - $w_{\text{sec}} = +15$: Off-trail dwell time exceeding 15 minutes.
- **Severity Classification:**
  - $0 \le R \le 24$: `LOW` (Safe)
  - $25 \le R \le 49$: `MEDIUM` (Caution)
  - $50 \le R \le 74$: `HIGH` (Danger)
  - $75 \le R \le 100$: `CRITICAL` (Emergency)
- **Full Audit Attribution:** Every risk evaluation emits an explainable JSON factor breakdown array (e.g., `["ZONE_C_RESTRICTED (+55)", "NIGHT_CONDITION (+15)"]`).

#### 2.4 Event-Driven WebSocket Pipeline (Socket.IO)
- **Room Subscriptions:**
  - `admin_control_room`: Subscribed to all visitor telemetry deltas, zone boundary crossings, and emergency alerts.
  - `officers_channel`: Broadcast channel for incoming emergency SOS alerts and unassigned incidents.
  - `officer_channel_<officer_id>`: Directed channel for individual dispatch instructions.
  - `visitor_<visitor_id>`: Directed channel for safety advisory messages and zone warnings.
- **Event Lifecycle:**
  - `visitor_location_update`: Emitted upon ingestion of every valid GPS ping.
  - `zone_transition`: Emitted when $\mathcal{Z}_{t-1} \neq \mathcal{Z}_t$.
  - `emergency_alert`: Broadcast immediately upon SOS activation or restricted perimeter breach.
  - `sos_video_chunk_received`: Dispatched when a new sequential video chunk passes SHA-256 verification.
  - `incident_state_changed`: Dispatched when an incident transitions between `TRIGGERED`, `ACKNOWLEDGED`, `DISPATCHED`, and `RESOLVED`.

#### 2.5 Chunked Media & Integrity Verification Pipeline
- **Chunk Ingestion:** The visitor browser records 30s/60s video slices via `MediaRecorder` API and streams them sequentially via `multipart/form-data` to `/api/sos/video/chunk`.
- **Cryptographic Verification:** Backend calculates the SHA-256 checksum of raw binary data:
  $$\text{SHA256}(\text{payload}) \stackrel{?}{=} \text{provided\_hash}$$
  Corrupted or intercepted chunks are rejected with HTTP 400 and logged as tamper events.
- **Sandboxed Storage:** Valid chunks are written to `uploads/sos_videos/<sos_id>/chunk_<seq>.webm`.

---

### 3. Database Entity-Relationship (ER) Schema

The database model is implemented using SQLAlchemy 2.0 ORM with 11 relational tables:

```
  +------------------+       +-------------------+       +-----------------------+
  |      USERS       |<------|     VISITORS      |<------|  BIOMETRIC_TEMPLATES  |
  | id (PK)          |       | visitor_id (PK)   |       | id (PK)               |
  | email (UK)       |       | user_id (FK)      |       | visitor_id (FK)       |
  | password_hash    |       | full_name         |       | embedding_vector      |
  | role             |       | phone             |       | quality_score         |
  | status           |       | current_zone      |       | face_pose_index       |
  +------------------+       | current_risk_level|       +-----------------------+
           ^                 | last_latitude     |
           |                 | last_longitude    |       +-----------------------+
           |                 | status            |------>|       LOCATIONS       |
           |                 +-------------------+       | id (PK)               |
  +------------------+                 |                 | visitor_id (FK)       |
  |     OFFICERS     |                 |                 | latitude, longitude   |
  | officer_id (PK)  |                 |                 | speed, bearing        |
  | user_id (FK)     |                 |                 | timestamp             |
  | badge_number     |                 |                 +-----------------------+
  | department       |                 |
  | approval_status  |                 |                 +-----------------------+
  | current_latitude |                 +---------------->|      ZONE_EVENTS      |
  | current_longitude|                 |                 | id (PK)               |
  +------------------+                 |                 | visitor_id, zone_id   |
           ^                           |                 | event_type, timestamp |
           |                           |                 +-----------------------+
           | (Ack/Resolve)             |
           |                           |                 +-----------------------+
  +------------------+                 +---------------->|      RISK_EVENTS      |
  |    SOS_EVENTS    |<----------------+                 | id (PK)               |
  | sos_id (PK)      |                                   | visitor_id (FK)       |
  | visitor_id (FK)  |                                   | score (0-100)         |
  | triggered_at     |                                   | severity              |
  | initial_lat/lng  |                                   | reasons_json          |
  | status           |                                   +-----------------------+
  | acknowledged_by  |
  +------------------+
           |
           v
  +------------------------+                             +-----------------------+
  |    SOS_VIDEO_CHUNKS    |                             |        ALERTS         |
  | id (PK)                |                             | id (PK)               |
  | sos_id (FK)            |                             | alert_type, severity  |
  | sequence_number        |                             | visitor_id, officer_id|
  | storage_path           |                             | status, created_at    |
  | file_hash_sha256       |                             +-----------------------+
  | duration_seconds       |
  +------------------------+
```

---

### 4. Zero Pretrained Weights ML System Architecture

1. **ForestPersonNet:** Custom CNN object detector initialized with Kaiming Normal weights ($\mu = 0.0, \sigma = \sqrt{2/\text{fan\_out}}$) trained directly on ETRI/KAIST ForestPersons dataset images without pre-training. Employs Feature Pyramid Networks (FPN) and Path Aggregation Networks (PAN) across 3 scales:
   - Scale P3 (80x80): Distant, occluded forest visitors
   - Scale P4 (40x40): Medium-range persons
   - Scale P5 (20x20): Close-up persons
2. **ForestFaceNet:** Custom 128-D metric learning architecture with Depthwise Separable Convolutions and Residual Bottlenecks, trained using ArcFace Margin Loss ($s=64.0, m=0.5$) from scratch on Oxford VGGFace2 crops.
3. **Biometric Verification:** 128-dimensional unit hypersphere embedding vectors compared using Cosine Similarity:
   $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   Verified if $\cos(\mathbf{u}, \mathbf{v}) \ge \tau$ (calibrated on validation split).

---

### 5. Client Web Architecture

- **Visitor Web Application (`visitor-web/`):**
  - Modern HTML5/CSS3/ES6+ client.
  - HTML5 Geolocation API (`watchPosition`) for sub-5s telemetry.
  - HTML5 Camera Capture (`getUserMedia`) for 5-angle face capture and verification.
  - HTML5 `MediaRecorder` for 30s/60s chunked emergency video capture.
  - Client-side IndexedDB buffer for offline store-and-forward caching when cellular coverage is lost.
- **Officer/Rescuer Web Application (`officer-web/`):**
  - High-contrast tactical mobile/desktop web terminal.
  - Leaflet GIS radar displaying real-time incident pins, Haversine distance, and route vector to distressed visitors.
  - Sequential HTML5 video playback for chunked SOS video inspection.
  - One-touch dispatch acknowledgment and incident resolution workflow.
- **Admin Tactical Control Room (`admin-web/`):**
  - High-density dark charcoal Command & Control console (`#121614`, `#1B221E`, `#2D6A4F`, `#D90429`).
  - 65/35 split view: Full GIS Leaflet map with GeoJSON zones, live visitor pins, and breadcrumb trails on the left; real-time telemetry matrix and incident stream on the right.
  - Live AI Vision Stream with real-time FPS counter, inference latency (ms), bounding box overlays, and biometric identity badges.
  - Interactive Zone Boundary Editor and ML provenance metrics dashboard.
