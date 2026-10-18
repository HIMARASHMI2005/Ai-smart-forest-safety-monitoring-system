# SMART FOREST GEO-FENCING & EMERGENCY MONITORING SYSTEM
### Comprehensive System Architecture, Custom Machine Learning & Cyber-Physical Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Shapely](https://img.shields.io/badge/Shapely-2.0+-blue.svg)](https://shapely.readthedocs.io/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg)](https://leafletjs.com/)
[![Tests](https://img.shields.io/badge/Tests-15%2F15%20Passing-success.svg)](run_tests.py)

---

## 1. System Overview & Engineering Scope
The **Smart Forest Geo-Fencing & Emergency Monitoring System** is an integrated cyber-physical and artificial intelligence platform designed for wildland conservation zones, national parks, wildlife sanctuaries, and eco-tourism corridors. The platform bridges real-time geospatial tracking, computational geometry, biometric identity verification, edge camera surveillance, deterministic multi-variable risk evaluation, and emergency search-and-rescue (SAR) dispatch.

### Core Tenets
1. **Zero Pretrained Weights:** Every neural network (`ForestPersonNet` for human detection, `ForestFaceNet` for facial embedding extraction) is initialized from scratch (Kaiming Normal weight initialization) and trained strictly on official ground-truth datasets (**ETRI/KAIST ForestPersons** and **Oxford VGGFace2**). No fine-tuning from ImageNet, COCO, or commercial checkpoints.
2. **Deterministic Mathematical Geofencing:** Geofence containment is computed via computational geometry (Point-in-Polygon ray-casting algorithm via GEOS/Shapely), guaranteeing zero spatial hallucination.
3. **Transparent & Explainable Risk Scoring:** Emergency threat scores ($0 - 100$) are calculated via deterministic, rule-based inference with complete attribution tracking (e.g., night condition $+15$, restricted zone $+55$, unknown face $+30$, SOS $= 100$).
4. **Command & Control Aesthetics:** Built for operational control rooms. High-density, dark charcoal and forest green palette (`#121614`, `#1B221E`, `#2D6A4F`, `#D90429`), free of superficial design trends or gradients.
5. **Unified Web Architecture:** A single Flask + Flask-SocketIO backend serves three browser-based web portals:
   - **Visitor Web Portal** (`/visitor/*`)
   - **Officer / Rescuer Web Portal** (`/officer/*`)
   - **Admin Control Room** (`/` and `/*`)

---

## 2. System Architecture Diagram

```text
                         SMART FOREST OPERATIONAL SYSTEM
                                        |
              +-------------------------+-------------------------+
              |                         |                         |
       Visitor Web App           Officer Web App           Admin Web App
     (HTML5 / Geolocation      (Leaflet GIS Radar /      (Tactical Control Room /
      / MediaRecorder /        Incident Triage /          Split Map / AI Stream /
       IndexedDB Cache)        Video Inspection)          Boundary Manager)
              |                         |                         |
              +-------------------------+-------------------------+
                                        |
                          Flask Application Gateway
                        (REST APIs + Flask-SocketIO)
                                        |
       +--------------------------------+--------------------------------+
       |                                |                                |
   GPS Engine                     Risk Engine                      Vision Engine
  (Shapely PIP /                 (Deterministic                   (ForestPersonNet /
   BBox Prefilter /               $0-100$ Rule Scoring /          ForestFaceNet 128-D /
   Transition Calculus)           Factor Attribution)             Cosine Matcher)
       |                                |                                |
       +--------------------------------+--------------------------------+
                                        |
                                 SQLAlchemy ORM
                          (SQLite / PostgreSQL Target)
                                        |
                            Incident & SOS Pipeline
                    (30s/60s Video Chunks + SHA-256 Integrity)
                                        |
                            Real-time Socket.IO Events
                  (Location Deltas / Transitions / Alerts)
```

---

## 3. Technology Stack & Matrix

| Layer / Subsystem | Primary Technology | Version / Spec | Justification |
|:---|:---|:---|:---|
| **Backend Runtime** | Python | 3.10 / 3.11 | High performance, native compatibility with PyTorch, Shapely, OpenCV |
| **Web API & WebSockets** | Flask + Flask-SocketIO | Flask 3.0+, SocketIO 5.3+ | Event-driven architecture, zero polling overhead |
| **ORM & Database** | SQLAlchemy | 2.0+ (SQLite / PostgreSQL) | Strongly-typed schema definitions, relational integrity (11 tables) |
| **Spatial Computation** | Shapely / GEOS | 2.0+ | Exact planar topology algorithms (Jordan Curve Theorem ray-casting) |
| **Deep Learning Engine** | PyTorch | 2.2+ (CPU & CUDA) | Random Kaiming Normal initialization, autograd, zero pretrained weights |
| **Computer Vision Tool** | OpenCV (`opencv-python`) | 4.9+ | Frame capture, color space conversions, facial landmark alignment |
| **Admin Web Frontend** | Vanilla HTML5 / ES6+ JS | Native ES6 | Fast, zero build toolchain dependencies, high reliability |
| **Mapping & GIS Engine** | Leaflet.js | 1.9.4 | Native GeoJSON layer support, low memory consumption |
| **Visitor Web Portal** | HTML5 + Geolocation + MediaRecorder | ES6+, IndexedDB | Zero-install trailhead access, continuous GPS, chunked SOS video |
| **Officer Web Portal** | HTML5 + Leaflet + Socket.IO | ES6+ | Instant dispatch alerts, Haversine distance tracking, video triage |

---

## 4. Directory & Repository Structure

```
smart-forest/
├── backend/
│   ├── app/
│   │   ├── __init__.py                # Flask application factory & web routes
│   │   ├── config.py                  # Environment settings & secrets
│   │   ├── api/                       # RESTful API Route Controllers
│   │   │   ├── auth_middleware.py     # JWT HS256 & RBAC authorization guards
│   │   │   ├── auth_routes.py         # Registration, login, face-register/verify
│   │   │   ├── location_routes.py     # GPS ingestion & breadcrumb trails
│   │   │   ├── zone_routes.py         # Geofence polygon CRUD operations
│   │   │   ├── sos_routes.py          # Emergency SOS lifecycle & video chunks
│   │   │   ├── visitor_routes.py      # Visitor registry & check-in management
│   │   │   ├── officer_routes.py      # Officer availability & approvals
│   │   │   ├── alert_routes.py        # System alert dispatch & triage
│   │   │   └── vision_routes.py       # Live camera inference & test metrics
│   │   ├── database/
│   │   │   ├── connection.py          # SQLAlchemy 2.0 database engine & sessions
│   │   │   └── models.py              # 11 relational database entity models
│   │   ├── geofence/
│   │   │   ├── geometry_engine.py     # BBox pre-filter, Shapely ray-casting
│   │   │   └── zone_manager.py        # Polygon cache & priority resolution
│   │   ├── risk_engine/
│   │   │   ├── rules.py               # Deterministic factor weights & conditions
│   │   │   └── evaluator.py           # Additive formula & explainable breakdown
│   │   ├── vision/
│   │   │   ├── detector_service.py    # ForestPersonNet inference handler
│   │   │   ├── face_service.py        # ForestFaceNet & ArcFace cosine matcher
│   │   │   └── pipeline.py            # End-to-end frame processing pipeline
│   │   ├── sos/
│   │   │   └── video_handler.py       # Chunked video ingestion & SHA-256 verifier
│   │   └── websocket/
│   │       └── events.py              # Flask-SocketIO room broadcasting
│   ├── tests/
│   │   ├── test_auth.py               # Password hashing & JWT validation tests
│   │   ├── test_geofence.py           # BBox & Ray-casting geofence tests
│   │   ├── test_risk_engine.py        # Multi-factor risk engine & SOS tests
│   │   ├── test_sos_video.py          # Chunk upload & SHA-256 tamper tests
│   │   └── test_vision_pipeline.py    # CNN forward & unit-norm tests
│   ├── requirements.txt               # Production Python dependencies
│   ├── .env.example                   # Environment configuration template
│   └── run.py                         # Application entrypoint
│
├── ml/                                # Zero Pretrained Weights ML Pipelines
│   ├── forest_person_net/             # Custom Human Detector
│   │   ├── architecture.py            # 5-Stage Backbone + FPN/PAN Neck + Heads
│   │   ├── loss.py                    # Multi-task CIoU + BCE loss
│   │   ├── dataset.py                 # ETRI/KAIST ForestPersons PyTorch Dataset
│   │   ├── train.py                   # Scratch training pipeline (AdamW + Cosine)
│   │   ├── evaluate.py                # Held-out test set evaluation (mAP@0.5)
│   │   └── infer.py                   # Bounding box decoder & NMS filter
│   ├── forest_face_net/               # Custom 128-D Metric Learning Network
│   │   ├── architecture.py            # Depthwise Separable CNN + L2 Unit Sphere
│   │   ├── margin_loss.py             # ArcFace Additive Angular Margin Loss
│   │   ├── dataset.py                 # VGGFace2 aligned crop dataset
│   │   ├── train.py                   # Scratch training pipeline (ArcFace SGD)
│   │   ├── evaluate.py                # Verification metrics (FAR, FRR, ROC, AUC)
│   │   └── verify.py                  # Cosine similarity matcher (Threshold tau)
│   ├── datasets/                      # Dataset Download & Verification Tools
│   │   ├── download_forest_persons.py # ETRI/KAIST format fetcher
│   │   └── download_vggface2.py       # Oxford VGGFace2 format fetcher
│   ├── preprocessing/                 # Augmentation & Face Alignment
│   │   ├── augmentations.py           # Geometry & illumination transforms
│   │   └── face_aligner.py            # 5-Point landmark affine normalizer
│   └── evaluation/
│       ├── metrics_calculator.py      # mAP, Confusion Matrix, EER calculator
│       └── plot_curves.py             # Precision-Recall & ROC curve generator
│
├── visitor-web/                       # Visitor Browser Portal
│   ├── templates/                     # Jinja2 / HTML5 responsive templates
│   │   ├── login.html, register.html
│   │   ├── face_registration.html     # 5-Angle guided facial enrollment
│   │   ├── face_login.html            # 1:1 Biometric camera verification
│   │   ├── dashboard.html             # GPS status, current zone, risk level
│   │   ├── sos.html, sos_active.html  # 2-Stage SOS button & emergency mode
│   │   ├── sos_history.html, gps.html
│   │   ├── zone_status.html, risk.html
│   │   └── profile.html
│   └── static/                        # CSS & Client-side JavaScript
│       ├── css/visitor.css
│       └── js/ (auth.js, gps.js, sos.js, face_registration.js, face_login.js)
│
├── officer-web/                       # Officer / Rescuer Tactical Portal
│   ├── templates/                     # Incident triage & navigation templates
│   │   ├── login.html, register.html
│   │   ├── approval.html              # Pending badge authorization screen
│   │   ├── dashboard.html             # High-priority active incident feed
│   │   ├── incidents.html             # Searchable incident registry
│   │   ├── incident_details.html      # Incident dossier & quick triage
│   │   ├── map.html, navigation.html  # Leaflet shortest-path rescue radar
│   │   ├── video.html                 # Sequential SOS chunk video player
│   │   ├── resolve.html               # Mandatory incident resolution report
│   │   └── profile.html
│   └── static/
│       ├── css/officer.css
│       └── js/ (auth.js, gps.js, incidents.js, map.js, video.js)
│
├── admin-web/                         # Command & Control Operations Dashboard
│   ├── templates/
│   │   ├── base.html                  # Master tactical frame layout
│   │   ├── dashboard.html             # Operations overview & stats
│   │   ├── live_map.html              # 65/35 Split GIS map & telemetry matrix
│   │   ├── vision_monitor.html        # Real-time camera feed & AI inspection
│   │   ├── zones.html                 # Interactive polygon boundary editor
│   │   ├── alerts.html                # Emergency alert queue
│   │   ├── visitors.html              # Visitor directory & breadcrumb trails
│   │   ├── ai_models.html             # Model provenance & training metrics
│   │   └── logs.html                  # Complete audit trail
│   └── static/
│       ├── css/tactical_theme.css     # Command center dark aesthetic
│       └── js/ (socket_client.js, tactical_map.js, vision_stream.js, alerts_manager.js)
│
├── database/
│   ├── schema.sql                     # Complete 11-table SQL relational schema
│   └── seed_zones.sql                 # Seed boundaries for Zones A, B, C, D
│
├── docs/                              # Engineering & Academic Documentation
│   ├── architecture.md                # End-to-end system architecture
│   ├── ml_training_specification.md   # Mathematical equations & hyperparameters
│   ├── datasets_attribution.md        # ETRI/KAIST & VGGFace2 benchmarks
│   ├── api_reference.md               # Complete OpenAPI / REST specification
│   ├── testing_guide.md               # Unit and integration test guide
│   ├── privacy_and_biometrics.md      # Ephemeral RAM & template security
│   └── viva_defense_guide.md          # Comprehensive defense Q&A for viva panels
│
├── run_tests.py                       # Standalone automated test runner
└── README.md                          # Master project documentation
```

---

## 5. Installation & Setup Guide

### 5.1 Prerequisites
- Python 3.10 or 3.11 installed.
- Modern Web Browser (Chrome, Firefox, Safari, Edge) with WebRTC and Geolocation enabled.

### 5.2 Environment Setup
1. Clone or open the repository directory:
   ```bash
   cd "d:/final major project12"
   ```
2. Create and activate a virtual environment (optional, or use system Python):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install required Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 6. Running the System

### 6.1 Start the Backend Server
Launch the unified application server:
```bash
python backend/run.py
```
Upon startup, the backend automatically:
- Initializes the SQLite database (`smart_forest.db`) and verifies all 11 relational tables.
- Seeds the initial geospatial zones (Zone A, Zone B, Zone C, Zone D) into the database.
- Binds to `http://0.0.0.0:5000` with WebSocket support enabled.

### 6.2 Accessing the Web Interfaces
Open your browser and navigate to:
- **Admin Control Room:** `http://localhost:5000/dashboard`
  - Default Admin Credentials: `admin@forest.gov` / `Admin@Forest2026!`
- **Visitor Portal:** `http://localhost:5000/visitor/login` (or `/visitor/register`)
- **Officer Portal:** `http://localhost:5000/officer/login` (or `/officer/register`)

---

## 7. Running the Automated Test Suite

To verify all subsystems (geofence engine, deterministic risk engine, authentication, video integrity, and vision models), execute:
```bash
python -u run_tests.py
```

### Benchmark Results
```text
======================================================================
RUNNING SMART FOREST AUTOMATED TEST SUITE
======================================================================

[TEST SUITE] Geofence Engine (backend.tests.test_geofence):
  [PASS] test_bbox_prefilter
  [PASS] test_contains_point_ray_casting
  [PASS] test_transition_detection

[TEST SUITE] Risk Engine (backend.tests.test_risk_engine):
  [PASS] test_multi_variable_scoring
  [PASS] test_risk_severity_categorization
  [PASS] test_score_clamping_at_100
  [PASS] test_sos_override

[TEST SUITE] Authentication & JWT (backend.tests.test_auth):
  [PASS] test_jwt_token_generation
  [PASS] test_password_hashing

[TEST SUITE] SOS Video & SHA-256 (backend.tests.test_sos_video):
  [PASS] test_chunk_integrity_verification
  [PASS] test_sha256_calculation

[TEST SUITE] Vision Pipeline & ML (backend.tests.test_vision_pipeline):
  [PASS] test_arcface_loss
  [PASS] test_cosine_similarity_computation
  [PASS] test_forest_face_net_forward_and_norm
  [PASS] test_forest_person_net_forward

======================================================================
TEST RESULTS: 15 PASSED, 0 FAILED
======================================================================
ALL TESTS COMPLETED SUCCESSFULLY!
```

---

## 8. Detailed Engineering Documentation

For in-depth mathematical formulations, dataset attributions, and defense preparation, refer to the documentation in [`docs/`](docs/):
- **[System Architecture](docs/architecture.md):** Complete component specifications, data flows, and ER diagrams.
- **[ML Training Specification](docs/ml_training_specification.md):** Kaiming Normal initialization, CIoU loss, ArcFace margin formulas, and hyperparameters.
- **[Datasets Attribution](docs/datasets_attribution.md):** ETRI/KAIST ForestPersons and Oxford VGGFace2 ground truth specifications.
- **[API Reference](docs/api_reference.md):** Full OpenAPI REST endpoints and Socket.IO event catalog.
- **[Testing Guide](docs/testing_guide.md):** Detailed test harness, edge case coverage, and verification.
- **[Biometric Privacy](docs/privacy_and_biometrics.md):** Ephemeral RAM buffers, 128-D mathematical irreversibility, and ISO/IEC 24745 compliance.
- **[Viva Defense Guide](docs/viva_defense_guide.md):** Key defense questions, mathematical proofs, and examiner expectations.
#   A i - s m a r t - f o r e s t - s a f e t y - m o n i t o r i n g - s y s t e m  
 