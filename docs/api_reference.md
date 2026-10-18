# RESTful API & WebSocket Interface Specification

## 1. Authentication & Authorization Standard
- **Transport Security:** HTTPS / WSS.
- **Authorization Header:** `Authorization: Bearer <jwt_access_token>`
- **Token Format:** Signed JSON Web Token (HS256) containing `user_id`, `role`, and expiration `exp`.

---

## 2. REST Endpoints Catalog

### 2.1 Authentication Subsystem (`/api/auth`)

#### `POST /api/auth/register`
Registers a new system user (Visitor, Officer, or Admin).
- **Request Body:**
```json
{
  "email": "sarah.walker@email.com",
  "password": "SecurePassword123!",
  "role": "VISITOR",
  "full_name": "Sarah Walker",
  "phone": "+1-555-019-2831",
  "emergency_contact_name": "David Walker",
  "emergency_contact_phone": "+1-555-019-2832"
}
```
- **Response `201 Created`:**
```json
{
  "status": "success",
  "message": "User registered successfully",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "u-a1b2c3d4",
    "email": "sarah.walker@email.com",
    "role": "VISITOR",
    "visitor_id": "VST-708912"
  }
}
```

#### `POST /api/auth/login`
Validates credentials and issues an operational JWT.
- **Request Body:**
```json
{
  "email": "sarah.walker@email.com",
  "password": "SecurePassword123!"
}
```
- **Response `200 OK`:**
```json
{
  "status": "success",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "u-a1b2c3d4",
    "email": "sarah.walker@email.com",
    "role": "VISITOR",
    "visitor_id": "VST-708912"
  }
}
```

#### `POST /api/auth/face-register`
Registers a 128-dimensional biometric template.
- **Security:** Requires Visitor JWT.
- **Request Body:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "pose_index": 0
}
```
- **Response `201 Created`:**
```json
{
  "status": "success",
  "message": "Biometric template registered",
  "template_id": "tmpl-9921b",
  "quality_score": 0.94
}
```

#### `POST /api/auth/face-verify`
Performs 1:1 biometric login against enrolled templates.
- **Request Body:**
```json
{
  "email": "sarah.walker@email.com",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```
- **Response `200 OK`:**
```json
{
  "status": "success",
  "verified": true,
  "confidence": 0.88,
  "threshold": 0.65,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": { "id": "u-a1b2c3d4", "role": "VISITOR", "visitor_id": "VST-708912" }
}
```

---

### 2.2 Telemetry & Geofencing (`/api/location` & `/api/zones`)

#### `POST /api/location/update`
Continuous GPS ingestion endpoint.
- **Request Body:**
```json
{
  "visitor_id": "VST-708912",
  "latitude": 11.662145,
  "longitude": 76.628491,
  "altitude": 920.4,
  "accuracy": 4.2,
  "speed": 1.4,
  "bearing": 182.0,
  "timestamp": "2026-10-08T10:14:22Z"
}
```
- **Response `200 OK`:**
```json
{
  "status": "success",
  "zone": "ZONE_A",
  "risk_score": 5,
  "severity": "LOW",
  "transition": null
}
```

#### `GET /api/location/visitor/<id>/history`
Retrieves historical breadcrumb coordinates.
- **Response `200 OK`:**
```json
{
  "visitor_id": "VST-708912",
  "locations": [
    { "latitude": 11.6621, "longitude": 76.6284, "speed": 1.2, "timestamp": "2026-10-08T10:10:00Z" }
  ]
}
```

#### `GET /api/zones`
Fetches all configured spatial zones with GeoJSON boundaries.
- **Response `200 OK`:**
```json
[
  {
    "id": "zone-1",
    "name": "Zone A: Eco Trail Corridor",
    "code": "ZONE_A",
    "severity": "LOW",
    "color_hex": "#2D6A4F",
    "polygon_geojson": { "type": "Polygon", "coordinates": [[[76.62, 11.66], [76.63, 11.66], ...]] }
  }
]
```

---

### 2.3 Emergency SOS Lifecycle (`/api/sos`)

#### `POST /api/sos/start`
Triggers an emergency incident (instantly elevates risk to 100).
- **Request Body:**
```json
{
  "visitor_id": "VST-708912",
  "latitude": 11.6631,
  "longitude": 76.6295
}
```
- **Response `201 Created`:**
```json
{
  "status": "success",
  "sos_id": "SOS-20261008-9842",
  "message": "Emergency broadcast initiated. Rescuers alerted."
}
```

#### `POST /api/sos/video/chunk`
Multipart upload for 30s/60s sequential H.264/WebM emergency video clips.
- **Form Data:**
  - `sos_id`: `SOS-20261008-9842`
  - `sequence_number`: `1`
  - `file_hash_sha256`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - `duration_seconds`: `30`
  - `video_file`: Binary chunk blob
- **Response `200 OK`:**
```json
{
  "status": "success",
  "sequence_number": 1,
  "checksum_verified": true
}
```

#### `POST /api/sos/resolve`
Resolves an active emergency.
- **Security:** Officer or Admin JWT.
- **Request Body:**
```json
{
  "sos_id": "SOS-20261008-9842",
  "resolution_notes": "Visitor safely escorted to base camp; treated for minor dehydration."
}
```
- **Response `200 OK`:**
```json
{
  "status": "success",
  "message": "Incident marked as resolved."
}
```

---

### 2.4 Live Vision & AI Inspection (`/api/vision` & `/api/ml`)

#### `POST /api/vision/infer-frame`
Tactical frame evaluation through custom CNN pipelines.
- **Request Body:**
```json
{
  "frame_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```
- **Response `200 OK`:**
```json
{
  "detections": [
    {
      "bbox": [140, 80, 220, 310],
      "confidence": 0.89,
      "class": "person",
      "identity": "VST-708912",
      "face_similarity": 0.81
    }
  ],
  "fps": 28.4,
  "latency_ms": 35.2
}
```

#### `GET /api/ml/metrics`
Returns unedited validation benchmark metrics directly output by training pipelines.
- **Response `200 OK`:**
```json
{
  "forest_person_net": {
    "mAP_50": 0.842,
    "precision": 0.871,
    "recall": 0.819,
    "inference_fps_cpu": 32.5
  },
  "forest_face_net": {
    "eer": 0.038,
    "optimal_threshold": 0.65,
    "auc": 0.988,
    "far_at_threshold": 0.012
  }
}
```

---

## 3. WebSocket Event Specification (Socket.IO)

| Event Name | Direction | Payload Example | Channel / Target |
|:---|:---|:---|:---|
| `visitor_location_update` | Server $\to$ Client | `{ "visitor_id": "VST-102", "lat": 11.662, "lng": 76.628, "zone": "ZONE_A", "risk": 5 }` | `admin_control_room` |
| `zone_transition` | Server $\to$ Client | `{ "visitor_id": "VST-102", "from_zone": "ZONE_A", "to_zone": "ZONE_C", "severity": "HIGH" }` | `admin_control_room`, `officers_channel` |
| `emergency_alert` | Server $\to$ Client | `{ "sos_id": "SOS-98", "visitor_id": "VST-102", "lat": 11.66, "lng": 76.62, "risk": 100 }` | Broadcast to all Rooms |
| `sos_video_chunk_received` | Server $\to$ Client | `{ "sos_id": "SOS-98", "seq": 1, "url": "/uploads/sos_videos/SOS-98/chunk_1.webm" }` | `admin_control_room`, `officers_channel` |
| `incident_state_changed` | Server $\to$ Client | `{ "sos_id": "SOS-98", "status": "ACKNOWLEDGED", "officer": "OFF-402" }` | Broadcast to all Rooms |
