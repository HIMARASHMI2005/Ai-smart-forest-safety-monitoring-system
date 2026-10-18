# Biometric Privacy & Template Security Architecture

## 1. Regulatory Context & Privacy Mandate
In wildland conservation, national park management, and eco-tourism corridors, visitor safety must be balanced with strict privacy protections. Facial biometric verification is deployed exclusively to confirm authorized visitor entry, prevent unauthorized impersonation, and identify lost or distressed individuals during emergency Search-and-Rescue (SAR) operations.

The Smart Forest System strictly conforms to:
- **GDPR Article 9 (Special Category Biometric Data)**
- **ISO/IEC 24745: Information Technology — Biometric Information Protection**
- **Digital Personal Data Protection Act (DPDPA 2023)**

---

## 2. Core Architectural Privacy Guarantees

```
+-------------------------------------------------------------------------------+
|                      EPHEMERAL BIOMETRIC INGESTION PIPELINE                   |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [ Camera Sensor (Browser getUserMedia) ]                                     |
|           |                                                                   |
|           v (TLS 1.3 In-Flight Encryption)                                    |
|  [ In-Memory Byte Stream (Volatile RAM) ]                                     |
|           |                                                                   |
|           +---> MTCNN / OpenCV Crop & 112x112 Alignment                       |
|           |                                                                   |
|           +---> ForestFaceNet Forward Pass                                    |
|           |                                                                   |
|           +---> 128-D Floating-Point Vector Generated                         |
|           |                                                                   |
|           +---> [ RAW IMAGE BUFFER OVERWRITTEN / FREED IMMEDIATELY ]          |
|                                                                               |
|  [ Persistent Database Layer (SQLAlchemy ORM) ]                               |
|           |                                                                   |
|           +---> Only 128-D Float Array Committed to biometric_templates       |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### 2.1 Zero Raw Facial Image Persistence Policy
- **Absolute Non-Persistence:** The platform **never writes raw facial photos, uncompressed frames, or cropped face images to disk or cloud object stores**.
- **Ephemeral Processing:** Incoming base64/JPEG payloads are decoded directly into volatile CPU memory buffers (RAM).
- **Immediate Memory Deallocation:** As soon as ForestFaceNet generates the 128-dimensional embedding, the underlying image matrix is overwritten and released to garbage collection.

### 2.2 Mathematical Irreversibility of 128-D Embeddings
- A 128-dimensional vector $\mathbf{v} \in \mathbb{R}^{128}$ on the unit sphere represents an extreme mathematical projection:
  $$\mathbf{X} \in \mathbb{R}^{112 \times 112 \times 3} \quad (37,632 \text{ dimensions}) \quad \xrightarrow{\text{ForestFaceNet}} \quad \mathbf{v} \in \mathbb{S}^{127} \quad (128 \text{ dimensions})$$
- This is a many-to-one dimensionality reduction ratio of **$294 : 1$**.
- It is mathematically impossible to reconstruct the original human likeness, eye color, skin texture, or facial features from the 128-D vector alone without an adversarial generative surrogate.

---

## 3. Storage Schema & Encryption at Rest

In the relational database, biometric records are isolated in `biometric_templates`:
```sql
CREATE TABLE biometric_templates (
    id VARCHAR(36) PRIMARY KEY,
    visitor_id VARCHAR(36) NOT NULL REFERENCES visitors(visitor_id) ON DELETE CASCADE,
    embedding_vector BLOB NOT NULL,           -- Exactly 128 x 32-bit IEEE 754 floats (512 bytes)
    model_version VARCHAR(32) NOT NULL,       -- "ForestFaceNet_v1"
    quality_score FLOAT NOT NULL,             -- Landmark sharpness & pose score (0.0 - 1.0)
    face_pose_index INTEGER NOT NULL,         -- 0: Front, 1: Left 15, 2: Right 15, 3: Up 10, 4: Down 10
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.1 Field-Level Access Controls
- Biometric vectors cannot be queried by unauthorized users or officers.
- Only the backend's internal verification service reads vector blobs during `POST /api/auth/face-verify`.
- No visitor biometric vectors are ever transmitted across WebSocket broadcast channels.

---

## 4. Multi-Angle Enrollment & Quality Gating
To prevent low-confidence matches or noisy templates, the registration flow enforces a 5-pose guided sequence:
1. **Pose 0:** Frontal ($0^\circ$)
2. **Pose 1:** Left yaw ($15^\circ$)
3. **Pose 2:** Right yaw ($15^\circ$)
4. **Pose 3:** Pitch up ($10^\circ$)
5. **Pose 4:** Pitch down ($10^\circ$)

Each frame must pass a quality threshold ($\text{quality\_score} \ge 0.70$) assessing illumination uniformity and facial sharpness before being committed as a template.
