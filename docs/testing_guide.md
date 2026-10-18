# Automated Testing & Verification Guide

## 1. Overview of the Test Suite
The Smart Forest system includes an automated test harness covering critical mathematical, algorithmic, cryptographic, and neural network subsystems.

All tests operate independently of mock frameworks and validate actual execution logic.

---

## 2. Test Execution Commands

### Recommended (Standalone Direct Runner)
Execute the unified test runner directly from the project root:
```bash
python -u run_tests.py
```

### Pytest Execution
Alternatively, if `pytest` is installed in the active environment:
```bash
pytest backend/tests -v
```

---

## 3. Subsystem Test Specifications

### 3.1 Geofence Engine Tests (`test_geofence.py`)
- **`test_bbox_prefilter`:** Verifies that axis-aligned bounding box pre-checks correctly pass interior points and reject out-of-range points in $O(1)$ time without calling Shapely.
- **`test_contains_point_ray_casting`:** Validates ray-casting intersection logic against complex polygon vertex loops.
- **`test_transition_detection`:** Validates state machine output for `ENTRY`, `EXIT`, `TRANSITION`, and duplicate zone pings.

### 3.2 Deterministic Risk Engine Tests (`test_risk_engine.py`)
- **`test_multi_variable_scoring`:** Validates multi-factor additive formula:
  $$\text{Score} = S_{\text{zone}} + w_{\text{unknown}} + w_{\text{crowd}} + w_{\text{night}} + w_{\text{sec}}$$
- **`test_risk_severity_categorization`:** Confirms boundary classifications:
  - $0-24 \to \text{LOW}$
  - $25-49 \to \text{MEDIUM}$
  - $50-74 \to \text{HIGH}$
  - $75-100 \to \text{CRITICAL}$
- **`test_score_clamping_at_100`:** Ensures combined weights exceeding 100 are clamped strictly to 100.
- **`test_sos_override`:** Validates that active SOS flag overrides all parameters and forces the score to 100 with CRITICAL severity.

### 3.3 Authentication & Security Tests (`test_auth.py`)
- **`test_password_hashing`:** Confirms PBKDF2/SHA-256 password hashing and prevents plain-text storage.
- **`test_jwt_token_generation`:** Validates JWT claims packaging, role verification, and HS256 digital signature validation.

### 3.4 SOS Video Integrity & Cryptography Tests (`test_sos_video.py`)
- **`test_sha256_calculation`:** Confirms exact SHA-256 binary hash computation.
- **`test_chunk_integrity_verification`:** Verifies that:
  - A legitimate chunk payload with a valid hash commits to disk and DB.
  - A corrupted or tampered payload is rejected with checksum mismatch error.

### 3.5 Vision Pipeline & ML Tests (`test_vision_pipeline.py`)
- **`test_forest_person_net_forward`:** Validates the FPN/PAN multi-scale CNN forward pass and checks output tensor dimensions:
  - P3: $(1, 18, 80, 80)$
  - P4: $(1, 18, 40, 40)$
  - P5: $(1, 18, 20, 20)$
- **`test_forest_face_net_forward_and_norm`:** Verifies that ForestFaceNet outputs a 128-dimensional embedding and confirms strict unit L2 norm:
  $$\|\mathbf{e}\|_2 = 1.0 \pm 10^{-5}$$
- **`test_cosine_similarity_computation`:** Verifies orthogonal vectors ($0.0$) and identical unit vectors ($1.0$).
- **`test_arcface_loss`:** Validates that ArcFace additive angular margin loss computes forward loss and propagates backward gradients through weight matrices.

---

## 4. Test Results Benchmark (100% Pass Rate)

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
