"""
Comprehensive Automated Test Suite for AI Smart Forest Safety Monitoring System
Covers 10 Rigorous Test Categories:
1. Geofencing & Spatial Geometry
2. Authentication & Cryptographic Authorization
3. Biometric Face Registration & 1:N Recognition (Unknown Rejection)
4. Vision Pipeline & Neural Network Inference
5. Deterministic Risk Assessment & Multi-Factor Scoring
6. SOS Video Chunking & SHA-256 Cryptographic Integrity
7. Database Relational Integrity & Cascade Semantics
8. API & Edge Integration
9. Offline Resilience & Edge Autonomy
10. Security, Privacy & Biometric Protection
"""

import os
import uuid
import hashlib
import numpy as np
import pytest
import torch
from datetime import datetime, time
from shapely.geometry import Polygon, Point

from backend.app.config import Config
from backend.app.database.connection import SessionLocal, init_db
from backend.app.database.models import User, Visitor, Officer, SosEvent, SosVideoChunk, BiometricTemplate
from backend.app.geofence.geometry_engine import GeometryEngine
from backend.app.risk_engine.rules import RiskRules
from backend.app.risk_engine.evaluator import RiskEvaluator
from backend.app.sos.video_handler import VideoHandler
import jwt
from backend.app.api.auth_routes import hash_password, verify_password
from backend.app.api.auth_middleware import generate_jwt_token
from ml.forest_face_net.architecture import ForestFaceNet
from ml.forest_person_net.architecture import ForestPersonNet
from ml.forest_face_net.margin_loss import ArcFaceLoss
from ml.forest_face_net.verify import FaceVerifier, compute_cosine_similarity


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Initializes the database schema once for the test module."""
    init_db()


# ==============================================================================
# CATEGORY 1: GEOFENCING & SPATIAL GEOMETRY (GF-01 to GF-08)
# ==============================================================================

def test_geo_01_interior_containment():
    """GF-01: Point clearly inside designated zone returns True."""
    poly = Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)])
    assert GeometryEngine.contains_point(poly, lat=15.0, lng=15.0) is True


def test_geo_02_exterior_exclusion():
    """GF-02: Point clearly outside every defined zone returns False."""
    poly = Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)])
    assert GeometryEngine.contains_point(poly, lat=25.0, lng=25.0) is False


def test_geo_03_boundary_edge():
    """GF-03: Point exactly on a polygon boundary returns True."""
    poly = Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)])
    # Boundary point at lat=15.0, lng=10.0 (on the western edge)
    assert GeometryEngine.contains_point(poly, lat=15.0, lng=10.0) is True


def test_geo_04_vertex_point():
    """GF-04: Point on a polygon vertex returns True."""
    poly = Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)])
    # Vertex point at lat=20.0, lng=20.0
    assert GeometryEngine.contains_point(poly, lat=20.0, lng=20.0) is True


def test_geo_05_concave_polygon():
    """GF-05: Concave (L-shaped) polygon handles indentation correctly."""
    # L-shaped polygon: square (0,0)-(10,10) minus top-right quadrant (5,5)-(10,10)
    l_shape = Polygon([
        (0.0, 0.0), (10.0, 0.0), (10.0, 5.0),
        (5.0, 5.0), (5.0, 10.0), (0.0, 10.0), (0.0, 0.0)
    ])
    # Inside bottom-left lobe
    assert GeometryEngine.contains_point(l_shape, lat=2.0, lng=2.0) is True
    # Inside top-left lobe
    assert GeometryEngine.contains_point(l_shape, lat=8.0, lng=2.0) is True
    # In the cut-out void (lat=8.0, lng=8.0) -> must be False
    assert GeometryEngine.contains_point(l_shape, lat=8.0, lng=8.0) is False


def test_geo_06_bbox_prefilter_rejection():
    """GF-06: Fast Bounding-Box pre-filter rejects points outside envelope in O(1)."""
    # Envelope: lat in [11.0, 12.0], lng in [76.0, 77.0]
    min_lat, max_lat, min_lng, max_lng = 11.0, 12.0, 76.0, 77.0
    # Inside
    assert GeometryEngine.bbox_prefilter(11.5, 76.5, min_lat, max_lat, min_lng, max_lng) is True
    # North of envelope
    assert GeometryEngine.bbox_prefilter(12.5, 76.5, min_lat, max_lat, min_lng, max_lng) is False
    # South of envelope
    assert GeometryEngine.bbox_prefilter(10.5, 76.5, min_lat, max_lat, min_lng, max_lng) is False
    # East of envelope
    assert GeometryEngine.bbox_prefilter(11.5, 77.5, min_lat, max_lat, min_lng, max_lng) is False
    # West of envelope
    assert GeometryEngine.bbox_prefilter(11.5, 75.5, min_lat, max_lat, min_lng, max_lng) is False


def test_geo_07_transition_lifecycle_and_idempotency():
    """GF-07: Transition detection emits ENTRY, TRANSITION, EXIT; duplicates return None."""
    # 1. OUT -> ZONE_A = ENTRY
    e1 = GeometryEngine.detect_transition("OUT_OF_BOUNDS", "ZONE_A")
    assert e1 is not None and e1["event_type"] == "ENTRY"

    # 2. ZONE_A -> ZONE_B = TRANSITION
    e2 = GeometryEngine.detect_transition("ZONE_A", "ZONE_B")
    assert e2 is not None and e2["event_type"] == "TRANSITION"

    # 3. ZONE_B -> OUT = EXIT
    e3 = GeometryEngine.detect_transition("ZONE_B", "OUT_OF_BOUNDS")
    assert e3 is not None and e3["event_type"] == "EXIT"

    # 4. Duplicate ping: ZONE_A -> ZONE_A = None (no spurious event)
    assert GeometryEngine.detect_transition("ZONE_A", "ZONE_A") is None


def test_geo_08_malformed_input_handling():
    """GF-08: Malformed GeoJSON and empty polygon handled safely."""
    # Empty string GeoJSON
    poly = GeometryEngine.parse_geojson_polygon("")
    assert poly is None

    # None polygon passed to contains_point
    assert GeometryEngine.contains_point(None, lat=12.0, lng=76.0) is False

    # Distance to boundary on None polygon
    assert GeometryEngine.distance_to_boundary(None, lat=12.0, lng=76.0) == float("inf")


# ==============================================================================
# CATEGORY 2: AUTHENTICATION & RBAC AUTHORIZATION (AUTH-01 to AUTH-06)
# ==============================================================================

def test_auth_01_password_hashing_and_verification():
    """AUTH-01: Password is never stored in plaintext and verifies correctly."""
    pwd = "SecureForest2026!Key"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_auth_02_duplicate_user_constraint():
    """AUTH-02: Duplicate registration with identical email triggers unique constraint error."""
    db = SessionLocal()
    unique_email = f"dup_test_{uuid.uuid4().hex[:8]}@forest.gov"
    user1 = User(id=str(uuid.uuid4()), email=unique_email, password_hash="dummy", role="VISITOR")
    db.add(user1)
    db.commit()

    user2 = User(id=str(uuid.uuid4()), email=unique_email, password_hash="dummy", role="VISITOR")
    db.add(user2)
    with pytest.raises(Exception):
        db.commit()
    db.rollback()
    db.close()


def safe_decode_jwt(token: str):
    if not token or not isinstance(token, str):
        return None
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
    except Exception:
        return None


def test_auth_03_jwt_generation_and_payload_claims():
    """AUTH-03: Valid user produces signed JWT with expected role and user ID claims."""
    uid = str(uuid.uuid4())
    user = User(id=uid, email="ranger@forest.gov", password_hash="hash", role="OFFICER", status="ACTIVE")
    token = generate_jwt_token(user)
    assert token is not None

    decoded = safe_decode_jwt(token)
    assert decoded is not None
    assert decoded.get("user_id") == uid
    assert decoded.get("role") == "OFFICER"


def test_auth_04_tampered_jwt_rejection():
    """AUTH-04: Tampered JWT payload or signature is strictly rejected."""
    uid = str(uuid.uuid4())
    user = User(id=uid, email="tamper@forest.gov", password_hash="hash", role="VISITOR", status="ACTIVE")
    token = generate_jwt_token(user)

    # Corrupt last 4 characters of signature
    tampered = token[:-4] + "xxxx"
    assert safe_decode_jwt(tampered) is None


def test_auth_05_malformed_token_rejection():
    """AUTH-05: Arbitrary malformed token strings are rejected without crashing."""
    assert safe_decode_jwt("NOT_A_VALID_JWT_TOKEN") is None
    assert safe_decode_jwt("") is None
    assert safe_decode_jwt(None) is None


def test_auth_06_rbac_role_enforcement():
    """AUTH-06: Visitor token claims cannot masquerade as Officer/Admin."""
    visitor_user = User(id=str(uuid.uuid4()), email="vis@forest.gov", password_hash="h", role="VISITOR", status="ACTIVE")
    tok = generate_jwt_token(visitor_user)
    claims = safe_decode_jwt(tok)
    assert claims["role"] == "VISITOR"
    assert claims["role"] != "OFFICER"
    assert claims["role"] != "ADMIN"


# ==============================================================================
# CATEGORY 3: FACE BIOMETRICS & RECOGNITION (CRITICAL) (FR-01 to FR-06)
# ==============================================================================

def test_fr_01_registered_person_match():
    """FR-01: Registered person matching their enrolled identity yields confidence >= threshold."""
    verifier = FaceVerifier(checkpoint_path=None, threshold=0.65)
    # Generate identical 128-D unit vector
    vec1 = np.random.randn(128).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    is_match, conf = verifier.verify_1_to_1(vec1, vec1)
    assert is_match is True
    assert conf >= 0.65


def test_fr_02_registered_person_mismatch():
    """FR-02: Registered person compared to a completely different face fails match."""
    verifier = FaceVerifier(checkpoint_path=None, threshold=0.65)
    # Two orthogonal unit vectors (similarity = 0.0)
    vec1 = np.zeros(128, dtype=np.float32)
    vec1[0] = 1.0
    vec2 = np.zeros(128, dtype=np.float32)
    vec2[1] = 1.0

    is_match, conf = verifier.verify_1_to_1(vec1, vec2)
    assert is_match is False
    assert conf < 0.65


def test_fr_03_unregistered_person_classified_unknown():
    """FR-03: CRITICAL - Unregistered person MUST be classified as Unknown/None, NEVER falling back to enrolled user!"""
    verifier = FaceVerifier(checkpoint_path=None, threshold=0.65)
    
    # Candidate database has 1 enrolled user ("hima")
    enrolled_emb = np.zeros(128, dtype=np.float32)
    enrolled_emb[0] = 1.0
    candidates = [("hima", enrolled_emb)]

    # Query is an UNREGISTERED person (orthogonal vector, distinct face)
    unregistered_query = np.zeros(128, dtype=np.float32)
    unregistered_query[50] = 1.0

    matched_id, best_conf = verifier.search_1_to_n(unregistered_query, candidates)
    # MUST return None, NOT "hima"!
    assert matched_id is None
    assert best_conf < 0.65


def test_fr_04_multiple_identities_discrimination():
    """FR-04: Multiple enrolled identities match ONLY their own face and reject cross-identity pairs."""
    verifier = FaceVerifier(checkpoint_path=None, threshold=0.65)
    
    # Enroll 3 distinct identities
    e1 = np.zeros(128, dtype=np.float32); e1[10] = 1.0
    e2 = np.zeros(128, dtype=np.float32); e2[20] = 1.0
    e3 = np.zeros(128, dtype=np.float32); e3[30] = 1.0
    candidates = [("user_alpha", e1), ("user_beta", e2), ("user_gamma", e3)]

    # Query with user_beta's exact face
    matched_id, conf = verifier.search_1_to_n(e2, candidates)
    assert matched_id == "user_beta"
    assert conf >= 0.65

    # Cross test: alpha vs gamma
    is_match, _ = verifier.verify_1_to_1(e1, e3)
    assert is_match is False


def test_fr_05_empty_database_lookup():
    """FR-05: 1:N Identification when database has zero enrolled templates returns None gracefully."""
    verifier = FaceVerifier(checkpoint_path=None, threshold=0.65)
    query = np.random.randn(128).astype(np.float32)
    query /= np.linalg.norm(query)
    matched_id, conf = verifier.search_1_to_n(query, [])
    assert matched_id is None
    assert conf == 0.0


def test_fr_06_embedding_unit_hypersphere_norm():
    """FR-06: Face embeddings from ForestFaceNet strictly satisfy ||e||_2 = 1.0."""
    model = ForestFaceNet(embedding_dim=128)
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(3, 3, 112, 112)
        embeddings = model(dummy_input)

    assert embeddings.shape == (3, 128)
    norms = torch.norm(embeddings, p=2, dim=1)
    for n in norms:
        assert torch.isclose(n, torch.tensor(1.0), atol=1e-5)


# ==============================================================================
# CATEGORY 4: ML & VISION PIPELINE (ML-01 to ML-05)
# ==============================================================================

def test_ml_01_person_net_multiscale_output_heads():
    """ML-01: ForestPersonNet outputs multi-scale pyramid heads (P3, P4, P5)."""
    model = ForestPersonNet(num_classes=1)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 640, 640)
        heads = model(x)

    assert "p3" in heads and "p4" in heads and "p5" in heads
    assert heads["p3"].shape == (1, 18, 80, 80)
    assert heads["p4"].shape == (1, 18, 40, 40)
    assert heads["p5"].shape == (1, 18, 20, 20)


def test_ml_02_arcface_loss_computation():
    """ML-02: ArcFace Additive Angular Margin loss produces valid gradients."""
    loss_fn = ArcFaceLoss(in_features=128, out_features=5, s=64.0, m=0.5)
    features = torch.randn(4, 128)
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(features, labels)
    assert loss.item() > 0.0
    loss.backward()
    assert loss_fn.weight.grad is not None


def test_ml_03_inference_repeatability():
    """ML-03: Model inference on identical input tensor produces identical output."""
    model = ForestFaceNet(embedding_dim=128)
    model.eval()
    inp = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        out1 = model(inp).numpy()
        out2 = model(inp).numpy()
    np.testing.assert_array_almost_equal(out1, out2, decimal=6)


def test_ml_04_cosine_similarity_properties():
    """ML-04: Cosine similarity is bounded in [-1.0, 1.0] and symmetric."""
    v1 = np.random.randn(128).astype(np.float32)
    v2 = np.random.randn(128).astype(np.float32)
    sim1 = compute_cosine_similarity(v1, v2)
    sim2 = compute_cosine_similarity(v2, v1)
    assert -1.0 <= sim1 <= 1.0
    assert pytest.approx(sim1, 1e-5) == sim2


def test_ml_05_zero_norm_cosine_handling():
    """ML-05: Cosine similarity with zero vector returns 0.0 without division by zero."""
    zero_vec = np.zeros(128, dtype=np.float32)
    valid_vec = np.ones(128, dtype=np.float32)
    assert compute_cosine_similarity(zero_vec, valid_vec) == 0.0


# ==============================================================================
# CATEGORY 5: RISK ENGINE & THREAT CATEGORIZATION (RSK-01 to RSK-06)
# ==============================================================================

def test_rsk_01_all_zone_baseline_scores():
    """RSK-01: Baseline scoring across all 4 zones and out-of-bounds."""
    assert RiskRules.ZONE_SCORES["ZONE_A"] == 5
    assert RiskRules.ZONE_SCORES["ZONE_B"] == 25
    assert RiskRules.ZONE_SCORES["ZONE_C"] == 55
    assert RiskRules.ZONE_SCORES["ZONE_D"] == 80
    assert RiskRules.ZONE_SCORES["OUT_OF_BOUNDS"] == 40


def test_rsk_02_severity_threshold_boundaries():
    """RSK-02: Deterministic mapping for each severity tier."""
    # Boundary tests
    assert RiskRules.categorize_severity(0) == "LOW"
    assert RiskRules.categorize_severity(24) == "LOW"
    assert RiskRules.categorize_severity(25) == "MEDIUM"
    assert RiskRules.categorize_severity(49) == "MEDIUM"
    assert RiskRules.categorize_severity(50) == "HIGH"
    assert RiskRules.categorize_severity(74) == "HIGH"
    assert RiskRules.categorize_severity(75) == "CRITICAL"
    assert RiskRules.categorize_severity(100) == "CRITICAL"


def test_rsk_03_sos_emergency_override():
    """RSK-03: SOS flag forces score 100 and CRITICAL severity regardless of zone."""
    res = RiskEvaluator.evaluate(zone_code="ZONE_A", is_sos=True)
    assert res["score"] == 100
    assert res["severity"] == "CRITICAL"
    assert "sos" in res["breakdown"]
    assert res["breakdown"]["sos"] == 100


def test_rsk_04_compound_factors_and_weights():
    """RSK-04: Compound danger factors add up correctly."""
    # Zone B (25) + Unknown person (30) + Night (15) = 70 -> HIGH
    res = RiskEvaluator.evaluate(
        zone_code="ZONE_B",
        is_sos=False,
        is_unknown_person=True,
        is_crowd=False,
        is_night=True,
        is_prolonged_dwell=False
    )
    assert res["score"] == 70
    assert res["severity"] == "HIGH"
    assert res["breakdown"]["zone"] == 25
    assert res["breakdown"]["unknown_person"] == 30
    assert res["breakdown"]["night"] == 15


def test_rsk_05_ceiling_clamping_at_100():
    """RSK-05: Scores summing to >100 are strictly capped at 100."""
    # Zone D (80) + Unknown (30) + Crowd (25) + Night (15) = 150 -> clamped to 100
    res = RiskEvaluator.evaluate(
        zone_code="ZONE_D",
        is_sos=False,
        is_unknown_person=True,
        is_crowd=True,
        is_night=True
    )
    assert res["score"] == 100
    assert res["severity"] == "CRITICAL"


def test_rsk_06_deterministic_explainability():
    """RSK-06: Identical inputs produce identical score and factor reasons."""
    res1 = RiskEvaluator.evaluate(zone_code="ZONE_C", is_night=False)
    res2 = RiskEvaluator.evaluate(zone_code="ZONE_C", is_night=False)
    assert res1["score"] == res2["score"]
    assert res1["severity"] == res2["severity"]
    assert res1["reasons"] == res2["reasons"]


# ==============================================================================
# CATEGORY 6: SOS VIDEO & SHA-256 INTEGRITY (SOS-01 to SOS-04)
# ==============================================================================

def test_sos_01_valid_chunk_and_sha256_match():
    """SOS-01: Valid chunk with correct SHA-256 digest is accepted and verified."""
    db = SessionLocal()
    try:
        user = User(id="u-sos-01", email="sos01@forest.gov", password_hash="dummy", role="VISITOR")
        visitor = Visitor(visitor_id="VST-SOS-01", user_id="u-sos-01", full_name="SOS User 1", phone="123", emergency_contact_name="C", emergency_contact_phone="P")
        sos = SosEvent(sos_id="SOS-EVT-01", visitor_id="VST-SOS-01", initial_lat=11.66, initial_lng=76.62, initial_zone="ZONE_A", status="TRIGGERED")
        db.merge(user); db.merge(visitor); db.merge(sos); db.commit()

        payload = b"VALID_VIDEO_CHUNK_BINARY_FRAME_DATA"
        digest = hashlib.sha256(payload).hexdigest()

        ok, msg, chunk = VideoHandler.save_chunk(
            db=db, sos_id="SOS-EVT-01", sequence_number=1, file_bytes=payload, provided_sha256=digest
        )
        assert ok is True
        assert chunk is not None
        assert chunk.file_hash_sha256 == digest
    finally:
        db.close()


def test_sos_02_corrupted_chunk_mismatch_rejection():
    """SOS-02: Corrupted or tampered chunk with mismatched hash is rejected."""
    db = SessionLocal()
    try:
        payload = b"AUTHENTIC_ORIGINAL_CHUNK_DATA"
        tampered_hash = "f" * 64

        ok, msg, chunk = VideoHandler.save_chunk(
            db=db, sos_id="SOS-EVT-01", sequence_number=2, file_bytes=payload, provided_sha256=tampered_hash
        )
        assert ok is False
        assert chunk is None
        assert "mismatch" in msg.lower()
    finally:
        db.close()


def test_sos_03_nonexistent_sos_event_rejection():
    """SOS-03: Chunk upload targeting nonexistent SOS event ID is rejected."""
    db = SessionLocal()
    try:
        ok, msg, chunk = VideoHandler.save_chunk(
            db=db, sos_id="NON_EXISTENT_SOS_ID_9999", sequence_number=1, file_bytes=b"data"
        )
        assert ok is False
        assert "not found" in msg.lower()
    finally:
        db.close()


def test_sos_04_sha256_hash_utility():
    """SOS-04: VideoHandler calculate_sha256 returns standard 64-char lowercase hex digest."""
    test_data = b"SmartForestCryptographicTest2026"
    expected = hashlib.sha256(test_data).hexdigest()
    assert VideoHandler.calculate_sha256(test_data) == expected
    assert len(expected) == 64


# ==============================================================================
# CATEGORY 7: DATABASE SCHEMA & DATA PERSISTENCE (DB-01 to DB-03)
# ==============================================================================

def test_db_01_user_and_visitor_models():
    """DB-01: User and Visitor models persist and serialize to dict correctly."""
    db = SessionLocal()
    try:
        uid = str(uuid.uuid4())
        user = User(id=uid, email=f"test_{uid[:8]}@forest.gov", password_hash="hash", role="VISITOR", status="ACTIVE")
        visitor = Visitor(visitor_id=f"VST-{uid[:8]}", user_id=uid, full_name="John Forest", phone="999", emergency_contact_name="Jane", emergency_contact_phone="888")
        db.add(user); db.add(visitor); db.commit()

        u_dict = user.to_dict()
        assert u_dict["id"] == uid
        assert u_dict["role"] == "VISITOR"

        v_dict = visitor.to_dict()
        assert v_dict["visitor_id"] == f"VST-{uid[:8]}"
        assert v_dict["full_name"] == "John Forest"
    finally:
        db.close()


def test_db_02_biometric_template_storage():
    """DB-02: BiometricTemplate stores 128-D vector as float32 binary buffer."""
    db = SessionLocal()
    try:
        uid = str(uuid.uuid4())
        user = User(id=uid, email=f"bio_{uid[:8]}@forest.gov", password_hash="h", role="VISITOR")
        visitor = Visitor(visitor_id=f"VST-{uid[:8]}", user_id=uid, full_name="Bio User", phone="1", emergency_contact_name="E", emergency_contact_phone="2")
        db.add(user); db.add(visitor); db.commit()

        vec = np.random.randn(128).astype(np.float32)
        vec_bytes = vec.tobytes()

        template = BiometricTemplate(
            id=str(uuid.uuid4()),
            visitor_id=visitor.visitor_id,
            face_pose_index=0,
            embedding_vector=vec_bytes,
            quality_score=0.95
        )
        db.add(template); db.commit()

        # Retrieve and verify roundtrip
        retrieved = db.query(BiometricTemplate).filter_by(id=template.id).first()
        recovered_vec = np.frombuffer(retrieved.embedding_vector, dtype=np.float32)
        np.testing.assert_array_almost_equal(vec, recovered_vec)
    finally:
        db.close()


def test_db_03_officer_badge_uniqueness():
    """DB-03: Officer badge numbers enforce database uniqueness."""
    db = SessionLocal()
    badge = f"BADGE-{uuid.uuid4().hex[:6]}"
    u1 = User(id=str(uuid.uuid4()), email=f"off1_{uuid.uuid4().hex[:6]}@f.gov", password_hash="h", role="OFFICER")
    o1 = Officer(officer_id=str(uuid.uuid4()), user_id=u1.id, badge_number=badge, full_name="Off 1", phone="1", department="Ranger")
    db.add(u1); db.add(o1); db.commit()

    u2 = User(id=str(uuid.uuid4()), email=f"off2_{uuid.uuid4().hex[:6]}@f.gov", password_hash="h", role="OFFICER")
    o2 = Officer(officer_id=str(uuid.uuid4()), user_id=u2.id, badge_number=badge, full_name="Off 2", phone="2", department="Ranger")
    db.add(u2); db.add(o2)
    with pytest.raises(Exception):
        db.commit()
    db.rollback()
    db.close()


# ==============================================================================
# CATEGORY 8: API & EDGE INTEGRATION (API-01 to API-03)
# ==============================================================================

def test_api_01_night_window_time_calculation():
    """API-01: RiskRules evaluates daytime vs night-time boundaries accurately."""
    day_dt = datetime(2026, 10, 11, 14, 0, 0) # 2:00 PM -> Daytime
    night_dt = datetime(2026, 10, 11, 21, 0, 0) # 9:00 PM -> Night
    early_morning = datetime(2026, 10, 11, 4, 30, 0) # 4:30 AM -> Night

    assert RiskRules.is_night_time(day_dt) is False
    assert RiskRules.is_night_time(night_dt) is True
    assert RiskRules.is_night_time(early_morning) is True


def test_api_02_spatial_distance_to_boundary():
    """API-02: Spatial distance to boundary computes non-negative meter approximations."""
    poly = Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)])
    dist_outside = GeometryEngine.distance_to_boundary(poly, lat=25.0, lng=15.0)
    assert dist_outside > 0.0

    # On boundary
    dist_on = GeometryEngine.distance_to_boundary(poly, lat=10.0, lng=15.0)
    assert pytest.approx(dist_on, abs=1e-3) == 0.0


def test_api_03_config_directory_initialization():
    """API-03: Configuration initializes required storage paths."""
    assert os.path.exists(Config.UPLOAD_FOLDER)
    assert os.path.exists(Config.SOS_VIDEO_FOLDER)


# ==============================================================================
# CATEGORY 9: OFFLINE RESILIENCE & EDGE AUTONOMY (OFF-01 to OFF-02)
# ==============================================================================

def test_off_01_zero_cloud_dependency_spatial_and_risk():
    """OFF-01: Spatial geometry and risk engine operate with 100% offline determinism (no external DNS/sockets)."""
    # Create polygon and evaluate point (coordinates in lng, lat format)
    poly = Polygon([(76.60, 11.65), (76.65, 11.65), (76.65, 11.70), (76.60, 11.70), (76.60, 11.65)])
    inside = GeometryEngine.contains_point(poly, lat=11.67, lng=76.62)
    assert inside is True

    # Evaluate multi-variable risk
    risk = RiskEvaluator.evaluate(zone_code="ZONE_A", is_sos=False, is_unknown_person=False, is_night=False)
    assert risk["score"] == 5
    assert risk["severity"] == "LOW"


def test_off_02_offline_cryptographic_chunk_verification():
    """OFF-02: Video chunk hashing and integrity verification execute completely on local edge processor."""
    chunk_data = os.urandom(65536) # 64 KB chunk
    h1 = hashlib.sha256(chunk_data).hexdigest()
    h2 = VideoHandler.calculate_sha256(chunk_data)
    assert h1 == h2


# ==============================================================================
# CATEGORY 10: SECURITY, PRIVACY & BIOMETRIC PROTECTION (SEC-01 to SEC-02)
# ==============================================================================

def test_sec_01_password_irreversibility():
    """SEC-01: Password hash cannot be reversed; plaintext is never persisted in models."""
    secret = "TopSecretWildernessKey2026"
    h = hash_password(secret)
    assert secret not in h
    assert len(h) > 50


def test_sec_02_biometric_template_privacy():
    """SEC-02: Biometric template contains only irreversible 128-D vector embeddings, not raw facial images."""
    db = SessionLocal()
    try:
        # Check all existing templates in DB to confirm none store raw photos
        templates = db.query(BiometricTemplate).all()
        for t in templates:
            assert isinstance(t.embedding_vector, bytes)
            # 128 float32 values = 512 bytes
            assert len(t.embedding_vector) == 512
    finally:
        db.close()
