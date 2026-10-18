import hashlib
import pytest
from backend.app.database.connection import SessionLocal, init_db
from backend.app.database.models import User, Visitor, SosEvent
from backend.app.sos.video_handler import VideoHandler

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_sha256_calculation():
    raw_bytes = b"TACTICAL_VIDEO_CHUNK_STREAM_DATA_001"
    expected = hashlib.sha256(raw_bytes).hexdigest()
    assert VideoHandler.calculate_sha256(raw_bytes) == expected

def test_chunk_integrity_verification():
    init_db()
    db = SessionLocal()
    try:
        # Create user & visitor
        user = User(id="test-u-sos", email="sos_tester@forest.gov", password_hash="dummy", role="VISITOR")
        visitor = Visitor(visitor_id="VST-TEST-SOS", user_id="test-u-sos", full_name="SOS Tester", phone="123", emergency_contact_name="Contact", emergency_contact_phone="456")
        sos = SosEvent(sos_id="SOS-TEST-99", visitor_id="VST-TEST-SOS", initial_lat=11.66, initial_lng=76.62, initial_zone="ZONE_A", status="TRIGGERED")
        db.merge(user)
        db.merge(visitor)
        db.merge(sos)
        db.commit()

        video_bytes = b"SAMPLE_H264_CHUNK_PAYLOAD_TEST"
        valid_hash = hashlib.sha256(video_bytes).hexdigest()

        # Valid chunk upload
        ok, msg, chunk = VideoHandler.save_chunk(
            db=db,
            sos_id="SOS-TEST-99",
            sequence_number=1,
            file_bytes=video_bytes,
            provided_sha256=valid_hash,
            duration_seconds=30
        )
        assert ok is True
        assert chunk is not None
        assert chunk.file_hash_sha256 == valid_hash

        # Corrupt / Tampered chunk upload
        bad_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        bad_ok, bad_msg, _ = VideoHandler.save_chunk(
            db=db,
            sos_id="SOS-TEST-99",
            sequence_number=2,
            file_bytes=video_bytes,
            provided_sha256=bad_hash
        )
        assert bad_ok is False
        assert "mismatch" in bad_msg.lower()

    finally:
        db.close()
