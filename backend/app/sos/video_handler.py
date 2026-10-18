import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from backend.app.config import Config
from backend.app.database.models import SosEvent, SosVideoChunk

class VideoHandler:
    """
    Manages chunked emergency SOS video ingestion, SHA-256 checksum integrity verification,
    sandboxed disk storage, and database persistence.
    """

    @staticmethod
    def calculate_sha256(data_bytes: bytes) -> str:
        """Calculates standard SHA-256 hexadecimal hash string."""
        hasher = hashlib.sha256()
        hasher.update(data_bytes)
        return hasher.hexdigest()

    @classmethod
    def save_chunk(
        cls,
        db: Session,
        sos_id: str,
        sequence_number: int,
        file_bytes: bytes,
        provided_sha256: Optional[str] = None,
        duration_seconds: int = 30
    ) -> Tuple[bool, str, Optional[SosVideoChunk]]:
        """
        Saves a chunk of video for an active SOS incident.
        Verifies SHA-256 checksum if provided.
        Returns (success: bool, message: str, chunk_obj: Optional[SosVideoChunk])
        """
        # 1. Verify SOS event existence
        sos = db.query(SosEvent).filter(SosEvent.sos_id == sos_id).first()
        if not sos:
            return False, f"SOS Event {sos_id} not found", None

        # 2. SHA-256 Integrity Verification
        computed_hash = cls.calculate_sha256(file_bytes)
        if provided_sha256 and provided_sha256.lower() != computed_hash.lower():
            return False, f"Integrity check failed: checksum mismatch (expected {provided_sha256}, got {computed_hash})", None

        # 3. Create sandboxed directory
        sos_dir = Path(Config.SOS_VIDEO_FOLDER) / sos_id
        os.makedirs(sos_dir, exist_ok=True)

        # 4. Save video file to disk
        file_ext = "webm"
        filename = f"chunk_{sequence_number:04d}_{computed_hash[:8]}.{file_ext}"
        storage_path = sos_dir / filename

        with open(storage_path, "wb") as f:
            f.write(file_bytes)

        file_size = len(file_bytes)

        # 5. Persist record in database
        chunk = SosVideoChunk(
            sos_id=sos_id,
            sequence_number=sequence_number,
            storage_path=str(storage_path),
            file_hash_sha256=computed_hash,
            duration_seconds=duration_seconds,
            file_size_bytes=file_size,
            recorded_at=datetime.utcnow(),
            uploaded_at=datetime.utcnow()
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        return True, "Chunk uploaded and verified successfully", chunk
