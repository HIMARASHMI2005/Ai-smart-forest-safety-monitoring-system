import os
import time
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.config import Config
from backend.app.database.models import BiometricTemplate, Visitor
from ml.forest_face_net.verify import FaceVerifier, compute_cosine_similarity
from ml.preprocessing.face_aligner import FaceAligner

class FaceService:
    """Biometric processing, quality gating, and 128-D embedding service."""
    _verifier_instance = None
    _aligner_instance = None
    _candidate_cache = None
    _candidate_cache_time = 0.0

    @classmethod
    def get_verifier(cls) -> FaceVerifier:
        if cls._verifier_instance is None:
            ckpt = str(Config.FACE_RECOGNITION_CHECKPOINT)
            if not os.path.exists(ckpt):
                ckpt = None
            cls._verifier_instance = FaceVerifier(
                checkpoint_path=ckpt,
                threshold=Config.FACE_VERIFY_THRESHOLD
            )
        return cls._verifier_instance

    @classmethod
    def get_aligner(cls) -> FaceAligner:
        if cls._aligner_instance is None:
            cls._aligner_instance = FaceAligner()
        return cls._aligner_instance

    @classmethod
    def process_and_extract_embedding(cls, image_bgr: np.ndarray, is_precropped: bool = False) -> Tuple[bool, Optional[np.ndarray], float, str]:
        """
        Runs quality gate, aligns 112x112 face crop, extracts 128-D L2-normalized vector.
        Returns (success: bool, embedding_128: Optional[np.ndarray], quality_score: float, msg: str)
        """
        aligner = cls.get_aligner()
        success, crop_112, q_score, msg = aligner.align_and_crop(image_bgr, is_precropped=is_precropped)
        if not success or crop_112 is None:
            return False, None, q_score, msg

        verifier = cls.get_verifier()
        emb_128 = verifier.extract_embedding(crop_112)
        return True, emb_128, q_score, "Face processed and 128-D embedding extracted successfully"

    @classmethod
    def verify_visitor_face(
        cls, db: Session, visitor_id: str, query_emb: np.ndarray
    ) -> Tuple[bool, float, str]:
        """
        1:1 Face Verification:
        Compares query_emb against all registered pose templates for visitor_id.
        Accepts if match confidence passes threshold.
        """
        if query_emb is None:
            return False, 0.0, "Invalid query face embedding"

        if db is None:
            return False, 0.0, "Database session unavailable"

        try:
            templates = db.query(BiometricTemplate).filter(BiometricTemplate.visitor_id == visitor_id).all()
        except Exception as e:
            return False, 0.0, f"Database query failed: {e}"

        if not templates:
            return False, 0.0, f"No biometric templates found for visitor {visitor_id}"

        verifier = cls.get_verifier()
        best_match = False
        best_conf = 0.0

        for t in templates:
            template_vec = np.frombuffer(t.embedding_vector, dtype=np.float32)
            is_match, conf = verifier.verify_1_to_1(query_emb, template_vec)
            if conf > best_conf:
                best_conf = conf
            if is_match:
                best_match = True

        status_msg = f"Match verified (Confidence: {best_conf:.4f} >= {verifier.threshold})" if best_match \
            else f"Verification failed (Confidence: {best_conf:.4f} < {verifier.threshold})"

        return best_match, round(best_conf, 4), status_msg

    @classmethod
    def get_candidate_templates(cls, db: Session, max_age: float = 3.0) -> List[Tuple[str, np.ndarray]]:
        now = time.time()
        if cls._candidate_cache is not None and (now - cls._candidate_cache_time) < max_age:
            return cls._candidate_cache
        try:
            templates = db.query(BiometricTemplate).all()
            candidate_list = [
                (t.visitor_id, np.frombuffer(t.embedding_vector, dtype=np.float32))
                for t in templates
            ]
            cls._candidate_cache = candidate_list
            cls._candidate_cache_time = now
            return candidate_list
        except Exception:
            return cls._candidate_cache or []

    @classmethod
    def identify_face(
        cls, db: Session, query_emb: np.ndarray
    ) -> Tuple[Optional[str], float, str]:
        """
        1:N Biometric Identification:
        Compares query_emb across all enrolled templates in database.
        Returns (matched_visitor_id or None, best_confidence, status_msg).
        """
        if query_emb is None:
            return None, 0.0, "Invalid or missing query face embedding"

        if db is None:
            return None, 0.0, "Database session unavailable"

        candidate_list = cls.get_candidate_templates(db)
        if not candidate_list:
            return None, 0.0, "No enrolled templates in database"

        verifier = cls.get_verifier()

        matched_id, best_conf = verifier.search_1_to_n(query_emb, candidate_list)

        if matched_id:
            return matched_id, round(float(best_conf), 4), f"Identified as {matched_id} (Confidence: {best_conf:.4f})"
        else:
            return None, 0.0, f"Unrecognized / UNKNOWN identity (Match confidence: {best_conf:.4f} < {verifier.threshold})"
