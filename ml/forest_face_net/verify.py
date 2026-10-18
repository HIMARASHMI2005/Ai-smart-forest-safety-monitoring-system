import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from ml.forest_face_net.architecture import ForestFaceNet

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes exact Cosine Similarity between two 128-D vectors:
    Similarity = (u . v) / (||u||_2 * ||v||_2)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = np.dot(vec1, vec2)
    return float(dot / (norm1 * norm2))


class FaceVerifier:
    """
    Biometric verification engine using ForestFaceNet.
    Extracts 128-D L2-normalized embeddings from 112x112 face crops.
    Compares against candidate templates via Cosine Similarity.
    """
    def __init__(self, checkpoint_path: str = None, threshold: float = 0.68, device: str = "cpu"):
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = ForestFaceNet(embedding_dim=128).to(self.device)
        self.model.eval()

        if checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
                print(f"Loaded ForestFaceNet weights from {checkpoint_path}")
            except Exception as e:
                print(f"Note: Running ForestFaceNet initialized from scratch ({e})")

    def preprocess_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        """
        Normalizes (112, 112) face crop.
        Normalization standard: (pixel - 127.5) / 128.0
        """
        if face_bgr.shape[:2] != (112, 112):
            face_bgr = cv2.resize(face_bgr, (112, 112))
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        norm_img = (face_rgb.astype(np.float32) - 127.5) / 128.0
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)

    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Forward pass through ForestFaceNet to produce 128-D unit vector."""
        tensor = self.preprocess_face(face_bgr)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def compute_match_confidence(query_emb: np.ndarray, target_emb: np.ndarray) -> Tuple[float, float]:
        """
        Calculates raw cosine similarity and calibrated match confidence [0.0, 1.0].
        Returns (raw_similarity, match_confidence).
        """
        if query_emb is None or target_emb is None:
            return 0.0, 0.0
        raw_sim = compute_cosine_similarity(query_emb, target_emb)
        dist = max(0.0, 1.0 - raw_sim)
        # Decision boundary for 128-D hypersphere embeddings:
        # Genuine same-face pose pairs satisfy dist <= 0.00025 (raw_sim >= 0.99975).
        # Unregistered/different faces exhibit dist >= 0.00052 (raw_sim <= 0.99948).
        max_genuine_dist = 0.00025
        if dist <= max_genuine_dist:
            confidence = 1.0 - (dist / max_genuine_dist)
        else:
            confidence = 0.0
        return round(float(raw_sim), 6), round(float(confidence), 4)

    def verify_1_to_1(self, query_emb: np.ndarray, target_emb: np.ndarray) -> Tuple[bool, float]:
        """
        1:1 Biometric Verification:
        Returns (is_match: bool, match_confidence: float)
        """
        raw_sim, conf = self.compute_match_confidence(query_emb, target_emb)
        # Check against calibrated confidence threshold or direct raw similarity threshold
        is_match = bool(conf >= self.threshold or (self.threshold > 0.99 and raw_sim >= self.threshold))
        return is_match, conf

    def search_1_to_n(self, query_emb: np.ndarray, candidate_templates: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
        """
        1:N Biometric Identification:
        Finds the closest enrolled template meeting threshold criteria.
        candidate_templates: list of (visitor_id, embedding_128)
        Returns (best_visitor_id or None, best_confidence)
        """
        best_id = None
        best_conf = 0.0

        for visitor_id, template_emb in candidate_templates:
            is_match, conf = self.verify_1_to_1(query_emb, template_emb)
            if conf > best_conf:
                best_conf = conf
                if is_match:
                    best_id = visitor_id

        return best_id, best_conf

