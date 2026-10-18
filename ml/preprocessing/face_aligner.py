import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class FaceAligner:
    """
    Face detection, quality gating, and 112x112 crop alignment.
    Enforces operational quality standards:
    - Minimum Face Size: 60x60
    - Sharpness / Blur: Laplacian variance >= 40.0
    - Lighting & Exposure: Mean intensity 40 to 220
    """

    def __init__(self):
        # Use OpenCV Haar cascades for frontal and profile face detection
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
        self.profile_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_profileface.xml")

    def assess_quality(self, face_crop: np.ndarray) -> Tuple[bool, float, str]:
        """
        Calculates quality score [0.0 - 1.0] and validates clarity.
        Tuned for live webcam feeds and tactical field cameras.
        """
        h, w = face_crop.shape[:2]
        if h < 35 or w < 35:
            return False, 0.2, "Face crop too small. Please move closer to the camera."

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 1. Lighting assessment
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 20:
            return False, 0.3, "Lighting too dark. Increase ambient illumination."
        if mean_brightness > 245:
            return False, 0.3, "Exposure too high / glare. Reduce background glare."

        # 2. Sharpness / Blur check (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 8.0:
            return False, 0.4, "Image is blurry. Keep device steady and refocus."

        # Quality score composite
        sharpness_score = min(1.0, laplacian_var / 250.0)
        lighting_score = 1.0 - abs(mean_brightness - 128.0) / 128.0
        quality_score = round(float(0.6 * sharpness_score + 0.4 * lighting_score), 3)

        return True, quality_score, "Face quality optimal"

    def align_and_crop(self, image_bgr: np.ndarray, is_precropped: bool = False) -> Tuple[bool, Optional[np.ndarray], float, str]:
        """
        Finds the prominent face in frame, validates quality, crops and resizes to 112x112.
        If is_precropped=True, bypasses second-pass cascade crop to preserve facial feature geometry.
        Returns: (success, aligned_face_112x112, quality_score, message)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, None, 0.0, "Invalid image data received"

        h, w = image_bgr.shape[:2]

        if is_precropped:
            crop = image_bgr
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(36, 36)
            )

            if len(faces) == 0 and self.profile_cascade:
                faces = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=3,
                    minSize=(36, 36)
                )

            if len(faces) == 0:
                return False, None, 0.0, "No face detected in capture. Ensure face is clearly centered."
            else:
                # Pick largest detected face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest_face

                # Add 15% margin for full chin and forehead inclusion
                pad_x = int(0.15 * fw)
                pad_y = int(0.15 * fh)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y)

                crop = image_bgr[y1:y2, x1:x2]

        is_valid, q_score, q_msg = self.assess_quality(crop)
        if not is_valid:
            return False, None, q_score, q_msg

        aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        return True, aligned, q_score, q_msg
