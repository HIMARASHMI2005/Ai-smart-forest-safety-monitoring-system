import cv2
import numpy as np

class WildlandAugmentations:
    """
    Carefully calibrated data augmentations for wildland search-and-rescue.
    Prohibited: Extreme solarization, rainbow distortions, synthetic hallucinations.
    """
    @staticmethod
    def random_horizontal_flip(image: np.ndarray, boxes: np.ndarray = None, p: float = 0.5):
        if np.random.rand() > p:
            return image, boxes

        flipped_img = cv2.flip(image, 1)
        if boxes is None or len(boxes) == 0:
            return flipped_img, boxes

        w = image.shape[1]
        new_boxes = boxes.copy()
        # [x1, y1, x2, y2]
        new_boxes[:, 0] = w - boxes[:, 2]
        new_boxes[:, 2] = w - boxes[:, 0]
        return flipped_img, new_boxes

    @staticmethod
    def brightness_jitter(image: np.ndarray, delta: float = 0.12):
        """Bounded brightness variation (+/- 12%) preserving natural forest lighting."""
        factor = 1.0 + np.random.uniform(-delta, delta)
        table = np.array([min(255, max(0, int(i * factor))) for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def slight_rotation(image: np.ndarray, max_angle: float = 10.0):
        """Slight rotation (+/- 10 degrees) for pose variation."""
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
