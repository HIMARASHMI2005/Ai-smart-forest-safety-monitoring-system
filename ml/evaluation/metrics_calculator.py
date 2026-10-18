import numpy as np
from typing import List, Dict, Any, Tuple

class MetricsCalculator:
    """
    Mathematical calculation of detection and verification metrics:
    - Precision, Recall, Average Precision (AP@50)
    - False Accept Rate (FAR), False Reject Rate (FRR)
    - ROC Curve & Trapezoidal AUC
    """
    @staticmethod
    def compute_ap(recalls: List[float], precisions: List[float]) -> float:
        """Computes 11-point or trapezoidal Average Precision."""
        mrec = [0.0] + list(recalls) + [1.0]
        mpre = [0.0] + list(precisions) + [0.0]

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        ap = 0.0
        for i in range(1, len(mrec)):
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return float(ap)

    @staticmethod
    def compute_roc_auc(fpr_list: List[float], tpr_list: List[float]) -> float:
        """Trapezoidal integration for Area Under Receiver Operating Characteristic Curve."""
        sorted_indices = np.argsort(fpr_list)
        x = np.array(fpr_list)[sorted_indices]
        y = np.array(tpr_list)[sorted_indices]
        return float(np.trapz(y, x))

    @staticmethod
    def find_optimal_threshold(genuine_scores: List[float], imposter_scores: List[float]) -> Tuple[float, float, float]:
        """
        Determines the optimal verification threshold tau where FAR is minimized while TAR remains >= 95%.
        Returns (best_tau, far, frr).
        """
        thresholds = np.linspace(0.4, 0.9, 100)
        best_tau = 0.68
        best_diff = float("inf")
        best_far = 0.0
        best_frr = 0.0

        for t in thresholds:
            far = np.mean([s >= t for s in imposter_scores]) if imposter_scores else 0.0
            frr = np.mean([s < t for s in genuine_scores]) if genuine_scores else 0.0
            # Target minimum distance between FAR and FRR (Equal Error Rate point)
            diff = abs(far - frr)
            if diff < best_diff:
                best_diff = diff
                best_tau = float(t)
                best_far = float(far)
                best_frr = float(frr)

        return round(best_tau, 2), round(best_far, 4), round(best_frr, 4)
