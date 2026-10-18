from datetime import datetime, time
from typing import Dict, Any

class RiskRules:
    """
    Deterministic weights and parameters for multi-factor risk inference.
    """
    WEIGHT_SOS = 100
    WEIGHT_UNKNOWN_PERSON = 30
    WEIGHT_CROWD_CLUSTER = 25
    WEIGHT_NIGHT_CONDITION = 15
    WEIGHT_PROLONGED_DWELL = 15

    ZONE_SCORES = {
        "ZONE_A": 5,           # Eco Trail (Low threat)
        "ZONE_B": 25,          # Monitored Buffer (Caution)
        "ZONE_C": 55,          # Restricted Corridor (High threat)
        "ZONE_D": 80,          # Critical Wildlife Sanctuary (Critical hazard)
        "OUT_OF_BOUNDS": 40    # Uncharted / Boundary breach
    }

    NIGHT_START = time(18, 30)  # 18:30 (6:30 PM)
    NIGHT_END = time(6, 0)      # 06:00 (6:00 AM)

    @classmethod
    def is_night_time(cls, check_dt: datetime = None) -> bool:
        """Determines if the given timestamp falls into the night window [18:30 - 06:00]."""
        dt = check_dt or datetime.now()
        current_time = dt.time()
        if cls.NIGHT_START <= current_time or current_time <= cls.NIGHT_END:
            return True
        return False

    @staticmethod
    def categorize_severity(score: int) -> str:
        """
        Deterministic Risk Categorization:
        0 - 24:  LOW / SAFE
        25 - 49: MEDIUM / CAUTION
        50 - 74: HIGH / DANGER
        75 - 100: CRITICAL / EMERGENCY
        """
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "LOW"
