import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.app.risk_engine.rules import RiskRules

class RiskEvaluator:
    """
    Deterministic Multi-Variable Risk Evaluation Engine.
    Guarantees transparent, zero-hallucination attribution tracking.
    Formula:
    R = min(100, w_SOS * I_SOS + S_zone(Z) + w_unknown * I_unknown +
                 w_crowd * I_crowd + w_night * I_night + w_sec * I_sec)
    """

    @classmethod
    def evaluate(
        cls,
        zone_code: str,
        is_sos: bool = False,
        is_unknown_person: bool = False,
        is_crowd: bool = False,
        is_night: Optional[bool] = None,
        is_prolonged_dwell: bool = False,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Executes deterministic multi-factor risk inference.
        Returns:
            score: int (0 - 100)
            severity: str ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
            reasons: list of explainable factor strings
            breakdown: dict of factor names and numeric points
        """
        dt = timestamp or datetime.now()
        night_active = RiskRules.is_night_time(dt) if is_night is None else is_night

        score = 0
        reasons: List[str] = []
        breakdown: Dict[str, int] = {}

        # 1. SOS Emergency Factor (Immediate override to 100)
        if is_sos:
            score = 100
            reasons.append("SOS_EMERGENCY_TRIGGERED (+100)")
            breakdown["sos"] = 100
            return {
                "score": 100,
                "severity": "CRITICAL",
                "reasons": reasons,
                "breakdown": breakdown,
                "timestamp": dt.isoformat()
            }

        # 2. Zone Containment Threat
        zone_threat = RiskRules.ZONE_SCORES.get(zone_code, RiskRules.ZONE_SCORES["OUT_OF_BOUNDS"])
        score += zone_threat
        reasons.append(f"{zone_code}_TERRAIN_BASE (+{zone_threat})")
        breakdown["zone"] = zone_threat

        # 3. Biometric Identity Threat (Unknown / Unregistered)
        if is_unknown_person:
            score += RiskRules.WEIGHT_UNKNOWN_PERSON
            reasons.append(f"UNKNOWN_UNREGISTERED_IDENTITY (+{RiskRules.WEIGHT_UNKNOWN_PERSON})")
            breakdown["unknown_person"] = RiskRules.WEIGHT_UNKNOWN_PERSON

        # 4. Crowd Clustering Threat (>3 persons in monitored/restricted sectors)
        if is_crowd:
            score += RiskRules.WEIGHT_CROWD_CLUSTER
            reasons.append(f"UNAUTHORIZED_CROWD_CLUSTER (+{RiskRules.WEIGHT_CROWD_CLUSTER})")
            breakdown["crowd"] = RiskRules.WEIGHT_CROWD_CLUSTER

        # 5. Night Window Condition (18:30 - 06:00)
        if night_active:
            score += RiskRules.WEIGHT_NIGHT_CONDITION
            reasons.append(f"NIGHT_HOURS_HAZARD (+{RiskRules.WEIGHT_NIGHT_CONDITION})")
            breakdown["night"] = RiskRules.WEIGHT_NIGHT_CONDITION

        # 6. Prolonged Dwell Time (>15 mins stationary / off-trail)
        if is_prolonged_dwell:
            score += RiskRules.WEIGHT_PROLONGED_DWELL
            reasons.append(f"PROLONGED_OFF_TRAIL_DWELL (+{RiskRules.WEIGHT_PROLONGED_DWELL})")
            breakdown["prolonged_dwell"] = RiskRules.WEIGHT_PROLONGED_DWELL

        # Clamp score to maximum 100
        total_score = min(100, max(0, score))
        severity = RiskRules.categorize_severity(total_score)

        return {
            "score": total_score,
            "severity": severity,
            "reasons": reasons,
            "breakdown": breakdown,
            "timestamp": dt.isoformat()
        }
