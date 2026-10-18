import pytest
from backend.app.risk_engine.rules import RiskRules
from backend.app.risk_engine.evaluator import RiskEvaluator

def test_risk_severity_categorization():
    assert RiskRules.categorize_severity(10) == "LOW"
    assert RiskRules.categorize_severity(24) == "LOW"
    assert RiskRules.categorize_severity(25) == "MEDIUM"
    assert RiskRules.categorize_severity(49) == "MEDIUM"
    assert RiskRules.categorize_severity(50) == "HIGH"
    assert RiskRules.categorize_severity(74) == "HIGH"
    assert RiskRules.categorize_severity(75) == "CRITICAL"
    assert RiskRules.categorize_severity(100) == "CRITICAL"

def test_sos_override():
    # SOS overrides any zone to 100 CRITICAL
    res = RiskEvaluator.evaluate(zone_code="ZONE_A", is_sos=True)
    assert res["score"] == 100
    assert res["severity"] == "CRITICAL"
    assert "SOS_EMERGENCY_TRIGGERED (+100)" in res["reasons"]

def test_multi_variable_scoring():
    # Zone C (+55) + Night (+15) -> 70 -> HIGH
    res = RiskEvaluator.evaluate(
        zone_code="ZONE_C",
        is_sos=False,
        is_unknown_person=False,
        is_crowd=False,
        is_night=True,
        is_prolonged_dwell=False
    )
    assert res["score"] == 70
    assert res["severity"] == "HIGH"
    assert any("ZONE_C" in r for r in res["reasons"])
    assert any("NIGHT" in r for r in res["reasons"])

def test_score_clamping_at_100():
    # Zone D (+80) + Unknown (+30) + Crowd (+25) -> sum = 135 -> clamped to 100
    res = RiskEvaluator.evaluate(
        zone_code="ZONE_D",
        is_sos=False,
        is_unknown_person=True,
        is_crowd=True,
        is_night=False
    )
    assert res["score"] == 100
    assert res["severity"] == "CRITICAL"
