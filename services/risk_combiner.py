"""
FraudGuard — Risk Combiner
============================
Combines the ML model output with legal/compliance rule matches to produce
the final risk decision for each transaction.

Output states (defined in POCAMLA-aligned terminology):
  normal                        — no anomaly, no rule triggered
  anomaly_detected              — ML flags but no legal rule (below auto-submit threshold)
  regulatory_report_required    — a legal mandatory-reporting rule fired (e.g. REG40 CTR)
  suspicious_activity_report_candidate — STR candidate, needs analyst validation
  critical_compliance_trigger   — KES policy threshold OR CRITICAL severity rule fired
  auto_submitted_to_frc         — auto-submit condition met and submission initiated
  frc_acknowledged              — FRC returned acknowledgement (set by submission service)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from services.legal_rules_engine import (
    MatchedRule,
    should_auto_submit,
    highest_severity,
    primary_report_type,
    primary_frc_case_type,
    ML_AUTO_SUBMIT_SCORE,
    ML_HIGH_RISK_SCORE,
)

log = logging.getLogger(__name__)

# ── Case status constants ─────────────────────────────────────────────────────
STATUS_NORMAL              = "normal"
STATUS_ANOMALY             = "anomaly_detected"
STATUS_REGULATORY          = "regulatory_report_required"
STATUS_STR_CANDIDATE       = "suspicious_activity_report_candidate"
STATUS_CRITICAL_TRIGGER    = "critical_compliance_trigger"
STATUS_AUTO_SUBMITTED      = "auto_submitted_to_frc"
STATUS_FRC_ACKNOWLEDGED    = "frc_acknowledged"


@dataclass
class CombinedRiskDecision:
    # Raw inputs
    ml_score: float                         # 0.0–1.0
    ml_prediction: int                      # 0 or 1
    ml_risk_level: str                      # LOW | MEDIUM | HIGH

    # Compliance output
    matched_legal_rules: List[MatchedRule]  = field(default_factory=list)
    matched_policy_rules: List[MatchedRule] = field(default_factory=list)

    # Combined decision
    final_risk_level: str  = "LOW"          # LOW | MEDIUM | HIGH | CRITICAL
    case_status: str       = STATUS_NORMAL
    report_type: str       = "no_report_required"
    frc_case_type: str     = "suspicious_activity_report"
    requires_auto_submit: bool = False
    submission_mode: str   = "none"         # none | automatic | manual_review_required

    # Explanation fields
    compliance_flags: List[str] = field(default_factory=list)
    all_matched_rules: List[MatchedRule] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert MatchedRule lists to plain dicts
        d["matched_legal_rules"]  = [r.to_dict() for r in self.matched_legal_rules]
        d["matched_policy_rules"] = [r.to_dict() for r in self.matched_policy_rules]
        d["all_matched_rules"]    = [r.to_dict() for r in self.all_matched_rules]
        return d


def _ml_risk_level(score: float) -> str:
    """Convert ML probability to risk label."""
    if score >= 0.80:
        return "HIGH"
    if score >= ML_HIGH_RISK_SCORE:
        return "MEDIUM"
    return "LOW"


def combine(
    ml_score: float,
    ml_prediction: int,
    matched_rules: List[MatchedRule],
) -> CombinedRiskDecision:
    """
    Combine ML output and legal rule matches into a final risk decision.

    Logic (applied in priority order):
    1. CRITICAL — any CRITICAL rule fired  → critical_compliance_trigger
    2. REGULATORY — mandatory reporting rule (auto_submit=True, frc_required)
       with severity HIGH/CRITICAL         → regulatory_report_required
    3. ML HIGH auto-submit threshold       → suspicious_activity_report_candidate
    4. ML MEDIUM + legal rules             → anomaly_detected or STR candidate
    5. No rules, no ML flag               → normal
    """
    ml_level = _ml_risk_level(ml_score)
    ml_pct   = round(ml_score * 100, 1)

    # Split rules into legal vs policy
    legal_rules  = [r for r in matched_rules if r.is_legal_rule]
    policy_rules = [r for r in matched_rules if not r.is_legal_rule]

    all_rules     = matched_rules
    severity      = highest_severity(all_rules) if all_rules else ml_level
    auto_submit   = should_auto_submit(all_rules)
    rep_type      = primary_report_type(all_rules)
    frc_ctype     = primary_frc_case_type(all_rules)

    compliance_flags: List[str] = []

    # ── Determine final risk level ────────────────────────────────────────
    # Elevate ML risk using legal severity
    severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    ml_order = severity_order.get(ml_level, 1)
    rule_order = severity_order.get(severity, 1) if all_rules else 0
    combined_order = max(ml_order, rule_order)
    final_level = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}.get(combined_order, "MEDIUM")

    # ── Determine case status ─────────────────────────────────────────────
    if any(r.severity == "CRITICAL" for r in all_rules):
        case_status = STATUS_CRITICAL_TRIGGER
        compliance_flags.append("critical_legal_or_policy_rule_triggered")
    elif any(r.rule_type == "regulatory_threshold" and r.auto_submit for r in all_rules):
        case_status = STATUS_REGULATORY
        compliance_flags.append("mandatory_regulatory_report_triggered")
    elif any(r.rule_type == "declaration_requirement" and r.auto_submit for r in all_rules):
        case_status = STATUS_REGULATORY
        compliance_flags.append("cross_border_declaration_triggered")
    elif auto_submit:
        case_status = STATUS_CRITICAL_TRIGGER
        compliance_flags.append("auto_submit_rule_triggered")
    elif ml_score >= ML_AUTO_SUBMIT_SCORE:
        case_status = STATUS_STR_CANDIDATE
        compliance_flags.append(f"ml_score_{ml_pct}_pct_exceeds_auto_submit_threshold")
    elif ml_prediction == 1 or ml_score >= ML_HIGH_RISK_SCORE or legal_rules or policy_rules:
        case_status = STATUS_ANOMALY
        compliance_flags.append("ml_anomaly_or_rule_match")
    else:
        case_status = STATUS_NORMAL

    # ── Determine submission mode ─────────────────────────────────────────
    if auto_submit or (ml_score >= ML_AUTO_SUBMIT_SCORE and case_status in
                       (STATUS_STR_CANDIDATE, STATUS_CRITICAL_TRIGGER, STATUS_REGULATORY)):
        requires_auto = True
        sub_mode = "automatic"
        compliance_flags.append("auto_submission_required")
    elif legal_rules or policy_rules:
        requires_auto = False
        sub_mode = "manual_review_required"
    else:
        requires_auto = False
        sub_mode = "none"

    # If rep_type is still "no_report_required" but we have rules, set a sensible default
    if rep_type == "no_report_required" and (legal_rules or policy_rules):
        rep_type = "suspicious_transaction_report"

    log.info(
        f"[RiskCombiner] ml={ml_pct}% | rules={len(all_rules)} | "
        f"final_risk={final_level} | status={case_status} | "
        f"auto_submit={requires_auto}"
    )

    return CombinedRiskDecision(
        ml_score=ml_score,
        ml_prediction=ml_prediction,
        ml_risk_level=ml_level,
        matched_legal_rules=legal_rules,
        matched_policy_rules=policy_rules,
        final_risk_level=final_level,
        case_status=case_status,
        report_type=rep_type,
        frc_case_type=frc_ctype,
        requires_auto_submit=requires_auto,
        submission_mode=sub_mode,
        compliance_flags=compliance_flags,
        all_matched_rules=all_rules,
    )
