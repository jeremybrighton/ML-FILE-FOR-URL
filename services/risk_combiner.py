"""
FraudGuard — Risk Combiner v2 (Gated Submission)
=================================================
Combines ML output + legal/compliance rule matches into the final risk
decision. Controls WHICH transactions are auto-submitted to FRC.

Submission gates (evaluated in priority order):
  1. Mandatory legal threshold  (REG40 CTR >= USD 15,000 | REG10 >= USD 10,000)
       → case_status: mandatory_regulatory_submission
  2. KES critical amount        (>= KES 1,950,000 policy threshold)
       → case_status: critical_amount_submission
  3. ML suspicion score >= 25%  (ml_auto_submit_score)
       → case_status: suspicious_transaction_auto_submission
  4. Score 10%-25% or anomaly flags but below threshold
       → case_status: suspicious_below_submission_threshold  (NO submit)
  5. Clean / no anomaly
       → case_status: monitoring_only  (NO submit)

submission_classification labels:
  suspicious_transaction_auto_submission   — ML >=25%
  critical_amount_threshold_case           — amount >= KES 1,950,000
  mandatory_cash_transaction_report        — REG40 >= USD 15,000
  cross_border_declaration_case            — REG10 >= USD 10,000
  mandatory_regulatory_submission          — any other mandatory rule
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import List

from services.legal_rules_engine import (
    MatchedRule,
    should_auto_submit,
    highest_severity,
    primary_report_type,
    primary_frc_case_type,
    ML_AUTO_SUBMIT_SCORE,
    ML_HIGH_RISK_SCORE,
    _policy,
)

log = logging.getLogger(__name__)

ML_MONITORING_ONLY_SCORE: float = float(_policy.get("ml_monitoring_only_score", 0.10))

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_MONITORING_ONLY  = "monitoring_only"
STATUS_BELOW_THRESHOLD  = "suspicious_below_submission_threshold"
STATUS_REGULATORY       = "mandatory_regulatory_submission"
STATUS_CRITICAL_AMOUNT  = "critical_amount_submission"
STATUS_SUSPICIOUS_AUTO  = "suspicious_transaction_auto_submission"
STATUS_AUTO_SUBMITTED   = "auto_submitted_to_frc"
STATUS_FRC_ACKNOWLEDGED = "frc_acknowledged"

# ── Classification labels ─────────────────────────────────────────────────────
CLASSIFY_SUSPICIOUS      = "suspicious_transaction_auto_submission"
CLASSIFY_CRITICAL_AMOUNT = "critical_amount_threshold_case"
CLASSIFY_CASH_CTR        = "mandatory_cash_transaction_report"
CLASSIFY_CROSSBORDER     = "cross_border_declaration_case"
CLASSIFY_MANDATORY       = "mandatory_regulatory_submission"


@dataclass
class CombinedRiskDecision:
    ml_score:      float
    ml_prediction: int
    ml_risk_level: str

    matched_legal_rules:  List[MatchedRule] = field(default_factory=list)
    matched_policy_rules: List[MatchedRule] = field(default_factory=list)

    final_risk_level:          str  = "LOW"
    case_status:               str  = STATUS_MONITORING_ONLY
    report_type:               str  = "no_report_required"
    frc_case_type:             str  = "suspicious_activity_report"
    requires_auto_submit:      bool = False
    submission_mode:           str  = "none"
    submission_reason:         str  = ""
    submission_classification: str  = ""

    compliance_flags:  List[str]        = field(default_factory=list)
    all_matched_rules: List[MatchedRule] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matched_legal_rules"]  = [r.to_dict() for r in self.matched_legal_rules]
        d["matched_policy_rules"] = [r.to_dict() for r in self.matched_policy_rules]
        d["all_matched_rules"]    = [r.to_dict() for r in self.all_matched_rules]
        return d


def _ml_risk_level(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= ML_AUTO_SUBMIT_SCORE:
        return "MEDIUM"
    return "LOW"


def _final_risk(ml: str, rule_sev: str) -> str:
    order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    lvl = max(order.get(ml, 1), order.get(rule_sev, 0))
    return {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}.get(lvl, "MEDIUM")


def combine(
    ml_score: float,
    ml_prediction: int,
    matched_rules: List[MatchedRule],
) -> CombinedRiskDecision:
    """
    Apply submission gating and produce the final compliance decision.
    """
    ml_level  = _ml_risk_level(ml_score)
    ml_pct    = round(ml_score * 100, 1)

    legal_rules  = [r for r in matched_rules if r.is_legal_rule]
    policy_rules = [r for r in matched_rules if not r.is_legal_rule]
    all_rules    = matched_rules

    sev       = highest_severity(all_rules) if all_rules else ml_level
    rep_type  = primary_report_type(all_rules)
    frc_ctype = primary_frc_case_type(all_rules)
    final_lvl = _final_risk(ml_level, sev)

    flags:        List[str] = []
    case_status   = STATUS_MONITORING_ONLY
    requires_auto = False
    sub_mode      = "none"
    sub_reason    = ""
    sub_class     = ""

    # ── GATE 1: Mandatory legal threshold (REG40 / REG10) ─────────────────
    mandatory = [
        r for r in all_rules
        if r.rule_type in ("regulatory_threshold", "declaration_requirement") and r.auto_submit
    ]
    if mandatory:
        rule          = mandatory[0]
        case_status   = STATUS_REGULATORY
        requires_auto = True
        sub_mode      = "automatic"
        flags.append(f"mandatory_legal_{rule.rule_id}")
        if rule.rule_id == "REG40_CASH_CTR":
            sub_class  = CLASSIFY_CASH_CTR
            sub_reason = (
                f"Mandatory cash transaction report (POCAMLA Regulation 40): amount equals or "
                f"exceeds USD 15,000 equivalent. ML suspicion score: {ml_pct}%. "
                f"This report is mandatory regardless of whether the transaction appears suspicious."
            )
        elif rule.rule_id == "REG10_CROSSBORDER":
            sub_class  = CLASSIFY_CROSSBORDER
            sub_reason = (
                f"Cross-border declaration (POCAMLA Regulation 10 / Section 12): amount equals "
                f"or exceeds USD 10,000 equivalent. ML suspicion score: {ml_pct}%."
            )
        else:
            sub_class  = CLASSIFY_MANDATORY
            sub_reason = (
                f"Mandatory regulatory threshold met under {rule.section_or_regulation}. "
                f"ML suspicion score: {ml_pct}%."
            )

    # ── GATE 2: KES critical amount policy threshold ───────────────────────
    elif any(r.rule_id == "POLICY_KES_THRESHOLD" and r.auto_submit for r in all_rules):
        kes_rule      = next(r for r in all_rules if r.rule_id == "POLICY_KES_THRESHOLD")
        kes_amt       = kes_rule.meta.get("amount_kes", 0)
        case_status   = STATUS_CRITICAL_AMOUNT
        requires_auto = True
        sub_mode      = "automatic"
        sub_class     = CLASSIFY_CRITICAL_AMOUNT
        flags.append("critical_kes_1950000_threshold")
        sub_reason = (
            f"Critical amount threshold: KES {kes_amt:,.0f} equals or exceeds the "
            f"institutional critical threshold of KES 1,950,000. ML suspicion score: {ml_pct}%. "
            f"Submitted under institutional compliance policy and POCAMLA Section 44."
        )

    # ── GATE 3: ML suspicion score >= 25% ─────────────────────────────────
    elif ml_score >= ML_AUTO_SUBMIT_SCORE:
        case_status   = STATUS_SUSPICIOUS_AUTO
        requires_auto = True
        sub_mode      = "automatic"
        sub_class     = CLASSIFY_SUSPICIOUS
        flags.append(f"ml_{ml_pct}_pct_above_25pct_threshold")
        sub_reason = (
            f"Suspicious transaction: ML anomaly model assigned {ml_pct}% suspicion probability, "
            f"meeting the 25% institutional auto-submission threshold. "
            f"Classification: suspicious transaction — not a confirmed fraud determination. "
            f"Submitted for FRC assessment under POCAMLA Section 44 and Regulation 38."
        )
        for r in all_rules:
            if r.rule_id not in ("REG40_CASH_CTR", "REG10_CROSSBORDER", "POLICY_KES_THRESHOLD"):
                flags.append(f"also_{r.rule_id}")

    # ── GATE 4: Anomaly below submission threshold — internal only ─────────
    elif (
        ml_score >= ML_MONITORING_ONLY_SCORE
        or ml_prediction == 1
        or any(not r.auto_submit for r in all_rules)
    ):
        case_status   = STATUS_BELOW_THRESHOLD
        requires_auto = False
        sub_mode      = "monitoring_only"
        flags.append(f"ml_{ml_pct}_pct_below_25pct_not_submitted")
        sub_reason = (
            f"Anomaly detected (ML: {ml_pct}%) but below 25% auto-submission threshold. "
            f"Retained as internal monitoring alert. Manual review recommended."
        )

    # ── GATE 5: Clean transaction ──────────────────────────────────────────
    else:
        case_status   = STATUS_MONITORING_ONLY
        requires_auto = False
        sub_mode      = "none"
        sub_reason    = (
            f"No suspicious indicators. ML score: {ml_pct}%. Internal record only."
        )

    if rep_type == "no_report_required" and (legal_rules or policy_rules):
        rep_type = "suspicious_transaction_report"

    log.info(
        f"[RiskCombiner] ml={ml_pct}% rules={len(all_rules)} "
        f"risk={final_lvl} status={case_status} submit={requires_auto} class={sub_class or 'none'}"
    )

    return CombinedRiskDecision(
        ml_score=ml_score,
        ml_prediction=ml_prediction,
        ml_risk_level=ml_level,
        matched_legal_rules=legal_rules,
        matched_policy_rules=policy_rules,
        final_risk_level=final_lvl,
        case_status=case_status,
        report_type=rep_type,
        frc_case_type=frc_ctype,
        requires_auto_submit=requires_auto,
        submission_mode=sub_mode,
        submission_reason=sub_reason,
        submission_classification=sub_class,
        compliance_flags=flags,
        all_matched_rules=all_rules,
    )
