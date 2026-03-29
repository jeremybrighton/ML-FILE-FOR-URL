"""
FraudGuard — Report Generator
================================
Builds the structured compliance report and FRC payload from the
CombinedRiskDecision and raw transaction data.

Report types supported:
  1. cash_transaction_report                (REG40 CTR)
  2. suspicious_transaction_report          (Section 44 / REG38 STR)
  3. cross_border_declaration_alert         (REG10 cross-border)
  4. critical_policy_threshold_escalation   (KES 1,950,000 policy rule)
  5. no_report_required                     (clean transaction)
"""
from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.risk_combiner import CombinedRiskDecision, STATUS_AUTO_SUBMITTED
from services.legal_rules_engine import INSTITUTION_CODE, INSTITUTION_NAME
from utils.currency_converter import to_usd, to_kes, infer_currency

log = logging.getLogger(__name__)

# ── Report type labels ────────────────────────────────────────────────────────
REPORT_LABELS = {
    "cash_transaction_report":              "Cash Transaction Report (CTR)",
    "suspicious_transaction_report":        "Suspicious Transaction Report (STR)",
    "cross_border_declaration_alert":       "Cross-Border Declaration Alert",
    "critical_policy_threshold_escalation": "Critical Policy Threshold Escalation",
    "no_report_required":                   "No Regulatory Report Required",
}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _mask_account(account_id: str) -> str:
    """Mask an account identifier for the report payload."""
    s = str(account_id or "")
    if len(s) <= 4:
        return "****"
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def _build_legal_reason(decision: CombinedRiskDecision) -> str:
    """Build the combined legal reason text from all matched rules."""
    reasons = []
    for r in decision.all_matched_rules:
        reasons.append(
            f"[{r.section_or_regulation} — {r.act_name}]: {r.legal_reason}"
        )
    if not reasons:
        ml_pct = round(decision.ml_score * 100, 1)
        return (
            f"This transaction was flagged by the FraudGuard ML anomaly detection model "
            f"with a fraud probability of {ml_pct}%. No mandatory legal reporting threshold "
            f"was met at this time. This record is retained for compliance audit purposes."
        )
    return " | ".join(reasons)


def _build_evidence_summary(txn: dict, decision: CombinedRiskDecision) -> List[str]:
    """Build a plain-text evidence list for the FRC payload."""
    evidence = []
    ml_pct = round(decision.ml_score * 100, 1)
    evidence.append(f"ML fraud model probability: {ml_pct}%")
    evidence.append(f"ML prediction: {'FLAGGED' if decision.ml_prediction else 'CLEAN'}")
    evidence.append(f"ML risk level: {decision.ml_risk_level}")

    amount = _safe_float(txn.get("amount"))
    currency = infer_currency(txn)
    evidence.append(f"Transaction amount: {currency} {amount:,.2f}")

    tx_type = str(txn.get("type", "UNKNOWN")).upper()
    evidence.append(f"Transaction type: {tx_type}")

    old_org = _safe_float(txn.get("oldbalanceOrg"))
    new_org = _safe_float(txn.get("newbalanceOrig"))
    if old_org > 0 and new_org == 0:
        evidence.append("Source account balance fully depleted to zero after transaction")

    old_dest = _safe_float(txn.get("oldbalanceDest"))
    new_dest = _safe_float(txn.get("newbalanceDest"))
    if old_dest == 0 and new_dest > 0:
        evidence.append("Destination account had zero prior balance before receiving funds")

    channel = txn.get("channel")
    if channel:
        evidence.append(f"Channel: {channel}")

    region = txn.get("region")
    if region:
        evidence.append(f"Region: {region}")

    for flag in decision.compliance_flags:
        evidence.append(f"Compliance flag: {flag}")

    for rule in decision.all_matched_rules:
        evidence.append(
            f"Legal rule matched: {rule.rule_name} "
            f"({rule.section_or_regulation}) — severity {rule.severity}"
        )

    return evidence


def generate_compliance_output(txn: dict, decision: CombinedRiskDecision) -> dict:
    """
    Build the full compliance output dict for a single transaction.
    This is attached to the transaction record in MongoDB and returned
    in API responses.
    """
    currency = infer_currency(txn)
    amount   = _safe_float(txn.get("amount"))
    amount_usd = to_usd(amount, currency)
    amount_kes = to_kes(amount, currency)
    ml_pct   = round(decision.ml_score * 100, 1)
    legal_reason = _build_legal_reason(decision)

    return {
        "ml_score":              decision.ml_score,
        "ml_score_pct":          ml_pct,
        "ml_prediction":         decision.ml_prediction,
        "ml_risk_level":         decision.ml_risk_level,
        "matched_legal_rules": [
            {
                "rule_id":              r.rule_id,
                "act_name":             r.act_name,
                "section_or_regulation": r.section_or_regulation,
                "rule_name":            r.rule_name,
                "severity":             r.severity,
                "reason":               r.legal_reason,
                "report_type":          r.report_type,
                "auto_submit":          r.auto_submit,
            }
            for r in decision.matched_legal_rules
        ],
        "matched_policy_rules": [
            {
                "rule_id":   r.rule_id,
                "rule_name": r.rule_name,
                "severity":  r.severity,
                "reason":    r.legal_reason,
                "report_type": r.report_type,
                "auto_submit": r.auto_submit,
            }
            for r in decision.matched_policy_rules
        ],
        "final_risk_level":  decision.final_risk_level,
        "case_status":       decision.case_status,
        "report_type":       decision.report_type,
        "report_type_label": REPORT_LABELS.get(decision.report_type, decision.report_type),
        "frc_case_type":     decision.frc_case_type,
        "legal_reason":      legal_reason,
        "submission_mode":   decision.submission_mode,
        "frc_submission_status": "pending" if decision.requires_auto_submit else "not_required",
        "compliance_flags":  decision.compliance_flags,
        "currency":          currency,
        "amount_usd_equivalent": round(amount_usd, 2),
        "amount_kes_equivalent": round(amount_kes, 2),
    }


def build_frc_payload(
    txn: dict,
    decision: CombinedRiskDecision,
    compliance_output: dict,
    transaction_id: str,
    internal_case_id: Optional[str] = None,
) -> dict:
    """
    Build the structured FRC intake payload.
    This is posted to POST /api/v1/intake/cases on the FRC backend.
    """
    now = datetime.now(timezone.utc).isoformat()
    currency = infer_currency(txn)
    amount   = _safe_float(txn.get("amount"))
    amount_usd = to_usd(amount, currency)
    amount_kes = to_kes(amount, currency)
    ml_pct   = round(decision.ml_score * 100, 1)

    # Build triggering rules list for FRC (from all matched rules)
    triggering_rules: List[str] = []
    for r in decision.all_matched_rules:
        triggering_rules.extend(r.triggering_rules_frc)
    triggering_rules = list(dict.fromkeys(triggering_rules))  # deduplicate, preserve order

    legal_reason = _build_legal_reason(decision)
    evidence_summary = _build_evidence_summary(txn, decision)

    # Map our report_type to FRC's accepted report_type values
    frc_report_type_map = {
        "cash_transaction_report":              "regulatory_threshold_report",
        "critical_policy_threshold_escalation": "suspicious_activity_report",
        "cross_border_declaration_alert":       "suspicious_activity_report",
        "suspicious_transaction_report":        "suspicious_activity_report",
        "no_report_required":                   "suspicious_activity_report",
    }
    frc_report_type = frc_report_type_map.get(decision.report_type, "suspicious_activity_report")

    # Source-of-funds flags
    old_bal_org = _safe_float(txn.get("oldbalanceOrg"))
    new_bal_org = _safe_float(txn.get("newbalanceOrig"))
    old_bal_dest = _safe_float(txn.get("oldbalanceDest"))
    sof_flags = []
    if old_bal_org > 0 and new_bal_org == 0:
        sof_flags.append("full_source_balance_depletion")
    if old_bal_dest == 0:
        sof_flags.append("zero_balance_destination_account")
    if decision.ml_score >= 0.50:
        sof_flags.append("ml_anomaly_detected")

    is_cross_border = any(
        r.rule_id == "REG10_CROSSBORDER" for r in decision.all_matched_rules
    )

    return {
        # ── FRC IntakeRequest fields ───────────────────────────────────────
        "external_report_id":  internal_case_id or transaction_id,
        "report_type":         frc_report_type,
        "amount":              round(amount, 2),
        "currency":            currency,
        "transaction_summary": (
            f"FraudGuard auto-submission: {REPORT_LABELS.get(decision.report_type, decision.report_type)}. "
            f"ML score: {ml_pct}%. Status: {decision.case_status}. "
            f"Rules matched: {len(decision.all_matched_rules)}."
        )[:2000],
        "triggering_rules":    triggering_rules or ["POCAMLA-S44-STR-GENERAL"],
        "risk_score":          round(decision.ml_score, 4),
        "narrative":           legal_reason[:5000],
        "timestamp":           txn.get("timestamp") or txn.get("created_at") or now,
        "evidence_refs": [
            {
                "label":            item,
                "reference_type":   "note",
                "reference_value":  item,
                "description":      f"FraudGuard evidence item — {decision.case_status}",
            }
            for item in evidence_summary[:10]
        ],
        "submission_metadata": {
            # ── Extended fields (stored in FRC submission_metadata) ────────
            "source_system":           "FraudGuard",
            "source_transaction_id":   transaction_id,
            "source_internal_case_id": internal_case_id,
            "institution_code":        INSTITUTION_CODE,
            "institution_name":        INSTITUTION_NAME,
            "transaction_type":        str(txn.get("type", "UNKNOWN")).upper(),
            "amount_in_usd_equivalent": round(amount_usd, 2),
            "amount_in_kes_equivalent": round(amount_kes, 2),
            "channel":                 txn.get("channel"),
            "origin_account_masked":   _mask_account(str(txn.get("nameOrig", ""))),
            "destination_account_masked": _mask_account(str(txn.get("nameDest", ""))),
            "ml_score":                decision.ml_score,
            "ml_risk_level":           decision.ml_risk_level,
            "final_risk_level":        decision.final_risk_level,
            "case_status":             decision.case_status,
            "report_type":             decision.report_type,
            "matched_legal_rules": [
                {"rule_id": r.rule_id, "section": r.section_or_regulation}
                for r in decision.matched_legal_rules
            ],
            "matched_policy_rules": [
                {"rule_id": r.rule_id, "rule_name": r.rule_name}
                for r in decision.matched_policy_rules
            ],
            "source_of_funds_flags":   sof_flags,
            "cross_border_flag":       is_cross_border,
            "auto_submission_reason":  (
                f"Automatic submission triggered by: "
                + ", ".join(r.rule_id for r in decision.all_matched_rules if r.auto_submit)
                + f" | ML score {ml_pct}%"
            ),
            "compliance_flags":        decision.compliance_flags,
            "submission_timestamp":    now,
        },
    }
