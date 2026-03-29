"""
FraudGuard — Legal Rules Engine (POCAMLA Compliance)
=====================================================
Evaluates every transaction against the configured POCAMLA-based legal
and compliance rules defined in config/legal_rules.json.

Returns a list of MatchedRule objects each describing:
  - which rule fired
  - the legal act and section
  - whether this rule requires automatic FRC submission
  - the formatted legal reason string
  - the report type

This module has NO side effects — it only evaluates rules and returns
structured results. Submission is handled by frc_submission_service.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.currency_converter import to_usd, to_kes, infer_currency

log = logging.getLogger(__name__)

# ── Load rules config ─────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "legal_rules.json")

def _load_rules_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"[LegalRulesEngine] Failed to load legal_rules.json: {e}")
        return {"legal_rules": [], "policy_config": {}}

_rules_config: dict = _load_rules_config()
_legal_rules: list  = _rules_config.get("legal_rules", [])
_policy: dict       = _rules_config.get("policy_config", {})

# ── Convenience policy accessors ─────────────────────────────────────────────
KES_CRITICAL_THRESHOLD: float = float(_policy.get("kes_critical_threshold", 1_950_000))
USD_CTR_THRESHOLD: float      = float(_policy.get("usd_ctr_threshold", 15_000))
USD_CROSSBORDER_THRESHOLD: float = float(_policy.get("usd_crossborder_threshold", 10_000))
ML_AUTO_SUBMIT_SCORE: float   = float(_policy.get("ml_auto_submit_score", 0.70))
ML_HIGH_RISK_SCORE: float     = float(_policy.get("ml_high_risk_score", 0.50))
INSTITUTION_NAME: str         = _policy.get("institution", {}).get("institution_name", "FraudGuard Demo Bank")
INSTITUTION_CODE: str         = _policy.get("institution", {}).get("institution_code", "FRAUDGUARD-BANK")


@dataclass
class MatchedRule:
    rule_id: str
    rule_name: str
    act_name: str
    section_or_regulation: str
    rule_type: str
    severity: str                # CRITICAL | HIGH | MEDIUM | LOW
    auto_submit: bool
    requires_frc_submission: bool
    frc_case_type: str
    report_type: str
    submission_mode: str
    legal_reason: str
    triggering_rules_frc: List[str] = field(default_factory=list)
    is_legal_rule: bool = True   # False for policy/config rules
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _is_cross_border(txn: dict) -> bool:
    """Heuristic: detect cross-border transactions."""
    channel = str(txn.get("channel", "")).lower()
    tx_type = str(txn.get("type", "")).upper()
    region  = str(txn.get("region", "")).lower()
    name_dest = str(txn.get("nameDest", "")).upper()

    if any(k in channel for k in ("swift", "international", "cross", "forex", "wire")):
        return True
    if "international" in region or "foreign" in region:
        return True
    # In the dataset, destinations starting with "C" = customer, "M" = merchant
    # Explicit cross-border field
    if txn.get("cross_border") in (True, 1, "1", "true", "True"):
        return True
    return False


def _get_pattern_flags(txn: dict, amount: float, old_bal_org: float, new_bal_org: float,
                       old_bal_dest: float, new_bal_dest: float, tx_type: str) -> List[str]:
    """Identify pattern flags for Regulation 37 source-of-funds assessment."""
    flags = []
    if old_bal_org > 0 and new_bal_org == 0:
        flags.append("full_source_balance_depletion")
    if old_bal_dest == 0 and new_bal_dest > 0:
        flags.append("zero_balance_destination_account")
    if tx_type in ("TRANSFER", "CASH_OUT"):
        flags.append(f"high_risk_transaction_type_{tx_type}")
    if amount >= 500_000:
        flags.append("large_amount")
    channel = str(txn.get("channel", "")).lower()
    if channel:
        flags.append(f"channel_{channel.replace(' ', '_')}")
    return flags


def evaluate(txn: dict, ml_score: float) -> List[MatchedRule]:
    """
    Evaluate all enabled legal and compliance rules against a transaction.

    Parameters
    ----------
    txn       : raw transaction dict (as received from the client)
    ml_score  : float 0.0–1.0 from the ML model's predict_proba

    Returns
    -------
    List[MatchedRule] — may be empty if no rules fire.
    """
    matched: List[MatchedRule] = []

    # ── Extract transaction fields ────────────────────────────────────────
    currency    = infer_currency(txn)
    amount      = _safe_float(txn.get("amount"))
    tx_type     = str(txn.get("type", "")).upper()
    old_bal_org = _safe_float(txn.get("oldbalanceOrg"))
    new_bal_org = _safe_float(txn.get("newbalanceOrig"))
    old_bal_dest = _safe_float(txn.get("oldbalanceDest"))
    new_bal_dest = _safe_float(txn.get("newbalanceDest"))

    amount_usd = to_usd(amount, currency)
    amount_kes = to_kes(amount, currency)
    ml_score_pct = round(ml_score * 100, 1)
    is_cross_border = _is_cross_border(txn)

    pattern_flags = _get_pattern_flags(
        txn, amount, old_bal_org, new_bal_org,
        old_bal_dest, new_bal_dest, tx_type
    )

    for rule_def in _legal_rules:
        if not rule_def.get("enabled", True):
            continue

        rule_id   = rule_def["rule_id"]
        rule_type = rule_def.get("rule_type", "")
        trigger   = rule_def.get("trigger_type", "")

        fired = False
        reason = ""

        # ── Rule: Cash Transaction Reporting (REG40) ──────────────────────
        if rule_id == "REG40_CASH_CTR":
            if amount_usd >= USD_CTR_THRESHOLD:
                fired = True
                reason = rule_def["legal_reason_template"].format(
                    amount=f"{amount:,.2f}",
                    currency=currency,
                    usd_equivalent=f"{amount_usd:,.2f}",
                )

        # ── Rule: Cross-Border Declaration (REG10) ───────────────────────
        elif rule_id == "REG10_CROSSBORDER":
            if is_cross_border and amount_usd >= USD_CROSSBORDER_THRESHOLD:
                fired = True
                reason = rule_def["legal_reason_template"].format(
                    amount=f"{amount:,.2f}",
                    currency=currency,
                    usd_equivalent=f"{amount_usd:,.2f}",
                )

        # ── Rule: ML-Detected STR (SEC44_REG38) ──────────────────────────
        elif rule_id == "SEC44_REG38_STR":
            threshold = _safe_float(rule_def.get("threshold_value"), 0.70)
            if ml_score >= threshold:
                fired = True
                reason = rule_def["legal_reason_template"].format(
                    ml_score_pct=f"{ml_score_pct}",
                )

        # ── Rule: Source of Funds (REG37) ─────────────────────────────────
        elif rule_id == "REG37_SOURCE_OF_FUNDS":
            threshold = _safe_float(rule_def.get("threshold_value"), 0.50)
            if ml_score >= threshold or len(pattern_flags) >= 2 or amount_usd >= 5_000:
                fired = True
                flags_text = ", ".join(pattern_flags) if pattern_flags else "large or unusual transaction amount"
                reason = rule_def["legal_reason_template"].format(
                    pattern_flags=flags_text,
                )

        # ── Rule: KES Policy Threshold (POLICY_KES_THRESHOLD) ────────────
        elif rule_id == "POLICY_KES_THRESHOLD":
            if amount_kes >= KES_CRITICAL_THRESHOLD:
                fired = True
                reason = rule_def["legal_reason_template"].format(
                    kes_amount=f"{amount_kes:,.2f}",
                )

        # ── Rule: Balance Depletion Pattern (SEC44_BALANCE_DEPLETION) ─────
        elif rule_id == "SEC44_BALANCE_DEPLETION":
            if (old_bal_org > 0 and new_bal_org == 0
                    and tx_type in ("TRANSFER", "CASH_OUT")):
                fired = True
                reason = rule_def["legal_reason_template"].format(
                    tx_type=tx_type,
                    amount=f"{amount:,.2f}",
                    currency=currency,
                )

        # ── Rule: Zero-Balance Destination (SEC44_ZERO_DEST_BALANCE) ──────
        elif rule_id == "SEC44_ZERO_DEST_BALANCE":
            if (old_bal_dest == 0 and new_bal_dest > 0
                    and tx_type in ("TRANSFER", "CASH_OUT")
                    and amount_usd >= 1_000):
                fired = True
                reason = rule_def["legal_reason_template"].format(
                    amount=f"{amount:,.2f}",
                    currency=currency,
                )

        if fired:
            is_legal = rule_def.get("rule_type", "") != "policy_threshold"
            matched.append(MatchedRule(
                rule_id=rule_id,
                rule_name=rule_def["rule_name"],
                act_name=rule_def["act_name"],
                section_or_regulation=rule_def["section_or_regulation"],
                rule_type=rule_def.get("rule_type", ""),
                severity=rule_def.get("severity", "MEDIUM"),
                auto_submit=rule_def.get("auto_submit", False),
                requires_frc_submission=rule_def.get("requires_frc_submission", False),
                frc_case_type=rule_def.get("frc_case_type", "suspicious_activity_report"),
                report_type=rule_def.get("report_type", "suspicious_transaction_report"),
                submission_mode=rule_def.get("submission_mode", "manual_review_required"),
                legal_reason=reason,
                triggering_rules_frc=rule_def.get("triggering_rules_frc", []),
                is_legal_rule=is_legal,
                meta={
                    "amount": amount,
                    "currency": currency,
                    "amount_usd": round(amount_usd, 2),
                    "amount_kes": round(amount_kes, 2),
                    "ml_score": ml_score,
                    "tx_type": tx_type,
                    "is_cross_border": is_cross_border,
                    "pattern_flags": pattern_flags,
                },
            ))
            log.info(f"[LegalRulesEngine] Rule fired: {rule_id} | "
                     f"severity={rule_def.get('severity')} auto_submit={rule_def.get('auto_submit')}")

    return matched


def should_auto_submit(matched_rules: List[MatchedRule]) -> bool:
    """Return True if any matched rule requires automatic FRC submission."""
    return any(r.auto_submit and r.requires_frc_submission for r in matched_rules)


def highest_severity(matched_rules: List[MatchedRule]) -> str:
    """Return the highest severity among matched rules."""
    order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    if not matched_rules:
        return "LOW"
    return max(matched_rules, key=lambda r: order.get(r.severity, 0)).severity


def primary_report_type(matched_rules: List[MatchedRule]) -> str:
    """
    Determine the primary report type from matched rules.
    Priority: cash_transaction_report > critical_policy_threshold_escalation
              > cross_border_declaration_alert > suspicious_transaction_report
    """
    priority = {
        "cash_transaction_report": 5,
        "critical_policy_threshold_escalation": 4,
        "cross_border_declaration_alert": 3,
        "suspicious_transaction_report": 2,
    }
    if not matched_rules:
        return "no_report_required"
    return max(matched_rules, key=lambda r: priority.get(r.report_type, 0)).report_type


def primary_frc_case_type(matched_rules: List[MatchedRule]) -> str:
    """Return the FRC case type for the highest-priority rule."""
    priority = {
        "regulatory_threshold_report": 3,
        "suspicious_activity_report": 2,
    }
    if not matched_rules:
        return "suspicious_activity_report"
    return max(matched_rules, key=lambda r: priority.get(r.frc_case_type, 0)).frc_case_type
