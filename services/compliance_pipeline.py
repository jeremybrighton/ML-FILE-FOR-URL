"""
FraudGuard — Compliance Pipeline Orchestrator
==============================================
Single entry point that runs the full compliance pipeline for one transaction:

  1. Run legal/compliance rules engine
  2. Combine ML result + legal rules → CombinedRiskDecision
  3. Generate structured compliance output
  4. If auto-submit required: build FRC payload + submit immediately
  5. Return ComplianceResult with all outputs

Usage (from predict_endpoint in app.py):
    from services.compliance_pipeline import run_compliance_pipeline

    result = run_compliance_pipeline(
        txn=txn_dict,
        ml_score=0.87,
        ml_prediction=1,
        transaction_id="TXN_001",
        internal_case_id=None,   # optional FG case ID
    )
    # result.compliance_output — attach to transaction record
    # result.frc_result         — FRC submission status
    # result.auto_submitted     — bool
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from services import legal_rules_engine as lre
from services.risk_combiner import combine, CombinedRiskDecision, STATUS_AUTO_SUBMITTED
from services.report_generator import generate_compliance_output, build_frc_payload
from services import frc_submission_service as frc_svc

log = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    decision: CombinedRiskDecision
    compliance_output: dict
    frc_payload: Optional[dict]
    frc_result: Optional[Any]   # FRCSubmissionResult or None
    auto_submitted: bool

    @property
    def frc_submission_status(self) -> str:
        if self.frc_result is None:
            return "not_required"
        return self.frc_result.status

    @property
    def frc_case_id(self) -> Optional[str]:
        if self.frc_result and self.frc_result.frc_case_id:
            return self.frc_result.frc_case_id
        return None


def run_compliance_pipeline(
    txn: dict,
    ml_score: float,
    ml_prediction: int,
    transaction_id: str,
    internal_case_id: Optional[str] = None,
) -> ComplianceResult:
    """
    Run the full compliance pipeline for a single transaction.

    Parameters
    ----------
    txn              : raw transaction dict
    ml_score         : ML probability (0.0–1.0)
    ml_prediction    : ML binary flag (0 or 1)
    transaction_id   : unique transaction ID string
    internal_case_id : optional FraudGuard analyst case ID

    Returns
    -------
    ComplianceResult
    """
    # ── Step 1: Evaluate legal/compliance rules ───────────────────────────
    matched_rules = lre.evaluate(txn, ml_score)

    # ── Step 2: Combine ML + rules → risk decision ────────────────────────
    decision = combine(ml_score, ml_prediction, matched_rules)

    # ── Step 3: Generate compliance output ────────────────────────────────
    compliance_output = generate_compliance_output(txn, decision)

    # ── Step 4: Auto-submit to FRC if required ────────────────────────────
    frc_payload  = None
    frc_result   = None
    auto_submitted = False

    if decision.requires_auto_submit:
        frc_payload = build_frc_payload(
            txn=txn,
            decision=decision,
            compliance_output=compliance_output,
            transaction_id=transaction_id,
            internal_case_id=internal_case_id,
        )
        log.info(
            f"[CompliancePipeline] Auto-submitting txn={transaction_id} "
            f"status={decision.case_status} rules={[r.rule_id for r in decision.all_matched_rules]}"
        )
        frc_result = frc_svc.submit(
            payload=frc_payload,
            transaction_id=transaction_id,
            internal_case_id=internal_case_id,
        )
        auto_submitted = frc_result.success
        if frc_result.success:
            compliance_output["frc_submission_status"] = "acknowledged"
            compliance_output["frc_case_id"]           = frc_result.frc_case_id
            compliance_output["case_status"]           = STATUS_AUTO_SUBMITTED
            decision.case_status                       = STATUS_AUTO_SUBMITTED
            log.info(
                f"[CompliancePipeline] FRC ACK txn={transaction_id} "
                f"frc_case_id={frc_result.frc_case_id}"
            )
        else:
            compliance_output["frc_submission_status"] = "failed"
            compliance_output["frc_submission_error"]  = frc_result.error
            log.error(
                f"[CompliancePipeline] FRC submission FAILED txn={transaction_id}: "
                f"{frc_result.error}"
            )
    else:
        compliance_output["frc_submission_status"] = (
            "pending_manual_review"
            if decision.submission_mode == "manual_review_required"
            else "not_required"
        )

    return ComplianceResult(
        decision=decision,
        compliance_output=compliance_output,
        frc_payload=frc_payload,
        frc_result=frc_result,
        auto_submitted=auto_submitted,
    )
