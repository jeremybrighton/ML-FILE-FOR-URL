"""
FraudGuard — FRC Submission Service
======================================
Handles automatic submission of compliance reports to the FRC backend.

Features:
  - Synchronous HTTP POST with retry logic (3 attempts, exponential backoff)
  - Stores FRC acknowledgement in MongoDB (compliance_submissions collection)
  - Returns a FRCSubmissionResult with full status tracking
  - Logs every submission attempt
  - Never raises — always returns a structured result

Env vars:
  FRC_API_URL     (default: https://financial-intelligence-processing-system.onrender.com/api/v1)
  FRC_API_KEY     (default: hardcoded FraudGuard institution key)
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
FRC_API_URL = os.environ.get(
    "FRC_API_URL",
    "https://financial-intelligence-processing-system.onrender.com/api/v1",
)
FRC_API_KEY = os.environ.get(
    "FRC_API_KEY",
    "frc_saaY9yYRvg4DeqTfEcENMArdqMOPiuHZ0AA4y2t8gtCEDpvE",
)
FRC_INTAKE_PATH = "/intake/cases"
FRC_TIMEOUT     = int(os.environ.get("FRC_SUBMISSION_TIMEOUT", "20"))
FRC_MAX_RETRIES = int(os.environ.get("FRC_MAX_RETRIES", "3"))


@dataclass
class FRCSubmissionResult:
    success: bool
    frc_case_id: Optional[str]
    status: str            # acknowledged | failed | skipped | already_submitted
    message: str
    error: Optional[str]
    attempts: int
    submitted_at: Optional[str]
    http_status: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


# ── MongoDB reference — injected by app.py after DB init ─────────────────────
# This avoids circular imports. Call inject_db() once at startup.
_compliance_submissions_col = None


def inject_db(db) -> None:
    """Call this from app.py after MongoDB is initialised."""
    global _compliance_submissions_col
    _compliance_submissions_col = db["compliance_submissions"]
    try:
        _compliance_submissions_col.create_index("transaction_id", background=True)
        _compliance_submissions_col.create_index("frc_case_id", background=True)
        _compliance_submissions_col.create_index("created_at", background=True)
    except Exception as e:
        log.warning(f"[FRCSubmission] Index creation warning: {e}")


def _store_result(
    transaction_id: str,
    internal_case_id: Optional[str],
    payload: dict,
    result: FRCSubmissionResult,
) -> None:
    """Persist submission record to MongoDB."""
    if _compliance_submissions_col is None:
        log.warning("[FRCSubmission] DB not injected — cannot store submission record.")
        return
    try:
        _compliance_submissions_col.insert_one({
            "transaction_id":    transaction_id,
            "internal_case_id":  internal_case_id,
            "frc_case_id":       result.frc_case_id,
            "status":            result.status,
            "success":           result.success,
            "message":           result.message,
            "error":             result.error,
            "attempts":          result.attempts,
            "http_status":       result.http_status,
            "submitted_at":      result.submitted_at,
            "payload_summary": {
                "report_type":    payload.get("report_type"),
                "amount":         payload.get("amount"),
                "currency":       payload.get("currency"),
                "risk_score":     payload.get("risk_score"),
                "external_id":    payload.get("external_report_id"),
            },
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        log.error(f"[FRCSubmission] Failed to store submission record: {e}")


def submit(
    payload: dict,
    transaction_id: str,
    internal_case_id: Optional[str] = None,
) -> FRCSubmissionResult:
    """
    Submit the FRC payload with retry logic.

    Returns FRCSubmissionResult — never raises.
    """
    if not FRC_API_KEY:
        result = FRCSubmissionResult(
            success=False,
            frc_case_id=None,
            status="failed",
            message="FRC_API_KEY not configured.",
            error="FRC_API_KEY not configured.",
            attempts=0,
            submitted_at=None,
            http_status=None,
        )
        _store_result(transaction_id, internal_case_id, payload, result)
        return result

    url = f"{FRC_API_URL.rstrip('/')}{FRC_INTAKE_PATH}"
    body_bytes = json.dumps(payload).encode("utf-8")

    last_error: Optional[str] = None
    last_status: Optional[int] = None

    for attempt in range(1, FRC_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "Content-Type":          "application/json",
                    "X-Institution-API-Key": FRC_API_KEY,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=FRC_TIMEOUT) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                frc_case_id = resp_data.get("frc_case_id")
                now = datetime.now(timezone.utc).isoformat()
                log.info(
                    f"[FRCSubmission] SUCCESS attempt={attempt} "
                    f"txn={transaction_id} frc_case_id={frc_case_id}"
                )
                result = FRCSubmissionResult(
                    success=True,
                    frc_case_id=frc_case_id,
                    status="acknowledged",
                    message=resp_data.get("message", "Case submitted to FRC successfully."),
                    error=None,
                    attempts=attempt,
                    submitted_at=now,
                    http_status=resp.status,
                )
                _store_result(transaction_id, internal_case_id, payload, result)
                return result

        except urllib.error.HTTPError as e:
            last_status = e.code
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            last_error = f"HTTP {e.code}: {err_body[:300]}"
            log.warning(
                f"[FRCSubmission] HTTPError attempt={attempt}/{FRC_MAX_RETRIES} "
                f"txn={transaction_id} status={e.code}: {err_body[:200]}"
            )
            # 4xx errors won't improve with retries — break early
            if 400 <= e.code < 500:
                break

        except Exception as e:
            last_error = str(e)[:300]
            log.warning(
                f"[FRCSubmission] Error attempt={attempt}/{FRC_MAX_RETRIES} "
                f"txn={transaction_id}: {last_error}"
            )

        if attempt < FRC_MAX_RETRIES:
            sleep_secs = 2 ** attempt   # 2, 4 seconds
            log.info(f"[FRCSubmission] Retrying in {sleep_secs}s...")
            time.sleep(sleep_secs)

    # All attempts exhausted
    log.error(
        f"[FRCSubmission] FAILED after {FRC_MAX_RETRIES} attempts "
        f"txn={transaction_id} last_error={last_error}"
    )
    result = FRCSubmissionResult(
        success=False,
        frc_case_id=None,
        status="failed",
        message=f"FRC submission failed after {FRC_MAX_RETRIES} attempts: {last_error}",
        error=last_error,
        attempts=FRC_MAX_RETRIES,
        submitted_at=None,
        http_status=last_status,
    )
    _store_result(transaction_id, internal_case_id, payload, result)
    return result
