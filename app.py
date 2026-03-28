"""
FraudGuard ML + Analyst AI Backend
====================================
Production-ready Flask backend for the FraudGuard Vercel frontend.

Covers:
  - Auth  : /login, /register, /request-otp, /verify-otp
              /login/verify, /login/resend (aliases for frontend compat)
              /login/forgot-password, /login/reset-password
  - ML    : /predict, /process-dataset, /explain/<id>, /report/<id>, /ai-case/<id>
  - Batch : /batches/<batch_id>/summary
              /batches/<batch_id>/transactions  (paginated, filterable)
              /batches/<batch_id>/all
              /batches/<batch_id>/report
  - Admin : /admin/users (GET/POST), /admin/users/<id> (DELETE/PUT)
              /admin/transactions, /admin/logs, /admin/stats
  - Analyst: /analyst/cases (GET/POST)
              /analyst/cases/<id> (GET/PUT/DELETE)
              /analyst/chat
              /analyst/review
              /analyst/reviews/<id>
              /analyst/cases/<id>/export
              /analyst/cases/<id>/request-evidence
              /analyst/cases/<id>/send-review
  - Health: /, /health

Start command (Render):
  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2

Required environment variables:
  MONGO_URI          MongoDB Atlas connection string
  OPENAI_API_KEY     OpenAI key (optional — fallback used if absent)
  OPENAI_MODEL       e.g. gpt-4o-mini  (default: gpt-4o-mini)
  SECRET_KEY         Flask secret key for signing tokens
  SENDER_EMAIL       Gmail address for OTP (optional)
  SENDER_PASSWORD    Gmail app password for OTP (optional)
  OTP_EXPIRY_MINUTES Minutes before OTP expires (default: 5)

Response format change (v2 — batch refactor):
  /predict and /process-dataset no longer return the full predictions array in the
  upload response when a dataset is large.  Instead they return a batch_id plus
  summary counts.  Use the /batches/* endpoints to retrieve all results.

  POST /process-dataset  →  {success, batch_id, total_rows, flagged_count,
                              legitimate_count, high_risk_count, medium_risk_count,
                              low_risk_count, processing_time_seconds}

  POST /predict          →  {success, batch_id, predictions (always included,
                              size == input), total_rows, flagged_count, ...}
"""

import os
import json
import math
import time
import random
import logging
import smtplib
import traceback
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from io import StringIO
from email.mime.text import MIMEText
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g


# ─────────────────────────────────────────────
# Custom exception for staged ML pipeline errors
# ─────────────────────────────────────────────
class PipelineError(Exception):
    """Raised when the ML prediction pipeline fails at a specific stage."""
    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage
from flask_cors import CORS
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.server_api import ServerApi
from werkzeug.security import generate_password_hash, check_password_hash

# Optional OpenAI — import only if available
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fraudguard-secret-key-change-in-production")

ALLOWED_ORIGINS = [
    "https://fraud-detector-b.vercel.app",
    "https://fraud-detector-topaz.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ─────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set!")

SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
OTP_EXPIRY_MINUTES = int(os.environ.get("OTP_EXPIRY_MINUTES", 5))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ─── FRC Integration ──────────────────────────────────────────────────────────
# Set these environment variables in Render (or .env locally).
FRC_API_URL      = os.environ.get("FRC_API_URL",
                                   "https://financial-intelligence-processing-system.onrender.com/api/v1")
FRC_API_KEY      = os.environ.get("FRC_API_KEY", "")          # X-Institution-API-Key
FRC_INTAKE_PATH  = "/intake/cases"

# Chunk size for insert_many batches (tunable via env var)
MONGO_INSERT_CHUNK = int(os.environ.get("MONGO_INSERT_CHUNK", 200))

PROJECT_PATH = "."
MODEL_PATH = os.path.join(PROJECT_PATH, "rf_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PROJECT_PATH, "feature_columns.json")

# ─────────────────────────────────────────────
# OpenAI Client (optional)
# ─────────────────────────────────────────────
openai_client = None
if _openai_available and OPENAI_API_KEY:
    try:
        from openai import OpenAI as _OpenAI
        openai_client = _OpenAI(api_key=OPENAI_API_KEY)
        log.info("✅ OpenAI client initialised")
    except Exception as e:
        log.warning(f"OpenAI init failed: {e}")

# ─────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────
try:
    mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"), tls=True)
    db = mongo_client["fraud_detection"]

    users_col            = db["users"]
    transactions_col     = db["transactions"]
    admin_col            = db["admin_actions"]
    ai_cache_col         = db["ai_cache"]
    sessions_col         = db["sessions"]
    analyst_cases_col    = db["analyst_cases"]
    analyst_reviews_col  = db["analyst_reviews"]
    batches_col          = db["batches"]          # NEW: batch metadata

    # Indexes
    sessions_col.create_index("token", unique=True, background=True)
    sessions_col.create_index("expires_at", expireAfterSeconds=0, background=True)
    analyst_cases_col.create_index("case_id", unique=True, background=True)
    analyst_cases_col.create_index("transaction_id", background=True)
    analyst_reviews_col.create_index("case_id", background=True)
    transactions_col.create_index("transaction_id", background=True)
    # New batch-oriented indexes
    transactions_col.create_index("batch_id", background=True)
    transactions_col.create_index([("batch_id", ASCENDING), ("prediction", ASCENDING)],
                                  background=True)
    transactions_col.create_index([("batch_id", ASCENDING), ("risk_level", ASCENDING)],
                                  background=True)
    transactions_col.create_index([("batch_id", ASCENDING), ("created_at", DESCENDING)],
                                  background=True)
    batches_col.create_index("batch_id", unique=True, background=True)

    mongo_client.admin.command("ping")
    log.info("✅ MongoDB connected successfully!")
except Exception as e:
    log.error(f"❌ MongoDB connection failed: {e}")
    raise

# ─────────────────────────────────────────────
# Load ML Model
# ─────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    log.info("✅ ML model loaded")
except Exception as e:
    log.error(f"❌ ML model load failed: {e}")
    raise

# ─────────────────────────────────────────────
# Global Error Handlers — always return JSON
# ─────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success": False, "error": "Bad request", "detail": str(e)}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"success": False, "error": "Unauthorized"}), 401

@app.errorhandler(403)
def forbidden(e):
    return jsonify({"success": False, "error": "Forbidden"}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Not found", "path": request.path}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    log.exception("Unhandled server error")
    return jsonify({"success": False, "error": "Internal server error"}), 500

# ─────────────────────────────────────────────
# CORS preflight — always 200 before any auth
# ─────────────────────────────────────────────
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

# ─────────────────────────────────────────────
# Serialization Helpers
# ─────────────────────────────────────────────
def json_safe_value(value):
    """Recursively convert any value to a JSON-serialisable Python type."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_safe_value(x) for x in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe_value(v) for v in value]
    return value


def serialize_document(doc):
    if not doc:
        return None
    return {
        ("_id" if k == "_id" else k): (str(value) if k == "_id" else json_safe_value(value))
        for k, value in doc.items()
    }


def serialize_documents(docs):
    return [serialize_document(d) for d in docs if d]

# ─────────────────────────────────────────────
# Generic Helpers
# ─────────────────────────────────────────────
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        v = float(value)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return default if value is None else int(value)
    except Exception:
        return default


def normalize_string(value):
    return None if value is None else str(value).strip()


def log_admin_action(action, details=None):
    try:
        admin_col.insert_one({
            "action": action,
            "details": details or {},
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        log.warning(f"Failed to log admin action: {e}")


# ─────────────────────────────────────────────
# FRC Intake Submission Helper
# ─────────────────────────────────────────────
def build_frc_payload(case: dict) -> dict:
    """
    Build the structured JSON payload for the FRC intake endpoint
    from a FraudGuard analyst case document.
    """
    raw_score = safe_float(case.get("risk_score", 0))
    # Normalise: if score was stored as a percentage (e.g. 87.3) convert to 0-1
    score_01 = raw_score / 100.0 if raw_score > 1 else raw_score

    txn_type = ""
    for ev in (case.get("evidence") or []):
        if ev.get("type") == "transaction_type":
            txn_type = ev.get("value", "")
            break

    # Map FraudGuard case_type → FRC report_type
    case_type_lower = (case.get("case_type") or "").lower()
    if "suspicious" in case_type_lower or "fraud" in case_type_lower:
        report_type = "suspicious_activity_report"
    else:
        report_type = "regulatory_threshold_report"

    # Map risk drivers to FRC triggering_rules
    triggering_rules = []
    if score_01 >= 0.7 or "TRANSFER" in txn_type.upper() or "CASH_OUT" in txn_type.upper():
        triggering_rules.append("POCAMLA-S44-STR-GENERAL")
    if score_01 >= 0.85:
        triggering_rules.append("POCAMLA-REG38-STR-DETAIL")
    amount = safe_float(case.get("structured_report", {}).get("risk_score") or
                        next((e.get("value", "0").replace("KES ", "").replace(",", "")
                              for e in (case.get("evidence") or [])
                              if e.get("type") == "amount"), "0"))
    if amount == 0:
        # Try to parse from evidence label "KES 250,000"
        for ev in (case.get("evidence") or []):
            if ev.get("type") == "amount":
                try:
                    amount = float(str(ev.get("value", "0")).replace("KES ", "").replace(",", ""))
                except Exception:
                    amount = 0
                break

    # Build concise transaction summary
    reasons = case.get("reasons") or []
    summary = case.get("summary") or "Transaction flagged as suspicious."
    tx_summary = f"{summary} Risk indicators: {'; '.join(reasons[:3])}." if reasons else summary

    # Evidence references
    evidence_refs = []
    for ev in (case.get("evidence") or []):
        if ev.get("type") in ("model_score", "rule_trigger", "transaction_type"):
            evidence_refs.append({
                "label": ev.get("label", ev.get("type")),
                "reference_type": "note",
                "reference_value": str(ev.get("value", "")),
                "description": f"FraudGuard evidence: {ev.get('type')}",
            })

    payload = {
        "external_report_id": case.get("case_id"),
        "report_type": report_type,
        "amount": round(amount, 2) if amount else None,
        "currency": "KES",
        "transaction_summary": tx_summary[:2000],
        "triggering_rules": triggering_rules if triggering_rules else ["POCAMLA-S44-STR-GENERAL"],
        "risk_score": round(score_01, 4),
        "narrative": (case.get("narrative_report") or case.get("summary") or "")[:5000],
        "timestamp": case.get("created_at"),
        "evidence_refs": evidence_refs[:10],
        "submission_metadata": {
            "source_system": "FraudGuard",
            "source_case_id": case.get("case_id"),
            "source_transaction_id": case.get("transaction_id"),
            "source_risk_level": case.get("risk_level"),
            "source_case_type": case.get("case_type"),
            "submitted_by": "FraudGuard analyst escalation",
        },
    }
    return payload


def submit_to_frc(case: dict) -> dict:
    """
    Submit a FraudGuard analyst case to the FRC backend intake endpoint.

    Returns a dict with keys:
      success    bool
      frc_case_id  str | None
      status     str
      message    str
      error      str | None
    """
    if not FRC_API_KEY:
        msg = "FRC_API_KEY not configured. Set the environment variable to enable FRC submission."
        log.warning(msg)
        return {"success": False, "frc_case_id": None, "status": "failed", "message": msg, "error": msg}

    url = f"{FRC_API_URL.rstrip('/')}{FRC_INTAKE_PATH}"
    payload = build_frc_payload(case)

    try:
        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Institution-API-Key": FRC_API_KEY,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            frc_case_id = resp_body.get("frc_case_id")
            log.info(f"FRC intake success: frc_case_id={frc_case_id} source={case.get('case_id')}")
            return {
                "success": True,
                "frc_case_id": frc_case_id,
                "status": "acknowledged",
                "message": resp_body.get("message", "Case submitted to FRC successfully."),
                "error": None,
            }
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        msg = f"FRC intake HTTP {e.code}: {err_body[:300]}"
        log.error(f"FRC submission failed: {msg} | source_case={case.get('case_id')}")
        return {"success": False, "frc_case_id": None, "status": "failed", "message": msg, "error": msg}
    except Exception as e:
        msg = f"FRC submission error: {str(e)[:300]}"
        log.error(f"{msg} | source_case={case.get('case_id')}")
        return {"success": False, "frc_case_id": None, "status": "failed", "message": msg, "error": msg}

# ─────────────────────────────────────────────
# Token / Session Management  (MongoDB-backed)
# ─────────────────────────────────────────────
SESSION_TTL_HOURS = 24

def create_session_token(user_doc):
    token = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)
    sessions_col.insert_one({
        "token": token,
        "user_id": str(user_doc["_id"]),
        "email": user_doc["email"],
        "name": user_doc.get("name", user_doc["email"].split("@")[0]),
        "role": user_doc.get("role", "user"),
        "is_active": user_doc.get("is_active", True),
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
    })
    return token


def get_session_from_token(token):
    if not token:
        return None
    session = sessions_col.find_one({"token": token})
    if not session:
        return None
    if session.get("expires_at") and datetime.utcnow() > session["expires_at"]:
        sessions_col.delete_one({"token": token})
        return None
    return session


def get_current_user():
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else None
    return get_session_from_token(token)


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "error": "Authentication required"}), 401
        if not user.get("is_active", True):
            return jsonify({"success": False, "error": "Account is deactivated"}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "error": "Authentication required"}), 401
        if user.get("role") != "admin":
            return jsonify({"success": False, "error": "Admin access required"}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# OTP Helpers
# ─────────────────────────────────────────────
def generate_otp(length=6):
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def send_email_otp(to_email, otp_code):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        log.warning("SMTP credentials not configured — OTP not sent.")
        return False
    try:
        msg = MIMEText(
            f"Your FraudGuard OTP: {otp_code}\nExpires in {OTP_EXPIRY_MINUTES} minutes."
        )
        msg["Subject"] = "FraudGuard — Your OTP Code"
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        log.info(f"OTP sent to {to_email}")
        return True
    except Exception as e:
        log.error(f"Failed to send OTP: {e}")
        return False

# ═════════════════════════════════════════════
# ML Helpers — hardened predict_internal
# ═════════════════════════════════════════════

# Required columns that must exist before we attempt prediction.
# These are the raw input columns the model was trained on (pre-dummies).
REQUIRED_INPUT_COLS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

def predict_internal(new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run batch fraud prediction.

    Parameters
    ----------
    new_data : pd.DataFrame
        Raw transaction data.  May contain extra columns — they are ignored.

    Returns
    -------
    pd.DataFrame with columns: prediction, fraud_score, risk_level

    Raises
    ------
    ValueError  with a 'stage' attribute set to the failing pipeline step.
    """

    # ── 1. Validate shape / required columns ──────────────────────────────
    log.info(f"[predict_internal] Input shape: {new_data.shape}")
    log.info(f"[predict_internal] Columns: {list(new_data.columns)}")

    null_counts = new_data.isnull().sum()
    nonzero_nulls = null_counts[null_counts > 0]
    if not nonzero_nulls.empty:
        log.warning(f"[predict_internal] Null counts: {nonzero_nulls.to_dict()}")

    log.info(f"[predict_internal] Dtypes: {new_data.dtypes.to_dict()}")

    missing = [c for c in REQUIRED_INPUT_COLS if c not in new_data.columns]
    if missing:
        raise PipelineError(f"Missing required columns: {missing}", "column_validation")

    # ── 2. get_dummies ─────────────────────────────────────────────────────
    try:
        t0 = time.time()
        processed = pd.get_dummies(new_data, drop_first=True)
        log.info(f"[predict_internal] get_dummies done in {time.time()-t0:.3f}s, "
                 f"shape after: {processed.shape}")
    except PipelineError:
        raise
    except Exception as exc:
        log.error(f"[predict_internal] get_dummies failed: {exc}")
        raise PipelineError(f"get_dummies failed: {exc}", "get_dummies") from exc

    # ── 3. Column alignment ────────────────────────────────────────────────
    try:
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0
        aligned = processed[feature_cols]
        log.info(f"[predict_internal] Alignment done, shape: {aligned.shape}")
    except PipelineError:
        raise
    except Exception as exc:
        log.error(f"[predict_internal] column alignment failed: {exc}")
        raise PipelineError(f"Column alignment failed: {exc}", "column_alignment") from exc

    # ── 4. astype(float) ───────────────────────────────────────────────────
    try:
        aligned = aligned.astype(float)
    except PipelineError:
        raise
    except Exception as exc:
        log.error(f"[predict_internal] astype(float) failed: {exc}")
        raise PipelineError(f"astype(float) failed: {exc}", "astype_float") from exc

    # ── 5. predict_proba ───────────────────────────────────────────────────
    try:
        t0 = time.time()
        prob = model.predict_proba(aligned)[:, 1]
        log.info(f"[predict_internal] predict_proba done in {time.time()-t0:.3f}s "
                 f"for {len(prob)} rows")
    except PipelineError:
        raise
    except Exception as exc:
        log.error(f"[predict_internal] predict_proba failed: {exc}")
        raise PipelineError(f"predict_proba failed: {exc}", "predict_proba") from exc

    pred = (prob >= 0.5).astype(int)
    risk = pd.Series(
        np.where(prob < 0.2, "LOW", np.where(prob < 0.8, "MEDIUM", "HIGH")),
        index=aligned.index,
    )
    return pd.DataFrame(
        {"prediction": pred, "fraud_score": prob, "risk_level": risk},
        index=aligned.index,
    )


# ═════════════════════════════════════════════
# Batch helpers
# ═════════════════════════════════════════════

def new_batch_id() -> str:
    """Generate a short, sortable batch identifier."""
    return f"BATCH-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"


def save_batch_metadata(batch_id: str, meta: dict):
    """Upsert the batch summary document."""
    batches_col.update_one(
        {"batch_id": batch_id},
        {"$set": {**meta, "batch_id": batch_id}},
        upsert=True,
    )


def insert_many_chunked(collection, docs: list, chunk_size: int = MONGO_INSERT_CHUNK):
    """
    Insert a list of documents in chunks to avoid oversized write operations.
    Returns the total number of inserted documents.
    """
    inserted = 0
    for start in range(0, len(docs), chunk_size):
        chunk = docs[start : start + chunk_size]
        if chunk:
            collection.insert_many(chunk, ordered=False)
            inserted += len(chunk)
    return inserted


def compute_batch_stats(df: pd.DataFrame) -> dict:
    """Derive summary counts from a scored dataframe."""
    total       = len(df)
    flagged     = int((df["prediction"] == 1).sum())
    legitimate  = total - flagged
    high_risk   = int((df["risk_level"] == "HIGH").sum())
    medium_risk = int((df["risk_level"] == "MEDIUM").sum())
    low_risk    = int((df["risk_level"] == "LOW").sum())
    return {
        "total_rows":       total,
        "flagged_count":    flagged,
        "legitimate_count": legitimate,
        "high_risk_count":  high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count":   low_risk,
    }

# ─────────────────────────────────────────────
# AI / OpenAI Helpers
# ─────────────────────────────────────────────
def call_openai_chat(prompt: str, fallback: str = "") -> str:
    if not openai_client:
        return fallback
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are FraudGuard AI, a professional fraud operations analyst assistant. "
                        "Use compliance-safe language. Never accuse anyone of fraud directly. "
                        "Say 'flagged as suspicious', 'possible fraud indicators', 'requires analyst review'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"OpenAI call failed: {e}")
        return fallback


def call_openai_json(prompt: str, fallback_payload: dict) -> dict:
    if not openai_client:
        return fallback_payload
    clean = ""
    try:
        raw = call_openai_chat(prompt)
        if not raw:
            return fallback_payload
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        log.warning("OpenAI returned non-JSON; wrapping as fallback")
        fb = fallback_payload.copy()
        fb["raw_text"] = clean
        return fb
    except Exception as e:
        log.warning(f"OpenAI JSON call failed: {e}")
        return fallback_payload


def build_transaction_context(txn: dict) -> dict:
    txn_safe = serialize_document(txn) or {}
    known = {
        "transaction_id": txn_safe.get("transaction_id"),
        "step":           txn_safe.get("step"),
        "type":           txn_safe.get("type"),
        "amount":         safe_float(txn_safe.get("amount")),
        "nameOrig":       txn_safe.get("nameOrig"),
        "recipient_name": txn_safe.get("recipient_name"),
        "nameDest":       txn_safe.get("nameDest"),
        "oldbalanceOrg":  safe_float(txn_safe.get("oldbalanceOrg")),
        "newbalanceOrig": safe_float(txn_safe.get("newbalanceOrig")),
        "oldbalanceDest": safe_float(txn_safe.get("oldbalanceDest")),
        "newbalanceDest": safe_float(txn_safe.get("newbalanceDest")),
        "timestamp":      txn_safe.get("timestamp"),
        "channel":        txn_safe.get("channel"),
        "region":         txn_safe.get("region"),
        "device_id":      txn_safe.get("device_id"),
        "prediction":     safe_int(txn_safe.get("prediction", 0)),
        "fraud_score":    round(safe_float(txn_safe.get("fraud_score", 0.0)), 6),
        "risk_level":     txn_safe.get("risk_level", "UNKNOWN"),
        "created_at":     txn_safe.get("created_at"),
    }
    excluded = set(known.keys()) | {"_id"}
    extra = {k: v for k, v in txn_safe.items() if k not in excluded}
    known["sender_name"] = known["nameOrig"]
    known["receiver_name"] = known["recipient_name"] or known["nameDest"]
    known["extra_fields"] = extra
    return known


def derive_rule_based_evidence(ctx: dict) -> dict:
    evidence, risk_drivers, authorities, actions = [], [], [], []
    case_type = "internal_review"

    amount       = safe_float(ctx.get("amount"))
    fraud_score  = safe_float(ctx.get("fraud_score"))
    tx_type      = (normalize_string(ctx.get("type")) or "").upper()
    old_org      = safe_float(ctx.get("oldbalanceOrg"))
    new_org      = safe_float(ctx.get("newbalanceOrig"))
    old_dest     = safe_float(ctx.get("oldbalanceDest"))
    new_dest     = safe_float(ctx.get("newbalanceDest"))
    channel      = normalize_string(ctx.get("channel"))
    region       = normalize_string(ctx.get("region"))
    device_id    = normalize_string(ctx.get("device_id"))
    sender_name  = normalize_string(ctx.get("sender_name"))
    receiver_name = normalize_string(ctx.get("receiver_name"))

    if fraud_score >= 0.8:
        evidence.append("Model assigned a high fraud score.")
        risk_drivers.append("High model fraud score")
    elif fraud_score >= 0.5:
        evidence.append("Model assigned a medium-to-high fraud score.")
        risk_drivers.append("Elevated model fraud score")

    if amount >= 1_000_000:
        evidence.append("Transaction amount is extremely high.")
        risk_drivers.append("Very high transaction amount")
    elif amount >= 100_000:
        evidence.append("Transaction amount is unusually high.")
        risk_drivers.append("High transaction amount")

    if old_org > 0 and new_org == 0:
        evidence.append("Source account balance depleted to zero.")
        risk_drivers.append("Full source balance depletion")

    if old_dest == 0 and new_dest > 0:
        evidence.append("Destination account had zero balance before receiving funds.")
        risk_drivers.append("Zero-balance destination account")

    if tx_type in {"TRANSFER", "CASH_OUT"}:
        evidence.append(f"Transaction type {tx_type} is high-risk for fraud monitoring.")
        risk_drivers.append(f"High-risk type: {tx_type}")

    if channel:   evidence.append(f"Channel: {channel}.")
    if region:    evidence.append(f"Region: {region}.")
    if device_id: evidence.append("Device identifier available.")
    if sender_name:   evidence.append(f"Sender: {sender_name}.")
    if receiver_name: evidence.append(f"Receiver: {receiver_name}.")

    actions.append("Escalate for analyst review before any external submission.")
    actions.append("Preserve transaction record and supporting logs.")
    if device_id: actions.append("Retain device metadata for investigation.")
    if fraud_score >= 0.8: actions.append("Prioritise for urgent manual review.")
    if tx_type in {"TRANSFER", "CASH_OUT"} and fraud_score >= 0.7:
        actions.append("Review linked transfer activity.")

    if fraud_score >= 0.7 or tx_type in {"TRANSFER", "CASH_OUT"}:
        authorities.append("DCI")
        case_type = "possible_cyber_or_financial_fraud"
    if amount >= 100_000 or fraud_score >= 0.85:
        authorities.append("FRC")
        case_type = "suspicious_transaction_review"
    if not authorities:
        authorities.append("Internal Review Only")

    return {
        "case_type": case_type,
        "evidence": list(dict.fromkeys(evidence)),
        "risk_drivers": list(dict.fromkeys(risk_drivers)),
        "recommended_authorities": list(dict.fromkeys(authorities)),
        "recommended_actions": list(dict.fromkeys(actions)),
    }


def get_cached_ai_result(transaction_id, task_type):
    doc = ai_cache_col.find_one({"transaction_id": transaction_id, "task_type": task_type})
    return serialize_document(doc) if doc else None


def save_cached_ai_result(transaction_id, task_type, payload):
    ai_cache_col.update_one(
        {"transaction_id": transaction_id, "task_type": task_type},
        {"$set": {"transaction_id": transaction_id, "task_type": task_type,
                  "payload": payload, "updated_at": datetime.utcnow()}},
        upsert=True,
    )


def generate_ai_transaction_explanation(txn):
    ctx = build_transaction_context(txn)
    derived = derive_rule_based_evidence(ctx)
    fallback = {
        "summary": "This transaction was flagged as suspicious and should be reviewed by an analyst.",
        "risk_drivers": derived["risk_drivers"],
        "recommendation": "Review manually and verify supporting evidence before taking action.",
        "confidence_note": "Rule-based fallback — OpenAI not available.",
    }
    prompt = f"""
Explain why this transaction received its fraud score.
Use ONLY the data below. Do not invent facts. Do not accuse anyone.
Use wording: "flagged as suspicious", "possible fraud indicators", "requires analyst review".

Return ONLY valid JSON:
{{"summary":"...","risk_drivers":["..."],"recommendation":"...","confidence_note":"..."}}

Transaction context:
{json.dumps(ctx, indent=2)}

Derived evidence:
{json.dumps(derived, indent=2)}
"""
    ai = call_openai_json(prompt, fallback)
    return {
        "transaction_id": ctx["transaction_id"],
        "fraud_score": ctx["fraud_score"],
        "risk_level": ctx["risk_level"],
        "prediction": ctx["prediction"],
        "sender_name": ctx["sender_name"],
        "receiver_name": ctx["receiver_name"],
        "summary": ai.get("summary", fallback["summary"]),
        "risk_drivers": ai.get("risk_drivers", derived["risk_drivers"]),
        "recommendation": ai.get("recommendation", fallback["recommendation"]),
        "confidence_note": ai.get("confidence_note", fallback["confidence_note"]),
        "evidence": derived["evidence"],
        "recommended_authorities": derived["recommended_authorities"],
        "extra_fields": ctx.get("extra_fields", {}),
    }


def generate_ai_report(txn):
    ctx = build_transaction_context(txn)
    derived = derive_rule_based_evidence(ctx)
    fallback = {
        "case_type": derived["case_type"],
        "recommended_authority": derived["recommended_authorities"],
        "incident_summary": "Transaction flagged as suspicious — recommended for internal analyst review.",
        "reason_for_suspicion": derived["risk_drivers"],
        "evidence": derived["evidence"],
        "recommended_actions": derived["recommended_actions"],
        "human_review_required": True,
    }
    prompt = f"""
Prepare a professional fraud incident report draft.
Use ONLY the data provided. No legal conclusions. No direct fraud accusations.

Return ONLY valid JSON:
{{"case_type":"...","recommended_authority":["DCI"],"incident_summary":"...",
"reason_for_suspicion":["..."],"evidence":["..."],"recommended_actions":["..."],"human_review_required":true}}

Transaction context:
{json.dumps(ctx, indent=2)}

Derived evidence:
{json.dumps(derived, indent=2)}
"""
    ai = call_openai_json(prompt, fallback)
    return {
        "transaction_id": ctx["transaction_id"],
        "fraud_score": ctx["fraud_score"],
        "risk_level": ctx["risk_level"],
        "sender_name": ctx["sender_name"],
        "receiver_name": ctx["receiver_name"],
        "report": {
            "case_type": ai.get("case_type", derived["case_type"]),
            "recommended_authority": ai.get("recommended_authority", derived["recommended_authorities"]),
            "incident_summary": ai.get("incident_summary", fallback["incident_summary"]),
            "reason_for_suspicion": ai.get("reason_for_suspicion", derived["risk_drivers"]),
            "evidence": ai.get("evidence", derived["evidence"]),
            "recommended_actions": ai.get("recommended_actions", derived["recommended_actions"]),
            "human_review_required": True,
        },
    }


def generate_ai_case_bundle(txn):
    explanation = generate_ai_transaction_explanation(txn)
    report = generate_ai_report(txn)
    return {
        "transaction_id": explanation["transaction_id"],
        "fraud_score": explanation["fraud_score"],
        "risk_level": explanation["risk_level"],
        "prediction": explanation["prediction"],
        "sender_name": explanation["sender_name"],
        "receiver_name": explanation["receiver_name"],
        "explanation": explanation,
        "report": report["report"],
    }

# ─────────────────────────────────────────────
# Analyst Case Helpers
# ─────────────────────────────────────────────
def _next_case_id() -> str:
    count = analyst_cases_col.count_documents({})
    return f"FG-2026-{count + 1:05d}"


def _serialize_case(doc: dict) -> dict:
    if not doc:
        return {}
    serialized = serialize_document(doc)
    if not serialized:
        return {}
    return {k: v for k, v in serialized.items()}


def _build_analyst_case(transaction_id: str, txn_data: dict) -> dict:
    raw_score = safe_float(txn_data.get("fraud_score", 0))
    score_pct = raw_score if raw_score > 1 else raw_score * 100
    score_01 = score_pct / 100.0

    risk_level = (
        "HIGH"       if score_01 >= 0.7  else
        "SUSPICIOUS" if score_01 >= 0.5  else
        "MEDIUM"     if score_01 >= 0.3  else "LOW"
    )

    ctx = build_transaction_context({**txn_data, "fraud_score": score_01})
    derived = derive_rule_based_evidence(ctx)

    reasons = derived["risk_drivers"] or [f"Fraud model score: {score_pct:.1f}%"]
    authorities = derived["recommended_authorities"]
    case_type = derived["case_type"]
    actions = derived["recommended_actions"]

    evidence_items = [
        {"type": "model_score",      "label": "Fraud Model Score",  "value": f"{score_pct:.1f}%"},
        {"type": "amount",           "label": "Transaction Amount", "value": f"KES {safe_float(txn_data.get('amount', 0)):,.0f}"},
        {"type": "channel",          "label": "Channel",            "value": str(txn_data.get("channel", "Unknown"))},
        {"type": "transaction_type", "label": "Transaction Type",   "value": str(txn_data.get("type", "Unknown"))},
        {"type": "sender",           "label": "Sender Account",     "value": str(txn_data.get("nameOrig", txn_data.get("sender", "Unknown")))},
        {"type": "recipient",        "label": "Recipient Account",  "value": str(txn_data.get("nameDest", txn_data.get("recipient", "Unknown")))},
    ]
    for ev in derived["evidence"]:
        evidence_items.append({"type": "rule_trigger", "label": "Rule Trigger", "value": ev})

    timeline = [
        {"timestamp": datetime.utcnow().isoformat(), "event": "transaction_flagged",
         "description": "Transaction flagged by ML fraud model"},
        {"timestamp": datetime.utcnow().isoformat(), "event": "case_created",
         "description": "Analyst case created and queued for review"},
    ]

    summary = (
        f"This transaction was flagged as {risk_level} risk with a fraud model score of "
        f"{score_pct:.1f}%. The system detected indicators consistent with possible fraud. "
        f"Human analyst review is required before any external reporting."
    )
    if openai_client:
        ai_summary = call_openai_chat(
            f"""Write a 2-sentence professional fraud case summary.
Case: {transaction_id} | Risk: {risk_level} ({score_pct:.1f}%) | Type: {case_type}
Indicators: {', '.join(reasons[:4])} | Routing: {', '.join(authorities)}
Use compliance-safe language. Never say 'confirmed fraud'. Return only the paragraph.""",
            fallback=summary,
        )
        if ai_summary:
            summary = ai_summary

    now = datetime.utcnow().isoformat()
    case_id = _next_case_id()

    narrative = (
        f"CASE SUMMARY REPORT\n{'='*50}\n"
        f"Case ID:       {case_id}\n"
        f"Transaction:   {transaction_id}\n"
        f"Risk Level:    {risk_level}  ({score_pct:.1f}%)\n"
        f"Case Type:     {case_type.replace('_', ' ').title()}\n"
        f"Routing:       {', '.join(authorities)}\n"
        f"Date:          {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"SUMMARY\n{'-'*50}\n{summary}\n\n"
        f"KEY INDICATORS\n{'-'*50}\n"
        + "\n".join(f"  • {r}" for r in reasons)
        + f"\n\nRECOMMENDED ACTIONS\n{'-'*50}\n"
        + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
        + f"\n\nCONFIDENTIALITY NOTICE\n{'-'*50}\n"
          "AI-assisted draft. All findings must be verified by qualified analyst "
          "personnel before any external submission.\n"
    )

    return {
        "case_id": case_id,
        "transaction_id": transaction_id,
        "customer_reference": str(txn_data.get("nameOrig", txn_data.get("sender", "Unknown"))),
        "risk_score": round(score_pct, 2),
        "risk_level": risk_level,
        "case_type": case_type,
        "status": "pending_review",
        "recommended_authorities": authorities,
        "human_review_required": True,
        "created_at": now,
        "last_action": "Case created — awaiting analyst review",
        "confidence_note": (
            "AI-generated draft based on available evidence. "
            "Analyst confirmation required before any external submission."
        ),
        "summary": summary,
        "reasons": reasons,
        "evidence": evidence_items,
        "timeline": timeline,
        "recommended_actions": actions,
        "narrative_report": narrative,
        "structured_report": {
            "case_id": case_id,
            "transaction_id": transaction_id,
            "report_type": case_type,
            "risk_score": round(score_pct, 2),
            "risk_level": risk_level,
            "report_to": authorities,
            "analyst_verification_required": True,
        },
        "audit": {
            "model_version": "v3.2.1",
            "prompt_version": "fraud-report-prompt-v3",
            "report_timestamp": now,
            "reviewer_decision": "",
            "reviewer_notes": "",
            "review_timestamp": "",
        },
    }


def _build_overall_analysis_case(
    scope: str,
    filters: dict,
    transactions_raw: list,
    txn_count: int,
) -> dict:
    now = datetime.utcnow().isoformat()
    case_id = _next_case_id()

    total = txn_count
    sample = transactions_raw

    flagged_count = sum(
        1 for t in sample
        if t.get("is_fraud") or safe_float(t.get("fraud_score", 0)) >= 0.5
    )
    legit_count = len(sample) - flagged_count

    if scope == "full_transaction_batch" and total > len(sample) and len(sample) > 0:
        ratio = flagged_count / len(sample)
        flagged_est = round(total * ratio)
        legit_est   = total - flagged_est
    else:
        flagged_est = flagged_count
        legit_est   = legit_count

    scores = [safe_float(t.get("fraud_score", 0)) for t in sample]
    scores_pct = [s * 100 if s <= 1 else s for s in scores]
    avg_score = sum(scores_pct) / len(scores_pct) if scores_pct else 0.0
    max_score = max(scores_pct) if scores_pct else 0.0

    risk_level = (
        "HIGH"       if avg_score >= 70 else
        "SUSPICIOUS" if avg_score >= 50 else
        "MEDIUM"     if avg_score >= 30 else "LOW"
    )

    scope_labels = {
        "full_transaction_batch": "Full Transaction Batch",
        "all_flagged":            "All Flagged Transactions",
        "high_risk":              "High-Risk Transactions",
        "medium_risk":            "Medium-Risk Transactions",
        "date_range":             "Date Range",
        "by_account":             "By Account / Customer",
        "by_risk_level":          "Custom Risk Level Filter",
    }
    scope_label = scope_labels.get(scope, scope.replace("_", " ").title())

    authorities = []
    case_type   = "overall_suspicious_activity_review"
    if avg_score >= 70 or max_score >= 85:
        authorities.append("DCI")
        case_type = "overall_high_risk_batch_review"
    if avg_score >= 50 or flagged_est > total * 0.3:
        authorities.append("FRC")
    if not authorities:
        authorities.append("Internal Review Only")
    authorities = list(dict.fromkeys(authorities))

    reasons = []
    if scope == "full_transaction_batch":
        reasons.append(f"Full dataset of {total} transactions submitted for operational analysis")
        if flagged_est > 0:
            reasons.append(f"{flagged_est} transaction(s) flagged as suspicious by the fraud model")
        if avg_score > 0:
            reasons.append(f"Average fraud model score across sample: {avg_score:.1f}%")
        if max_score >= 70:
            reasons.append(f"Highest individual score in sample: {max_score:.1f}%")
    else:
        reasons.append(f"Scope: {scope_label}")
        reasons.append(f"{len(sample)} transactions included in analysis sample")
        if avg_score > 0:
            reasons.append(f"Average risk score: {avg_score:.1f}%")

    evidence_items = [
        {"type": "batch_scope",    "label": "Analysis Scope",          "value": scope_label},
        {"type": "total_count",    "label": "Total Transactions",       "value": str(total)},
        {"type": "flagged_count",  "label": "Flagged / Suspicious",     "value": str(flagged_est)},
        {"type": "legit_count",    "label": "Legitimate / Non-Flagged", "value": str(legit_est)},
        {"type": "avg_score",      "label": "Avg Fraud Score (Sample)", "value": f"{avg_score:.1f}%"},
        {"type": "max_score",      "label": "Max Fraud Score (Sample)", "value": f"{max_score:.1f}%"},
        {"type": "sample_size",    "label": "Sample Sent to Backend",   "value": str(len(sample))},
    ]
    if filters.get("risk_level"):
        evidence_items.append({"type": "filter", "label": "Risk Level Filter", "value": filters["risk_level"]})
    if filters.get("date_from"):
        evidence_items.append({"type": "filter", "label": "Date From", "value": filters["date_from"]})
    if filters.get("date_to"):
        evidence_items.append({"type": "filter", "label": "Date To",   "value": filters["date_to"]})
    if filters.get("account"):
        evidence_items.append({"type": "filter", "label": "Account Filter", "value": filters["account"]})

    timeline = [
        {"timestamp": now, "event": "batch_submitted",
         "description": f"Analyst submitted {scope_label} for overall analysis"},
        {"timestamp": now, "event": "case_created",
         "description": "Overall analysis case created and queued for review"},
    ]

    if scope == "full_transaction_batch":
        summary_fallback = (
            f"This overall analysis case covers the complete uploaded transaction dataset of "
            f"{total} records. Of the sampled transactions, {flagged_est} were flagged as "
            f"suspicious and {legit_est} appear legitimate. The average fraud model score across "
            f"the sample is {avg_score:.1f}%. This case is recommended for analyst review to "
            f"assess whether suspicious activity represents isolated incidents or a broader pattern."
        )
    else:
        summary_fallback = (
            f"This overall analysis case covers {len(sample)} transactions matching the "
            f"'{scope_label}' scope. The average fraud model score is {avg_score:.1f}% with a "
            f"maximum of {max_score:.1f}%. Human analyst review is required before any external "
            f"reporting decision."
        )

    summary = summary_fallback
    if openai_client:
        prompt = f"""Write a 2-3 sentence professional fraud operations case summary for a BATCH analysis.
Scope: {scope_label} | Total Transactions: {total} | Flagged: {flagged_est} | Legitimate: {legit_est}
Avg Score: {avg_score:.1f}% | Max Score: {max_score:.1f}% | Risk Level: {risk_level}
Routing: {', '.join(authorities)}
Use compliance-safe language. Never say 'confirmed fraud'. Return only the paragraph."""
        ai_text = call_openai_chat(prompt, fallback=summary_fallback)
        if ai_text:
            summary = ai_text

    narrative = (
        f"OVERALL ANALYSIS CASE REPORT\n{'=' * 50}\n"
        f"Case ID:            {case_id}\n"
        f"Analysis Scope:     {scope_label}\n"
        f"Total Transactions: {total}\n"
        f"Flagged:            {flagged_est}\n"
        f"Legitimate:         {legit_est}\n"
        f"Avg Risk Score:     {avg_score:.1f}%\n"
        f"Max Risk Score:     {max_score:.1f}%\n"
        f"Overall Risk Level: {risk_level}\n"
        f"Routing:            {', '.join(authorities)}\n"
        f"Date:               {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"SUMMARY\n{'-' * 50}\n{summary}\n\n"
        f"KEY OBSERVATIONS\n{'-' * 50}\n"
        + "\n".join(f"  • {r}" for r in reasons)
        + f"\n\nCONFIDENTIALITY NOTICE\n{'-' * 50}\n"
          "AI-assisted draft. All findings must be verified by qualified analyst "
          "personnel before any external submission.\n"
    )

    return {
        "case_id":               case_id,
        "transaction_id":        f"BATCH-{scope.upper()}-{case_id}",
        "customer_reference":    f"Batch analysis — {scope_label}",
        "analysis_mode":         "overall_analysis",
        "scope":                 scope,
        "filters":               filters,
        "risk_score":            round(avg_score, 2),
        "risk_level":            risk_level,
        "case_type":             case_type,
        "status":                "pending_review",
        "recommended_authorities": authorities,
        "human_review_required": True,
        "created_at":            now,
        "last_action":           f"Overall analysis case created — {scope_label}",
        "confidence_note": (
            "AI-generated draft from overall batch analysis. "
            "Analyst confirmation required before any external submission."
        ),
        "summary":   summary,
        "reasons":   reasons,
        "evidence":  evidence_items,
        "timeline":  timeline,
        "recommended_actions": [
            f"Review overall risk distribution across {total} transactions",
            "Identify highest-scoring individual transactions for deeper investigation",
            "Determine whether suspicious activity is isolated or part of a broader pattern",
            "Document analyst observations before any escalation decision",
            "Obtain compliance approval before any external reporting",
        ],
        "narrative_report": narrative,
        "structured_report": {
            "case_id":       case_id,
            "scope":         scope,
            "report_type":   case_type,
            "total_count":   total,
            "flagged_count": flagged_est,
            "legit_count":   legit_est,
            "avg_risk_score": round(avg_score, 2),
            "risk_level":    risk_level,
            "report_to":     authorities,
            "analyst_verification_required": True,
        },
        "audit": {
            "model_version":    "v3.2.1",
            "prompt_version":   "fraud-report-prompt-v3",
            "report_timestamp": now,
            "reviewer_decision": "",
            "reviewer_notes":   "",
            "review_timestamp": "",
        },
    }


# ═════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({"success": True, "status": "FraudGuard API running", "model_loaded": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "service": "fraudguard-ml",
        "openai_configured": bool(openai_client),
        "openai_model": OPENAI_MODEL,
    })

# ─────────────────────────────────────────────
# Auth — Register
# ─────────────────────────────────────────────
@app.route("/register", methods=["POST"])
def register_user():
    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    password = data.get("password")
    name = data.get("name", "")
    role = data.get("role", "user")

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400

    if users_col.find_one({"email": email}):
        return jsonify({"success": False, "error": "User already exists"}), 400

    users_col.insert_one({
        "email": email,
        "name": name or email.split("@")[0],
        "password": generate_password_hash(password),
        "role": role,
        "is_active": True,
        "login_attempts": [],
        "created_at": datetime.utcnow(),
    })
    log_admin_action("register_user", {"email": email, "role": role})
    return jsonify({"success": True, "message": "User registered successfully"}), 201

# ─────────────────────────────────────────────
# Auth — Login
# ─────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login_user():
    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    password = data.get("password")

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

    if not check_password_hash(user["password"], password):
        users_col.update_one({"email": email},
            {"$push": {"login_attempts": {"status": "failed", "timestamp": datetime.utcnow()}}})
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

    if not user.get("is_active", True):
        return jsonify({"success": False, "error": "Account is deactivated"}), 403

    token = create_session_token(user)
    users_col.update_one({"email": email},
        {"$push": {"login_attempts": {"status": "success", "timestamp": datetime.utcnow()}}})
    log_admin_action("login", {"email": email})

    return jsonify({
        "success": True,
        "message": "Login successful",
        "session_token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user.get("name", email.split("@")[0]),
            "role": user.get("role", "user"),
            "is_active": user.get("is_active", True),
        },
    }), 200

# ─────────────────────────────────────────────
# Auth — OTP routes
# ─────────────────────────────────────────────
@app.route("/request-otp", methods=["POST"])
def request_otp():
    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    if not email:
        return jsonify({"success": False, "error": "Email required"}), 400
    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    otp = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    users_col.update_one({"email": email}, {"$set": {"otp_code": otp, "otp_expiry": expiry}})
    if send_email_otp(email, otp):
        return jsonify({"success": True, "message": f"OTP sent to {email}"}), 200
    return jsonify({"success": False, "error": "Failed to send OTP"}), 500


@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    otp   = normalize_string(data.get("otp"))
    if not email or not otp:
        return jsonify({"success": False, "error": "Email and OTP required"}), 400
    user = users_col.find_one({"email": email})
    if not user or "otp_code" not in user:
        return jsonify({"success": False, "error": "No OTP found — request a new one"}), 400
    if datetime.utcnow() > user.get("otp_expiry", datetime.utcnow()):
        return jsonify({"success": False, "error": "OTP expired — request a new one"}), 400
    if otp != user["otp_code"]:
        return jsonify({"success": False, "error": "Invalid OTP"}), 400
    users_col.update_one({"email": email}, {"$unset": {"otp_code": "", "otp_expiry": ""}})
    token = create_session_token(user)
    return jsonify({
        "success": True,
        "message": "OTP verified — login successful",
        "session_token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user.get("name", email.split("@")[0]),
            "role": user.get("role", "user"),
            "is_active": user.get("is_active", True),
        },
    }), 200


@app.route("/login/verify", methods=["POST"])
def login_verify_alias():
    data = request.get_json(silent=True) or {}
    otp = data.get("otp_code") or data.get("otp")
    temp_token = data.get("temp_token") or data.get("email")
    request._cached_json = ({"email": temp_token, "otp": otp}, True)
    return verify_otp()


@app.route("/login/resend", methods=["POST"])
def login_resend_alias():
    data = request.get_json(silent=True) or {}
    email = data.get("temp_token") or data.get("email")
    request._cached_json = ({"email": email}, True)
    return request_otp()


@app.route("/login/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    if not email:
        return jsonify({"success": False, "error": "Email required"}), 400
    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"success": True, "message": "If this email exists, a reset code has been sent"}), 200
    otp = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=15)
    users_col.update_one({"email": email},
        {"$set": {"reset_otp": otp, "reset_otp_expiry": expiry}})
    send_email_otp(email, otp)
    return jsonify({"success": True, "message": "If this email exists, a reset code has been sent"}), 200


@app.route("/login/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email    = normalize_string(data.get("email"))
    otp      = normalize_string(data.get("otp_code"))
    new_pass = data.get("new_password")
    if not email or not otp or not new_pass:
        return jsonify({"success": False, "error": "Email, otp_code, and new_password required"}), 400
    user = users_col.find_one({"email": email})
    if not user or "reset_otp" not in user:
        return jsonify({"success": False, "error": "Invalid or expired reset code"}), 400
    if datetime.utcnow() > user.get("reset_otp_expiry", datetime.utcnow()):
        return jsonify({"success": False, "error": "Reset code expired"}), 400
    if otp != user["reset_otp"]:
        return jsonify({"success": False, "error": "Invalid reset code"}), 400
    users_col.update_one({"email": email}, {
        "$set":   {"password": generate_password_hash(new_pass)},
        "$unset": {"reset_otp": "", "reset_otp_expiry": ""},
    })
    return jsonify({"success": True, "message": "Password reset successfully"}), 200


@app.route("/me", methods=["GET"])
@require_auth
def get_me():
    u = g.current_user
    return jsonify({
        "success": True,
        "user": {
            "id": u.get("user_id"),
            "email": u.get("email"),
            "name": u.get("name"),
            "role": u.get("role"),
            "is_active": u.get("is_active", True),
        },
    }), 200


@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else None
    if token:
        sessions_col.delete_one({"token": token})
    return jsonify({"success": True, "message": "Logged out"}), 200

# ═════════════════════════════════════════════
# ML — /predict  (hardened, batch-first)
# ═════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    t_request_start = time.time()
    data = request.get_json(silent=True) or {}
    t_parse = time.time()
    log.info(f"[/predict] request parse: {t_parse - t_request_start:.3f}s")

    transactions_list = data.get("transactions")
    if not isinstance(transactions_list, list) or len(transactions_list) == 0:
        return jsonify({"success": False, "error": "'transactions' must be a non-empty list"}), 400

    batch_id = new_batch_id()
    log.info(f"[/predict] batch_id={batch_id}, rows={len(transactions_list)}")

    try:
        # ── Build DataFrame ──────────────────────────────────────────────
        ID_COL = "__txn_id__"
        for i, txn in enumerate(transactions_list):
            if not isinstance(txn, dict):
                return jsonify({"success": False,
                                "error": "Each transaction must be an object",
                                "stage": "request_parse"}), 400
            txn[ID_COL] = txn.get("transaction_id") or txn.get("id") or f"TXN_{i+1}"

        t_df_start = time.time()
        df = pd.DataFrame(transactions_list).set_index(ID_COL)
        log.info(f"[/predict] DataFrame built in {time.time()-t_df_start:.3f}s, shape={df.shape}")

        # ── Predict ──────────────────────────────────────────────────────
        t_pred_start = time.time()
        try:
            results = predict_internal(df)
        except PipelineError as ve:
            stage = ve.stage
            log.error(f"[/predict] predict_internal failed at stage={stage}: {ve}")
            return jsonify({
                "success": False,
                "stage": stage,
                "message": str(ve),
                "batch_id": batch_id,
            }), 422
        t_pred_end = time.time()
        log.info(f"[/predict] prediction done in {t_pred_end - t_pred_start:.3f}s")

        # ── Attach results + build Mongo docs ────────────────────────────
        response_data = []
        mongo_docs    = []
        now           = datetime.utcnow()

        for i in range(len(results)):
            txn_id = str(df.index[i])
            pred   = int(results.iloc[i]["prediction"])
            score  = float(results.iloc[i]["fraud_score"])
            risk   = str(results.iloc[i]["risk_level"])

            response_data.append({
                "transaction_id": txn_id,
                "prediction":     pred,
                "fraud_score":    json_safe_value(score),
                "risk_level":     risk,
            })

            record = {k: json_safe_value(v)
                      for k, v in transactions_list[i].items()
                      if k != ID_COL}
            record.update({
                "transaction_id": txn_id,
                "prediction":     pred,
                "fraud_score":    json_safe_value(score),
                "risk_level":     risk,
                "batch_id":       batch_id,
                "created_at":     now,
            })
            mongo_docs.append(record)

        # ── Batch Mongo writes ────────────────────────────────────────────
        t_mongo_start = time.time()
        inserted = insert_many_chunked(transactions_col, mongo_docs, MONGO_INSERT_CHUNK)
        t_mongo_end = time.time()
        log.info(f"[/predict] Mongo write: {inserted} docs in {t_mongo_end - t_mongo_start:.3f}s "
                 f"({MONGO_INSERT_CHUNK}-doc chunks)")

        # ── Batch stats + metadata ────────────────────────────────────────
        stats = compute_batch_stats(results)
        processing_time = round(t_mongo_end - t_request_start, 3)

        save_batch_metadata(batch_id, {
            **stats,
            "source":               "predict_endpoint",
            "processing_time_seconds": processing_time,
            "created_at":           now,
        })

        log_admin_action("predict", {"batch_id": batch_id, "rows": len(transactions_list)})

        # ── Build response ────────────────────────────────────────────────
        t_resp_start = time.time()
        response = {
            "success":                  True,
            "batch_id":                 batch_id,
            "predictions":              response_data,   # always included for /predict
            **stats,
            "processing_time_seconds":  processing_time,
        }
        log.info(f"[/predict] response built in {time.time()-t_resp_start:.3f}s, "
                 f"total={processing_time}s")
        return jsonify(response), 200

    except Exception as e:
        log.error(f"[/predict] Unhandled error: {e}")
        log.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "stage":   "unhandled",
            "message": "Prediction failed",
            "detail":  str(e),
        }), 500

# ═════════════════════════════════════════════
# ML — /process-dataset  (hardened, batch-first)
# ═════════════════════════════════════════════
@app.route("/process-dataset", methods=["POST"])
def process_dataset():
    t_request_start = time.time()

    try:
        data = request.get_json(silent=True) or {}
        t_parse = time.time()
        log.info(f"[/process-dataset] request parse: {t_parse - t_request_start:.3f}s")

        csv_content = data.get("csv_content")
        file_name   = data.get("file_name", "dataset.csv")

        if not csv_content:
            return jsonify({"success": False, "error": "Missing 'csv_content'",
                            "stage": "request_parse"}), 400

        batch_id = new_batch_id()
        log.info(f"[/process-dataset] batch_id={batch_id}, file={file_name}")

        # ── Parse CSV ─────────────────────────────────────────────────────
        try:
            t_df_start = time.time()
            df = pd.read_csv(StringIO(csv_content))
            log.info(f"[/process-dataset] CSV parsed in {time.time()-t_df_start:.3f}s, "
                     f"shape={df.shape}")
        except Exception as exc:
            log.error(f"[/process-dataset] CSV parse failed: {exc}")
            return jsonify({"success": False, "stage": "csv_parse",
                            "message": f"CSV parse failed: {exc}"}), 422

        if df.empty:
            return jsonify({"success": False, "error": "CSV is empty",
                            "stage": "csv_parse"}), 400

        # Log dataframe diagnostics
        log.info(f"[/process-dataset] Columns: {list(df.columns)}")
        null_counts = df.isnull().sum()
        nonzero = null_counts[null_counts > 0]
        if not nonzero.empty:
            log.warning(f"[/process-dataset] Null counts: {nonzero.to_dict()}")
        log.info(f"[/process-dataset] Dtypes: {df.dtypes.to_dict()}")

        # ── Predict ────────────────────────────────────────────────────────
        t_pred_start = time.time()
        try:
            results = predict_internal(df)
        except PipelineError as ve:
            stage = ve.stage
            log.error(f"[/process-dataset] predict_internal failed at stage={stage}: {ve}")
            return jsonify({
                "success": False,
                "stage":   stage,
                "message": str(ve),
                "batch_id": batch_id,
            }), 422
        t_pred_end = time.time()
        log.info(f"[/process-dataset] prediction done in {t_pred_end - t_pred_start:.3f}s "
                 f"for {len(results)} rows")

        # ── Attach predictions ─────────────────────────────────────────────
        df = df.copy()
        df["prediction"]  = results["prediction"].values
        df["fraud_score"] = results["fraud_score"].values
        df["risk_level"]  = results["risk_level"].values
        df["batch_id"]    = batch_id

        if "transaction_id" not in df.columns:
            df["transaction_id"] = [f"TXN_{i+1}" for i in range(len(df))]

        # ── Compute stats ──────────────────────────────────────────────────
        stats = compute_batch_stats(results)

        # ── Build Mongo docs ───────────────────────────────────────────────
        now = datetime.utcnow()
        mongo_docs = []
        for row in df.to_dict(orient="records"):
            record = {k: json_safe_value(v) for k, v in row.items()}
            record["created_at"] = now
            mongo_docs.append(record)

        # ── Bulk Mongo insert in chunks ────────────────────────────────────
        t_mongo_start = time.time()
        inserted = insert_many_chunked(transactions_col, mongo_docs, MONGO_INSERT_CHUNK)
        t_mongo_end = time.time()
        log.info(f"[/process-dataset] Mongo write: {inserted} docs in "
                 f"{t_mongo_end - t_mongo_start:.3f}s ({MONGO_INSERT_CHUNK}-doc chunks)")

        # ── Save batch metadata ────────────────────────────────────────────
        processing_time = round(t_mongo_end - t_request_start, 3)
        save_batch_metadata(batch_id, {
            **stats,
            "file_name":               file_name,
            "source":                  "process_dataset",
            "processing_time_seconds": processing_time,
            "created_at":              now,
        })

        log_admin_action("process_dataset", {
            "batch_id":  batch_id,
            "file_name": file_name,
            "rows":      inserted,
        })

        # ── Build response (summary only — no full dataset in response) ────
        t_resp_start = time.time()
        response = {
            "success":                  True,
            "batch_id":                 batch_id,
            **stats,
            "file_name":                file_name,
            "processing_time_seconds":  processing_time,
            # Retrieval instructions for the frontend
            "results_url":              f"/batches/{batch_id}/transactions",
            "summary_url":              f"/batches/{batch_id}/summary",
        }
        log.info(f"[/process-dataset] response built in {time.time()-t_resp_start:.3f}s, "
                 f"total={processing_time}s")
        return jsonify(response), 200

    except Exception as e:
        log.error(f"[/process-dataset] Unhandled error: {e}")
        log.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "stage":   "unhandled",
            "message": "Dataset processing failed",
            "detail":  str(e),
        }), 500

# ═════════════════════════════════════════════
# Batch Retrieval Endpoints
# ═════════════════════════════════════════════

@app.route("/batches/<batch_id>/summary", methods=["GET"])
def batch_summary(batch_id):
    """
    GET /batches/<batch_id>/summary

    Returns the batch metadata + summary counts.
    """
    try:
        meta = batches_col.find_one({"batch_id": batch_id})
        if not meta:
            return jsonify({"success": False, "error": "Batch not found"}), 404
        return jsonify({"success": True, "batch": serialize_document(meta)}), 200
    except Exception as e:
        log.exception(f"[batch_summary] failed for {batch_id}")
        return jsonify({"success": False, "error": "Failed to load batch summary"}), 500


@app.route("/batches/<batch_id>/transactions", methods=["GET"])
def batch_transactions(batch_id):
    """
    GET /batches/<batch_id>/transactions

    Query parameters:
      page            (int, default 1)
      limit           (int, default 100, max 1000)
      suspicious_only (bool string "true"/"false", default false)
      risk_level      ("HIGH" | "MEDIUM" | "LOW")

    Returns paginated transaction documents for the batch.
    All rows are always available — use pagination to retrieve them.
    """
    try:
        page  = max(1, safe_int(request.args.get("page", 1), 1))
        limit = min(1000, max(1, safe_int(request.args.get("limit", 100), 100)))
        skip  = (page - 1) * limit

        query: dict = {"batch_id": batch_id}

        suspicious_only = request.args.get("suspicious_only", "").lower() == "true"
        if suspicious_only:
            query["prediction"] = 1

        risk_level = request.args.get("risk_level", "").upper()
        if risk_level in {"HIGH", "MEDIUM", "LOW"}:
            query["risk_level"] = risk_level

        total = transactions_col.count_documents(query)
        docs  = list(
            transactions_col.find(query)
            .sort("fraud_score", DESCENDING)
            .skip(skip)
            .limit(limit)
        )

        return jsonify({
            "success":    True,
            "batch_id":   batch_id,
            "page":       page,
            "limit":      limit,
            "total":      total,
            "pages":      math.ceil(total / limit) if limit else 1,
            "transactions": serialize_documents(docs),
        }), 200
    except Exception as e:
        log.exception(f"[batch_transactions] failed for {batch_id}")
        return jsonify({"success": False, "error": "Failed to load transactions"}), 500


@app.route("/batches/<batch_id>/all", methods=["GET"])
def batch_all(batch_id):
    """
    GET /batches/<batch_id>/all

    Returns ALL transactions in the batch (no pagination).
    Use carefully — intended for exports and AI assistant bulk access.
    Enforces a hard cap of 10,000 rows to protect memory.
    """
    try:
        CAP = 10_000
        docs = list(
            transactions_col.find({"batch_id": batch_id})
            .sort("fraud_score", DESCENDING)
            .limit(CAP)
        )
        total = transactions_col.count_documents({"batch_id": batch_id})
        return jsonify({
            "success":      True,
            "batch_id":     batch_id,
            "total":        total,
            "returned":     len(docs),
            "capped":       total > CAP,
            "transactions": serialize_documents(docs),
        }), 200
    except Exception as e:
        log.exception(f"[batch_all] failed for {batch_id}")
        return jsonify({"success": False, "error": "Failed to load batch"}), 500


@app.route("/batches/<batch_id>/report", methods=["GET"])
def batch_report(batch_id):
    """
    GET /batches/<batch_id>/report

    Returns a structured summary report for the batch including:
    - metadata / stats
    - top-10 highest scoring suspicious transactions
    - risk distribution breakdown
    """
    try:
        meta = batches_col.find_one({"batch_id": batch_id})
        if not meta:
            return jsonify({"success": False, "error": "Batch not found"}), 404

        # Top 10 suspicious
        top_flagged = list(
            transactions_col.find({"batch_id": batch_id, "prediction": 1})
            .sort("fraud_score", DESCENDING)
            .limit(10)
        )

        # Risk distribution
        pipeline = [
            {"$match": {"batch_id": batch_id}},
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}},
        ]
        dist_raw = list(transactions_col.aggregate(pipeline))
        risk_distribution = {d["_id"]: d["count"] for d in dist_raw if d.get("_id")}

        meta_safe = serialize_document(meta) or {}
        return jsonify({
            "success":           True,
            "batch_id":          batch_id,
            "summary":           meta_safe,
            "risk_distribution": risk_distribution,
            "top_flagged_transactions": serialize_documents(top_flagged),
        }), 200
    except Exception as e:
        log.exception(f"[batch_report] failed for {batch_id}")
        return jsonify({"success": False, "error": "Failed to generate batch report"}), 500


# ─────────────────────────────────────────────
# AI — Explain / Report / Bundle
# ─────────────────────────────────────────────
@app.route("/explain/<transaction_id>", methods=["GET"])
def explain_transaction(transaction_id):
    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"success": False, "error": "Transaction not found"}), 404
        cached = get_cached_ai_result(transaction_id, "explanation")
        if cached and isinstance(cached.get("payload"), dict):
            resp = {"success": True, "cached": True}
            resp.update(cached["payload"])  # type: ignore[arg-type]
            return jsonify(resp), 200
        result = generate_ai_transaction_explanation(txn)
        save_cached_ai_result(transaction_id, "explanation", result)
        resp = {"success": True, "cached": False}
        resp.update(result)
        return jsonify(resp), 200
    except Exception as e:
        log.exception("Explain failed")
        return jsonify({"success": False, "error": "Failed to explain transaction"}), 500


@app.route("/report/<transaction_id>", methods=["GET"])
def report_transaction(transaction_id):
    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"success": False, "error": "Transaction not found"}), 404
        cached = get_cached_ai_result(transaction_id, "report")
        if cached and isinstance(cached.get("payload"), dict):
            resp = {"success": True, "cached": True}
            resp.update(cached["payload"])  # type: ignore[arg-type]
            return jsonify(resp), 200
        result = generate_ai_report(txn)
        save_cached_ai_result(transaction_id, "report", result)
        resp = {"success": True, "cached": False}
        resp.update(result)
        return jsonify(resp), 200
    except Exception as e:
        log.exception("Report failed")
        return jsonify({"success": False, "error": "Failed to generate report"}), 500


@app.route("/ai-case/<transaction_id>", methods=["GET"])
def ai_case_bundle(transaction_id):
    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"success": False, "error": "Transaction not found"}), 404
        cached = get_cached_ai_result(transaction_id, "case_bundle")
        if cached and isinstance(cached.get("payload"), dict):
            resp = {"success": True, "cached": True}
            resp.update(cached["payload"])  # type: ignore[arg-type]
            return jsonify(resp), 200
        result = generate_ai_case_bundle(txn)
        save_cached_ai_result(transaction_id, "case_bundle", result)
        resp = {"success": True, "cached": False}
        resp.update(result)
        return jsonify(resp), 200
    except Exception as e:
        log.exception("AI case bundle failed")
        return jsonify({"success": False, "error": "Failed to generate AI case bundle"}), 500

# ─────────────────────────────────────────────
# Admin — Users
# ─────────────────────────────────────────────
@app.route("/admin/users", methods=["GET", "POST"])
def admin_users():
    try:
        if request.method == "GET":
            users = list(users_col.find().sort("created_at", -1))
            safe_users = []
            for u in users:
                d = serialize_document(u) or {}
                d.pop("password", None)
                safe_users.append(d)
            return jsonify({"success": True, "users": safe_users}), 200

        data = request.get_json(silent=True) or {}
        email    = normalize_string(data.get("email"))
        password = data.get("password")
        name     = data.get("name", "")
        role     = data.get("role", "user")
        if not email or not password:
            return jsonify({"success": False, "error": "Email and password required"}), 400
        if users_col.find_one({"email": email}):
            return jsonify({"success": False, "error": "User already exists"}), 400
        users_col.insert_one({
            "email": email, "name": name or email.split("@")[0],
            "password": generate_password_hash(password),
            "role": role, "is_active": True,
            "login_attempts": [], "created_at": datetime.utcnow(),
        })
        log_admin_action("add_user", {"email": email, "role": role})
        return jsonify({"success": True, "message": "User added successfully"}), 201
    except Exception as e:
        log.exception("admin_users failed")
        return jsonify({"success": False, "error": "Admin users request failed"}), 500


@app.route("/admin/users/<user_id>", methods=["DELETE", "PUT"])
def admin_user_detail(user_id):
    try:
        from bson import ObjectId
        try:
            oid = ObjectId(user_id)
        except Exception:
            return jsonify({"success": False, "error": "Invalid user ID"}), 400

        if request.method == "DELETE":
            result = users_col.delete_one({"_id": oid})
            if result.deleted_count == 0:
                return jsonify({"success": False, "error": "User not found"}), 404
            log_admin_action("delete_user", {"user_id": user_id})
            return jsonify({"success": True, "message": "User deleted"}), 200

        data = request.get_json(silent=True) or {}
        is_active = data.get("is_active", True)
        result = users_col.update_one({"_id": oid}, {"$set": {"is_active": is_active}})
        if result.matched_count == 0:
            return jsonify({"success": False, "error": "User not found"}), 404
        log_admin_action("toggle_user_status", {"user_id": user_id, "is_active": is_active})
        return jsonify({"success": True, "message": "User status updated"}), 200
    except Exception as e:
        log.exception("admin_user_detail failed")
        return jsonify({"success": False, "error": "Operation failed"}), 500


@app.route("/admin/users/<user_id>/status", methods=["PUT"])
def admin_user_status(user_id):
    return admin_user_detail(user_id)


@app.route("/admin/transactions", methods=["GET"])
def admin_transactions():
    try:
        limit = safe_int(request.args.get("limit", 100), 100)
        txns = list(transactions_col.find().sort("created_at", -1).limit(limit))
        return jsonify({"success": True, "transactions": serialize_documents(txns)}), 200
    except Exception as e:
        log.exception("admin_transactions failed")
        return jsonify({"success": False, "error": "Failed to load transactions"}), 500


@app.route("/admin/logs", methods=["GET"])
def admin_logs():
    try:
        limit = safe_int(request.args.get("limit", 100), 100)
        logs = list(admin_col.find().sort("timestamp", -1).limit(limit))
        return jsonify({"success": True, "logs": serialize_documents(logs)}), 200
    except Exception as e:
        log.exception("admin_logs failed")
        return jsonify({"success": False, "error": "Failed to load logs"}), 500


@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    try:
        stats = {
            "total_users":          users_col.count_documents({}),
            "total_transactions":   transactions_col.count_documents({}),
            "total_logs":           admin_col.count_documents({}),
            "flagged_transactions": transactions_col.count_documents({"prediction": 1}),
            "ai_cached_items":      ai_cache_col.count_documents({}),
            "analyst_cases":        analyst_cases_col.count_documents({}),
            "total_batches":        batches_col.count_documents({}),
        }
        return jsonify({"success": True, "stats": stats}), 200
    except Exception as e:
        log.exception("admin_stats failed")
        return jsonify({"success": False, "error": "Failed to load stats"}), 500

# ─────────────────────────────────────────────
# Analyst — Cases  GET / POST
# ─────────────────────────────────────────────
@app.route("/analyst/cases", methods=["GET", "POST"])
def analyst_cases():
    if request.method == "GET":
        try:
            cases = list(
                analyst_cases_col
                .find({"status": {"$ne": "resolved"}})
                .sort("created_at", -1)
                .limit(200)
            )
            return jsonify({"success": True, "cases": [_serialize_case(c) for c in cases]}), 200
        except Exception as e:
            log.exception("GET /analyst/cases failed")
            return jsonify({"success": False, "error": "Failed to fetch cases"}), 500

    try:
        data = request.get_json(silent=True) or {}
        analysis_mode = data.get("analysis_mode", "single_transaction")

        if analysis_mode == "overall_analysis":
            scope            = data.get("scope", "all_flagged")
            filters          = data.get("filters", {})
            transactions_raw = data.get("transactions", [])
            txn_count        = data.get("transaction_count", len(transactions_raw))

            case = _build_overall_analysis_case(scope, filters, transactions_raw, txn_count)
            analyst_cases_col.insert_one(case)
            log_admin_action("create_overall_analysis_case", {
                "case_id": case["case_id"],
                "scope": scope,
                "transaction_count": txn_count,
            })
            return jsonify({"success": True, "case": _serialize_case(case)}), 201

        transaction_id = data.get("transaction_id")
        txn_data       = data.get("transaction", {})
        if not transaction_id:
            return jsonify({"success": False, "error": "transaction_id is required"}), 400

        existing = analyst_cases_col.find_one({"transaction_id": transaction_id})
        if existing:
            return jsonify({"success": True, "case": _serialize_case(existing)}), 200

        case = _build_analyst_case(transaction_id, txn_data)
        analyst_cases_col.insert_one(case)
        log_admin_action("create_analyst_case", {
            "case_id": case["case_id"],
            "transaction_id": transaction_id,
        })
        return jsonify({"success": True, "case": _serialize_case(case)}), 201

    except Exception as e:
        log.exception("POST /analyst/cases failed")
        return jsonify({"success": False, "error": "Failed to create case", "detail": str(e)}), 500

# ─────────────────────────────────────────────
# Analyst — Case Detail  GET / PUT / DELETE
# ─────────────────────────────────────────────
@app.route("/analyst/cases/<case_id>", methods=["GET", "PUT", "DELETE"])
def analyst_case_detail(case_id):
    try:
        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404

        if request.method == "GET":
            return jsonify({"success": True, "case": _serialize_case(case)}), 200

        if request.method == "DELETE":
            analyst_cases_col.update_one({"case_id": case_id}, {"$set": {"status": "resolved"}})
            log_admin_action("close_analyst_case", {"case_id": case_id})
            return jsonify({"success": True, "message": "Case closed"}), 200

        data = request.get_json(silent=True) or {}
        allowed_updates = {k: v for k, v in data.items()
                           if k in {"status", "last_action", "notes"}}
        if allowed_updates:
            analyst_cases_col.update_one({"case_id": case_id}, {"$set": allowed_updates})
        updated = analyst_cases_col.find_one({"case_id": case_id})
        return jsonify({"success": True, "case": _serialize_case(updated)}), 200
    except Exception as e:
        log.exception(f"analyst_case_detail failed for {case_id}")
        return jsonify({"success": False, "error": "Operation failed"}), 500

# ─────────────────────────────────────────────
# Analyst — Chat (AI Copilot)
# ─────────────────────────────────────────────
@app.route("/analyst/chat", methods=["POST"])
def analyst_chat():
    try:
        data = request.get_json(silent=True) or {}
        case_id  = data.get("case_id")
        question = (data.get("question") or "").strip()
        if not case_id or not question:
            return jsonify({"success": False, "error": "case_id and question required"}), 400

        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404

        case_safe = _serialize_case(case) or {}
        answer = _copilot_response(question, case_safe)

        return jsonify({
            "success": True,
            "case_id": case_id,
            "question": question,
            "response": answer,
            "timestamp": datetime.utcnow().isoformat(),
        }), 200
    except Exception as e:
        log.exception("analyst_chat failed")
        return jsonify({"success": False, "error": "Chat request failed"}), 500


def _copilot_response(question: str, case: dict) -> str:
    risk_score  = safe_float(case.get("risk_score", 0))
    risk_level  = case.get("risk_level", "UNKNOWN")
    reasons     = case.get("reasons", [])
    evidence    = case.get("evidence", [])
    authorities = case.get("recommended_authorities", [])
    case_type   = case.get("case_type", "unknown")

    if openai_client:
        prompt = f"""You are the FraudGuard Analyst AI Copilot.
Help the analyst investigate this case. Stay grounded in case data.
Never say the person is guilty. Use "flagged as suspicious", "possible fraud indicators".

CASE:
- Case ID: {case.get('case_id')} | Transaction: {case.get('transaction_id')}
- Risk: {risk_score:.1f}% ({risk_level}) | Type: {case_type}
- Routing: {', '.join(authorities)}
- Reasons: {', '.join(reasons[:5]) or 'N/A'}
- Evidence items: {len(evidence)}

ANALYST QUESTION: {question}

Answer in 3-5 sentences. Be professional, direct, and operationally useful."""
        answer = call_openai_chat(prompt)
        if answer:
            return answer

    q = question.lower()
    if any(w in q for w in ["dci", "why dci", "route"]):
        return (f"DCI routing is recommended because the fraud score of {risk_score:.1f}% "
                f"and case type '{case_type}' are consistent with possible cyber-enabled or "
                f"electronic fraud. DCI handles criminal investigations involving digital payment "
                f"abuse, account takeover, and identity theft. External referral requires analyst approval.")
    if any(w in q for w in ["frc", "aml", "suspicious transaction"]):
        return (f"FRC routing is recommended because transaction patterns align with AML "
                f"monitoring criteria. A risk score of {risk_score:.1f}% warrants a Suspicious "
                f"Transaction Report (STR). FRC submission requires analyst and compliance sign-off.")
    if any(w in q for w in ["strongest", "best evidence", "main evidence"]):
        top = "; ".join(f"{e.get('label')}: {e.get('value')}" for e in evidence[:3])
        return (f"Strongest evidence items: {top}. "
                f"The fraud model score of {risk_score:.1f}% is the primary quantitative signal. "
                f"Supporting indicators include transaction type and balance movement patterns.")
    if any(w in q for w in ["missing", "gaps", "what else"]):
        return ("Potentially missing evidence: device metadata, IP geolocation, OTP and login "
                "attempt logs, account age, and full transaction history. Request these from "
                "the IT/security team before finalising the case.")
    if any(w in q for w in ["account takeover", "ato", "takeover"]):
        level = "are consistent with" if risk_score >= 50 else "partially suggest"
        return (f"Available indicators {level} possible account takeover. "
                f"A risk score of {risk_score:.1f}% warrants investigation into device changes, "
                f"authentication events, and OTP history before drawing conclusions.")
    if any(w in q for w in ["confidence", "reliable", "accurate"]):
        conf = "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW"
        return (f"Current confidence level: {conf} (model score {risk_score:.1f}%). "
                f"This is based on {len(evidence)} evidence items. "
                f"Analyst validation is required — additional evidence would increase confidence.")
    if any(w in q for w in ["action", "next", "what should", "recommend"]):
        return (f"Recommended steps: (1) Review all evidence. (2) Check the fraud timeline. "
                f"(3) Verify customer activity patterns. (4) Document your observations. "
                f"(5) Record a human review decision. "
                f"External reporting to {', '.join(authorities)} requires analyst approval.")
    if any(w in q for w in ["preserve", "before escalation", "evidence preservation"]):
        return ("Before escalation, preserve: transaction logs, device metadata, OTP/auth logs, "
                "account balance snapshots, IP records, and linked transaction references. "
                "These are critical for any subsequent criminal or regulatory investigation.")
    if any(w in q for w in ["structuring", "smurfing", "layering"]):
        return ("No clear structuring pattern visible from this single transaction. "
                "To assess structuring, review multiple transactions from the same source over "
                "time and check for amounts just below reporting thresholds.")
    if any(w in q for w in ["internal", "internal review"]):
        msg = ("does not currently meet the threshold" if risk_score < 50
               else "may warrant escalation, but additional analyst review is needed")
        return (f"This case {msg} for external reporting at this stage. "
                f"Internal review is appropriate when evidence is incomplete. "
                f"Document your rationale in the analyst notes before closing.")
    return (f"This case has a risk score of {risk_score:.1f}% ({risk_level}) with "
            f"{len(reasons)} flagging indicators. "
            f"Primary reasons: {'; '.join(reasons[:3]) or 'standard model alert'}. "
            f"Routing: {', '.join(authorities)}. Ask about evidence strength, "
            f"authority routing, next actions, or fraud pattern analysis for specific guidance.")

# ─────────────────────────────────────────────
# Analyst — Review  POST
# ─────────────────────────────────────────────
@app.route("/analyst/review", methods=["POST"])
def analyst_review():
    try:
        data = request.get_json(silent=True) or {}
        case_id        = data.get("case_id")
        decision       = data.get("decision")
        reviewer_notes = data.get("reviewer_notes", "")
        reviewer_name  = data.get("reviewer_name", "Analyst")

        if not case_id or not decision:
            return jsonify({"success": False, "error": "case_id and decision are required"}), 400

        valid_decisions = ["approve", "reject", "escalate", "hold_internal",
                           "request_evidence", "mark_reviewed"]
        if decision not in valid_decisions:
            return jsonify({"success": False,
                            "error": f"Invalid decision. Must be one of: {valid_decisions}"}), 400

        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404

        now = datetime.utcnow().isoformat()
        status_map = {
            "approve":          "approved",
            "reject":           "rejected",
            "escalate":         "escalated",
            "hold_internal":    "internal_review",
            "request_evidence": "pending_evidence",
            "mark_reviewed":    "reviewed",
        }
        message_map = {
            "approve":          f"Case approved for escalation by {reviewer_name}",
            "reject":           f"Case rejected by {reviewer_name}",
            "escalate":         f"Case escalated by {reviewer_name}",
            "hold_internal":    f"Case held for internal review by {reviewer_name}",
            "request_evidence": f"Additional evidence requested by {reviewer_name}",
            "mark_reviewed":    f"Case marked as reviewed by {reviewer_name}",
        }

        update_fields = {
            "status":                    status_map.get(decision, case.get("status")),
            "last_action":               message_map.get(decision, decision),
            "audit.reviewer_decision":   decision,
            "audit.reviewer_notes":      reviewer_notes,
            "audit.review_timestamp":    now,
        }

        # ── FRC auto-submission on escalate or approve ────────────────────────
        frc_result = None
        if decision in ("escalate", "approve"):
            existing_frc_id = case.get("frc_case_id")
            if existing_frc_id:
                # Already submitted — skip re-submission
                frc_result = {
                    "success": True,
                    "frc_case_id": existing_frc_id,
                    "status": "already_submitted",
                    "message": f"Case already submitted to FRC as {existing_frc_id}.",
                }
            else:
                frc_result = submit_to_frc(case)
                if frc_result["success"]:
                    update_fields["frc_submission_status"] = "acknowledged"
                    update_fields["frc_case_id"]           = frc_result.get("frc_case_id")
                    update_fields["frc_submitted_at"]      = now
                else:
                    update_fields["frc_submission_status"] = "failed"
                    update_fields["frc_submission_error"]  = frc_result.get("error", "")

        analyst_cases_col.update_one({"case_id": case_id}, {"$set": update_fields})

        review_doc = {
            "case_id":          case_id,
            "decision":         decision,
            "reviewer_name":    reviewer_name,
            "reviewer_notes":   reviewer_notes,
            "review_timestamp": now,
        }
        analyst_reviews_col.insert_one(review_doc)
        log_admin_action("analyst_review", {"case_id": case_id, "decision": decision})

        updated = analyst_cases_col.find_one({"case_id": case_id})
        response = {
            "success": True,
            "message": message_map.get(decision),
            "review": {k: v for k, v in review_doc.items() if k != "_id"},
            "case": _serialize_case(updated),
        }
        if frc_result is not None:
            response["frc_submission"] = frc_result
        return jsonify(response), 200
    except Exception as e:
        log.exception("analyst_review failed")
        return jsonify({"success": False, "error": "Review submission failed"}), 500

# ─────────────────────────────────────────────
# Analyst — Review History  GET
# ─────────────────────────────────────────────
@app.route("/analyst/reviews/<case_id>", methods=["GET"])
def analyst_reviews(case_id):
    try:
        reviews = list(
            analyst_reviews_col.find({"case_id": case_id}).sort("review_timestamp", -1)
        )
        return jsonify({
            "success": True,
            "reviews": [{k: v for k, v in r.items() if k != "_id"} for r in reviews],
        }), 200
    except Exception as e:
        log.exception("analyst_reviews failed")
        return jsonify({"success": False, "error": "Failed to fetch reviews"}), 500

# ─────────────────────────────────────────────
# Analyst — Case Action Endpoints
# ─────────────────────────────────────────────
@app.route("/analyst/cases/<case_id>/export", methods=["POST"])
def analyst_export(case_id):
    try:
        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404
        fmt = (request.get_json(silent=True) or {}).get("format", "json")
        log_admin_action("export_case", {"case_id": case_id, "format": fmt})
        return jsonify({"success": True, "message": f"Export recorded ({fmt})", "case_id": case_id}), 200
    except Exception as e:
        log.exception("analyst_export failed")
        return jsonify({"success": False, "error": "Export failed"}), 500


@app.route("/analyst/cases/<case_id>/request-evidence", methods=["POST"])
def analyst_request_evidence(case_id):
    try:
        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404
        notes = (request.get_json(silent=True) or {}).get("notes", "")
        analyst_cases_col.update_one({"case_id": case_id}, {"$set": {
            "status": "pending_evidence",
            "last_action": f"Evidence requested: {notes[:100]}",
        }})
        log_admin_action("request_evidence", {"case_id": case_id})
        return jsonify({"success": True, "message": "Evidence request recorded"}), 200
    except Exception as e:
        log.exception("analyst_request_evidence failed")
        return jsonify({"success": False, "error": "Failed to request evidence"}), 500


@app.route("/analyst/cases/<case_id>/send-review", methods=["POST"])
def analyst_send_review(case_id):
    try:
        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404
        analyst_cases_col.update_one({"case_id": case_id}, {"$set": {
            "status": "under_review",
            "last_action": "Sent for compliance review",
        }})
        log_admin_action("send_for_review", {"case_id": case_id})
        return jsonify({"success": True, "message": "Case sent for review"}), 200
    except Exception as e:
        log.exception("analyst_send_review failed")
        return jsonify({"success": False, "error": "Failed to send for review"}), 500


@app.route("/analyst/cases/<case_id>/submit-to-frc", methods=["POST"])
def analyst_submit_to_frc(case_id):
    """
    Manually submit (or re-submit) a FraudGuard analyst case to the FRC backend.

    POST body (optional JSON):
      { "force": true }   — set force=true to re-submit even if already submitted.

    Returns:
      { success, frc_case_id, status, message, frc_submission }
    """
    try:
        case = analyst_cases_col.find_one({"case_id": case_id})
        if not case:
            return jsonify({"success": False, "error": "Case not found"}), 404

        data  = request.get_json(silent=True) or {}
        force = bool(data.get("force", False))

        existing_frc_id = case.get("frc_case_id")
        if existing_frc_id and not force:
            return jsonify({
                "success": True,
                "frc_case_id": existing_frc_id,
                "status": "already_submitted",
                "message": f"Case already submitted to FRC as {existing_frc_id}. Pass force=true to re-submit.",
                "frc_submission": {
                    "success": True,
                    "frc_case_id": existing_frc_id,
                    "status": "already_submitted",
                },
            }), 200

        # Perform submission
        frc_result = submit_to_frc(case)
        now = datetime.utcnow().isoformat()

        if frc_result["success"]:
            analyst_cases_col.update_one({"case_id": case_id}, {"$set": {
                "frc_submission_status": "acknowledged",
                "frc_case_id":           frc_result.get("frc_case_id"),
                "frc_submitted_at":      now,
                "status":                "escalated",
                "last_action":           f"Submitted to FRC: {frc_result.get('frc_case_id')}",
            }})
        else:
            analyst_cases_col.update_one({"case_id": case_id}, {"$set": {
                "frc_submission_status": "failed",
                "frc_submission_error":  frc_result.get("error", ""),
                "last_action":           f"FRC submission failed: {frc_result.get('error', '')[:80]}",
            }})

        log_admin_action("submit_to_frc", {
            "case_id":   case_id,
            "success":   frc_result["success"],
            "frc_case_id": frc_result.get("frc_case_id"),
        })

        updated = analyst_cases_col.find_one({"case_id": case_id})
        return jsonify({
            "success":     frc_result["success"],
            "frc_case_id": frc_result.get("frc_case_id"),
            "status":      frc_result.get("status"),
            "message":     frc_result.get("message"),
            "frc_submission": frc_result,
            "case": _serialize_case(updated),
        }), 200 if frc_result["success"] else 502

    except Exception as e:
        log.exception("analyst_submit_to_frc failed")
        return jsonify({"success": False, "error": "FRC submission endpoint error"}), 500

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
