"""
FraudGuard ML + Analyst AI Backend
====================================
Production-ready Flask backend for the FraudGuard Vercel frontend.

Covers:
  - Auth  : /login, /register, /request-otp, /verify-otp
              /login/verify, /login/resend (aliases for frontend compat)
              /login/forgot-password, /login/reset-password
  - ML    : /predict, /process-dataset, /explain/<id>, /report/<id>, /ai-case/<id>
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
"""

import os
import json
import math
import random
import logging
import smtplib
import hashlib
import uuid
from datetime import datetime, timedelta
from io import StringIO
from email.mime.text import MIMEText
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from pymongo import MongoClient
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
# Default to a real model that exists — gpt-4o-mini is cheap and capable
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

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
    sessions_col         = db["sessions"]          # NEW: persistent sessions
    analyst_cases_col    = db["analyst_cases"]     # NEW: analyst cases
    analyst_reviews_col  = db["analyst_reviews"]   # NEW: review history

    # Ensure useful indexes
    sessions_col.create_index("token", unique=True, background=True)
    sessions_col.create_index("expires_at", expireAfterSeconds=0, background=True)
    analyst_cases_col.create_index("case_id", unique=True, background=True)
    analyst_cases_col.create_index("transaction_id", background=True)
    analyst_reviews_col.create_index("case_id", background=True)
    transactions_col.create_index("transaction_id", background=True)

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
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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
# Token / Session Management  (MongoDB-backed)
# ─────────────────────────────────────────────
SESSION_TTL_HOURS = 24

def create_session_token(user_doc):
    """Create a persistent session token stored in MongoDB."""
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
    """Look up session in MongoDB. Returns session doc or None."""
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
    """Extract Bearer token from Authorization header and return session dict."""
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else None
    return get_session_from_token(token)


def require_auth(f):
    """Decorator: require valid session token."""
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
    """Decorator: require admin role."""
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

# ─────────────────────────────────────────────
# ML Helpers
# ─────────────────────────────────────────────
def predict_internal(new_data: pd.DataFrame):
    processed = pd.get_dummies(new_data, drop_first=True)
    for col in feature_cols:
        if col not in processed.columns:
            processed[col] = 0
    aligned = processed[feature_cols].astype(float)
    prob = model.predict_proba(aligned)[:, 1]
    pred = (prob >= 0.5).astype(int)
    risk = pd.Series(
        np.where(prob < 0.2, "LOW", np.where(prob < 0.8, "MEDIUM", "HIGH")),
        index=aligned.index,
    )
    return pd.DataFrame({"prediction": pred, "fraud_score": prob, "risk_level": risk},
                        index=aligned.index)

# ─────────────────────────────────────────────
# AI / OpenAI Helpers
# ─────────────────────────────────────────────
def call_openai_chat(prompt: str, fallback: str = "") -> str:
    """
    Call OpenAI Chat Completions API.
    Returns the text response or fallback string if unavailable.
    Uses the Chat Completions API (not the beta Responses API).
    """
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
    """Call OpenAI and parse JSON response. Falls back to fallback_payload."""
    if not openai_client:
        return fallback_payload
    clean = ""
    try:
        raw = call_openai_chat(prompt)
        if not raw:
            return fallback_payload
        # Strip markdown code fences if present
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
    """Build a full analyst case from transaction data + AI summary."""
    raw_score = safe_float(txn_data.get("fraud_score", 0))
    # Normalise to 0–100 range for storage (frontend shows percentage)
    score_pct = raw_score if raw_score > 1 else raw_score * 100
    # Keep 0–1 for derive helpers
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

    # Evidence items
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

    # Narrative summary — OpenAI if available
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
    """
    Build an analyst case for overall / bulk / full-batch analysis.

    Supports scopes:
      full_transaction_batch — entire uploaded dataset (flagged + legitimate)
      all_flagged            — only flagged/suspicious transactions
      high_risk / medium_risk / by_account / date_range / by_risk_level — subsets
    """
    now = datetime.utcnow().isoformat()
    case_id = _next_case_id()

    total = txn_count
    sample = transactions_raw  # up to 50 sent by frontend

    # Compute basic stats from the sample
    flagged_count = sum(
        1 for t in sample
        if t.get("is_fraud") or safe_float(t.get("fraud_score", 0)) >= 0.5
    )
    legit_count = len(sample) - flagged_count

    # For full-batch, extrapolate from sample proportion if we have more than sample
    if scope == "full_transaction_batch" and total > len(sample) and len(sample) > 0:
        ratio = flagged_count / len(sample)
        flagged_est = round(total * ratio)
        legit_est   = total - flagged_est
    else:
        flagged_est = flagged_count
        legit_est   = legit_count

    # Score stats from sample
    scores = [safe_float(t.get("fraud_score", 0)) for t in sample]
    scores_pct = [s * 100 if s <= 1 else s for s in scores]
    avg_score = sum(scores_pct) / len(scores_pct) if scores_pct else 0.0
    max_score = max(scores_pct) if scores_pct else 0.0

    # Determine overall risk level from average score
    risk_level = (
        "HIGH"       if avg_score >= 70 else
        "SUSPICIOUS" if avg_score >= 50 else
        "MEDIUM"     if avg_score >= 30 else "LOW"
    )

    # Scope-specific label and description
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

    # Build case type and authorities based on overall risk
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

    # Reasons
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

    # Evidence items
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

    # Timeline
    timeline = [
        {"timestamp": now, "event": "batch_submitted",
         "description": f"Analyst submitted {scope_label} for overall analysis"},
        {"timestamp": now, "event": "case_created",
         "description": "Overall analysis case created and queued for review"},
    ]

    # Narrative summary — OpenAI if available, rule-based fallback
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
# Auth — Login  (returns session_token)
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
# Auth — OTP routes (original paths)
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

# ─────────────────────────────────────────────
# Auth — Frontend alias paths
# The frontend calls /login/verify, /login/resend, etc.
# ─────────────────────────────────────────────
@app.route("/login/verify", methods=["POST"])
def login_verify_alias():
    """Alias: frontend calls /login/verify with {temp_token, otp_code}."""
    data = request.get_json(silent=True) or {}
    # Accept either format
    otp = data.get("otp_code") or data.get("otp")
    temp_token = data.get("temp_token") or data.get("email")
    # temp_token here is actually the email in the frontend flow
    request._cached_json = ({"email": temp_token, "otp": otp}, True)
    return verify_otp()


@app.route("/login/resend", methods=["POST"])
def login_resend_alias():
    """Alias: frontend calls /login/resend with {temp_token}."""
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
        # Don't leak whether user exists
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

# ─────────────────────────────────────────────
# ML — Predict
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(silent=True) or {}
    transactions_list = data.get("transactions")

    if not isinstance(transactions_list, list) or len(transactions_list) == 0:
        return jsonify({"success": False, "error": "'transactions' must be a non-empty list"}), 400

    try:
        ID_COL = "__txn_id__"
        for i, txn in enumerate(transactions_list):
            if not isinstance(txn, dict):
                return jsonify({"success": False, "error": "Each transaction must be an object"}), 400
            txn[ID_COL] = txn.get("transaction_id") or txn.get("id") or f"TXN_{i+1}"

        df = pd.DataFrame(transactions_list).set_index(ID_COL)
        results = predict_internal(df)
        response_data = []

        for i in range(len(results)):
            txn_id = str(df.index[i])
            item = {
                "transaction_id": txn_id,
                "prediction": int(results.iloc[i]["prediction"]),
                "fraud_score": float(results.iloc[i]["fraud_score"]),
                "risk_level": str(results.iloc[i]["risk_level"]),
            }
            response_data.append(item)
            record = transactions_list[i].copy()
            record.pop(ID_COL, None)
            record.update(item)
            record["created_at"] = datetime.utcnow()
            transactions_col.insert_one(record)

        return jsonify({"success": True, "predictions": response_data}), 200
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"success": False, "error": "Prediction failed", "detail": str(e)}), 500

# ─────────────────────────────────────────────
# ML — Process Dataset
# ─────────────────────────────────────────────
@app.route("/process-dataset", methods=["POST"])
def process_dataset():
    try:
        data = request.get_json(silent=True) or {}
        csv_content = data.get("csv_content")
        file_name = data.get("file_name", "dataset.csv")
        if not csv_content:
            return jsonify({"success": False, "error": "Missing 'csv_content'"}), 400
        df = pd.read_csv(StringIO(csv_content))
        if df.empty:
            return jsonify({"success": False, "error": "CSV is empty"}), 400
        results = predict_internal(df)
        df["prediction"]  = results["prediction"].values
        df["fraud_score"] = results["fraud_score"].values
        df["risk_level"]  = results["risk_level"].values
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [f"TXN_{i+1}" for i in range(len(df))]
        inserted = 0
        for _, row in df.iterrows():
            record = {k: json_safe_value(v) for k, v in row.to_dict().items()}
            record["created_at"] = datetime.utcnow()
            transactions_col.insert_one(record)
            inserted += 1
        log_admin_action("process_dataset", {"file_name": file_name, "rows": inserted})
        return jsonify({
            "success": True,
            "message": f"{inserted} transactions processed",
            "file_name": file_name,
            "predictions": serialize_documents(df.to_dict(orient="records")),
        }), 200
    except Exception as e:
        log.exception("Dataset processing failed")
        return jsonify({"success": False, "error": "Failed to process dataset", "detail": str(e)}), 500

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
            return jsonify({"success": True, "cached": True, **cached["payload"]}), 200
        result = generate_ai_transaction_explanation(txn)
        save_cached_ai_result(transaction_id, "explanation", result)
        return jsonify({"success": True, "cached": False, **result}), 200
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
            return jsonify({"success": True, "cached": True, **cached["payload"]}), 200
        result = generate_ai_report(txn)
        save_cached_ai_result(transaction_id, "report", result)
        return jsonify({"success": True, "cached": False, **result}), 200
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
            return jsonify({"success": True, "cached": True, **cached["payload"]}), 200
        result = generate_ai_case_bundle(txn)
        save_cached_ai_result(transaction_id, "case_bundle", result)
        return jsonify({"success": True, "cached": False, **result}), 200
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
                d.pop("password", None)  # never expose password hash
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

        # PUT — toggle status
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
    """Alias matching frontend call pattern /admin/users/:id/status."""
    return admin_user_detail(user_id)

# ─────────────────────────────────────────────
# Admin — Transactions / Logs / Stats
# ─────────────────────────────────────────────
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

    # ── POST — create case ───────────────────────────────────────────────────
    try:
        data = request.get_json(silent=True) or {}
        analysis_mode = data.get("analysis_mode", "single_transaction")

        # ── Overall / Bulk / Full-batch analysis ────────────────────────────
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

        # ── Single transaction analysis (default) ───────────────────────────
        transaction_id = data.get("transaction_id")
        txn_data       = data.get("transaction", {})
        if not transaction_id:
            return jsonify({"success": False, "error": "transaction_id is required"}), 400

        # Deduplicate: return existing case if one already exists for this txn
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

        # PUT — partial update
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

    # Rule-based fallback
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
    # Generic
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

        analyst_cases_col.update_one({"case_id": case_id}, {"$set": {
            "status":                    status_map.get(decision, case.get("status")),
            "last_action":               message_map.get(decision, decision),
            "audit.reviewer_decision":   decision,
            "audit.reviewer_notes":      reviewer_notes,
            "audit.review_timestamp":    now,
        }})

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
        return jsonify({
            "success": True,
            "message": message_map.get(decision),
            "review": {k: v for k, v in review_doc.items() if k != "_id"},
            "case": _serialize_case(updated),
        }), 200
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

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
