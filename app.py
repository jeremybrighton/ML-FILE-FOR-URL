import os
import json
import math
import random
import logging
import smtplib
from datetime import datetime, timedelta
from io import StringIO
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://fraud-detector-b.vercel.app",
    "https://fraud-detector-topaz.vercel.app"
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# -----------------------------
# Environment Variables
# -----------------------------
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set!")

SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
OTP_EXPIRY_MINUTES = int(os.environ.get("OTP_EXPIRY_MINUTES", 5))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

PROJECT_PATH = "."
MODEL_PATH = os.path.join(PROJECT_PATH, "rf_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PROJECT_PATH, "feature_columns.json")

# -----------------------------
# OpenAI Client
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------------
# MongoDB Connection
# -----------------------------
try:
    mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"), tls=True)
    db = mongo_client["fraud_detection"]

    users_col = db["users"]
    transactions_col = db["transactions"]
    admin_col = db["admin_actions"]
    ai_cache_col = db["ai_cache"]

    mongo_client.admin.command("ping")
    logging.info("✅ MongoDB connected successfully!")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    raise

# -----------------------------
# Load ML Model and Features
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    logging.info("✅ ML model and feature columns loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load ML model or feature columns: {e}")
    raise

# -----------------------------
# Serialization Helpers
# -----------------------------
def json_safe_value(value):
    """Convert Mongo/Pandas/Numpy values into JSON-safe Python values."""
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value) if hasattr(pd, "isna") else False:
        return None

    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe_value(v) for v in value]

    return value


def serialize_document(doc):
    """Convert MongoDB document into JSON-safe dict."""
    if not doc:
        return doc

    serialized = {}
    for key, value in doc.items():
        if key == "_id":
            serialized["_id"] = str(value)
        else:
            serialized[key] = json_safe_value(value)
    return serialized


def serialize_documents(docs):
    return [serialize_document(doc) for doc in docs]


# -----------------------------
# Generic Helpers
# -----------------------------
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def normalize_string(value):
    if value is None:
        return None
    return str(value).strip()


def log_admin_action(action, details=None):
    try:
        admin_col.insert_one({
            "action": action,
            "details": details or {},
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        logging.warning(f"Failed to log admin action: {e}")


def generate_otp(length=6):
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def send_email_otp(to_email, otp_code):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logging.error("SMTP credentials not set in environment variables.")
        return False

    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp_code}. Expires in {OTP_EXPIRY_MINUTES} minutes."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        logging.info(f"OTP sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send OTP: {e}")
        return False


# -----------------------------
# ML Prediction Helpers
# -----------------------------
def predict_internal(new_data: pd.DataFrame):
    """Predict fraud probability and risk level."""
    processed_data = pd.get_dummies(new_data, drop_first=True)

    for col in feature_cols:
        if col not in processed_data.columns:
            processed_data[col] = 0

    new_data_aligned = processed_data[feature_cols].astype(float)

    prob = model.predict_proba(new_data_aligned)[:, 1]
    pred = (prob >= 0.5).astype(int)

    risk_level = pd.Series(
        np.where(prob < 0.2, "LOW", np.where(prob < 0.8, "MEDIUM", "HIGH")),
        index=new_data_aligned.index
    )

    results_df = pd.DataFrame({
        "prediction": pred,
        "fraud_score": prob,
        "risk_level": risk_level
    }, index=new_data_aligned.index)

    return results_df


# -----------------------------
# AI Assistant Helpers
# -----------------------------
def ensure_openai_configured():
    if not openai_client:
        raise ValueError("OPENAI_API_KEY is not configured on the backend.")


def build_transaction_context(txn):
    """
    Build a grounded AI context from a transaction.
    Supports known fraud fields and preserves all extra fields.
    """
    txn_safe = serialize_document(txn)

    known_fields = {
        "transaction_id": txn_safe.get("transaction_id"),
        "step": txn_safe.get("step"),
        "type": txn_safe.get("type"),
        "amount": safe_float(txn_safe.get("amount")),
        "nameOrig": txn_safe.get("nameOrig"),
        "recipient_name": txn_safe.get("recipient_name"),
        "nameDest": txn_safe.get("nameDest"),
        "oldbalanceOrg": safe_float(txn_safe.get("oldbalanceOrg")),
        "newbalanceOrig": safe_float(txn_safe.get("newbalanceOrig")),
        "oldbalanceDest": safe_float(txn_safe.get("oldbalanceDest")),
        "newbalanceDest": safe_float(txn_safe.get("newbalanceDest")),
        "timestamp": txn_safe.get("timestamp"),
        "channel": txn_safe.get("channel"),
        "region": txn_safe.get("region"),
        "device_id": txn_safe.get("device_id"),
        "prediction": safe_int(txn_safe.get("prediction", 0)),
        "fraud_score": round(safe_float(txn_safe.get("fraud_score", 0.0)), 6),
        "risk_level": txn_safe.get("risk_level", "UNKNOWN"),
        "created_at": txn_safe.get("created_at")
    }

    excluded = set(known_fields.keys()) | {"_id"}
    extra_fields = {
        key: value
        for key, value in txn_safe.items()
        if key not in excluded
    }

    sender_name = known_fields.get("nameOrig")
    receiver_name = known_fields.get("recipient_name") or known_fields.get("nameDest")

    return {
        **known_fields,
        "sender_name": sender_name,
        "receiver_name": receiver_name,
        "extra_fields": extra_fields
    }


def derive_rule_based_evidence(txn_context):
    """
    Derive grounded evidence from available transaction data.
    This does NOT replace your ML model. It supports explanation and reporting.
    """
    evidence = []
    risk_drivers = []
    recommended_authorities = []
    recommended_actions = []
    case_type = "internal_review"

    amount = safe_float(txn_context.get("amount"))
    fraud_score = safe_float(txn_context.get("fraud_score"))
    tx_type = normalize_string(txn_context.get("type") or "") or ""
    tx_type_upper = tx_type.upper()

    old_org = safe_float(txn_context.get("oldbalanceOrg"))
    new_org = safe_float(txn_context.get("newbalanceOrig"))
    old_dest = safe_float(txn_context.get("oldbalanceDest"))
    new_dest = safe_float(txn_context.get("newbalanceDest"))

    channel = normalize_string(txn_context.get("channel"))
    region = normalize_string(txn_context.get("region"))
    device_id = normalize_string(txn_context.get("device_id"))
    sender_name = normalize_string(txn_context.get("sender_name"))
    receiver_name = normalize_string(txn_context.get("receiver_name"))

    if fraud_score >= 0.8:
        evidence.append("The model assigned a high fraud score to this transaction.")
        risk_drivers.append("High model fraud score")
    elif fraud_score >= 0.5:
        evidence.append("The model assigned a medium-to-high fraud score to this transaction.")
        risk_drivers.append("Elevated model fraud score")

    if amount >= 1000000:
        evidence.append("The transaction amount is extremely high.")
        risk_drivers.append("Very high transaction amount")
    elif amount >= 100000:
        evidence.append("The transaction amount is unusually high.")
        risk_drivers.append("High transaction amount")

    if old_org > 0 and new_org == 0:
        evidence.append("The source account balance was depleted to zero after the transaction.")
        risk_drivers.append("Full source balance depletion")

    if old_dest == 0 and new_dest > 0:
        evidence.append("The destination account had zero previous balance before receiving funds.")
        risk_drivers.append("Destination account had zero prior balance")

    if tx_type_upper in {"TRANSFER", "CASH_OUT"}:
        evidence.append(f"The transaction type '{tx_type_upper}' is commonly treated as higher-risk for fraud monitoring.")
        risk_drivers.append(f"High-risk transaction type: {tx_type_upper}")

    if channel:
        evidence.append(f"Transaction channel recorded: {channel}.")
    if region:
        evidence.append(f"Transaction region recorded: {region}.")
    if device_id:
        evidence.append("A device identifier is available for investigation support.")
    if sender_name:
        evidence.append(f"Sender information is available: {sender_name}.")
    if receiver_name:
        evidence.append(f"Receiver information is available: {receiver_name}.")

    # Recommended actions
    recommended_actions.append("Escalate for analyst review before any external submission.")
    recommended_actions.append("Preserve the transaction record and supporting logs.")
    if device_id:
        recommended_actions.append("Retain device-related metadata for investigation.")
    if fraud_score >= 0.8:
        recommended_actions.append("Prioritize this case for urgent manual review.")
    if tx_type_upper in {"TRANSFER", "CASH_OUT"} and fraud_score >= 0.7:
        recommended_actions.append("Review linked transfer activity around this transaction.")

    # Authority routing suggestion
    if fraud_score >= 0.7 or tx_type_upper in {"TRANSFER", "CASH_OUT"}:
        recommended_authorities.append("DCI")
        case_type = "possible_cyber_or_financial_fraud"

    if amount >= 100000 or fraud_score >= 0.85:
        recommended_authorities.append("FRC")
        case_type = "suspicious_transaction_review"

    if not recommended_authorities:
        recommended_authorities.append("Internal Review Only")

    # Remove duplicates, preserve order
    recommended_authorities = list(dict.fromkeys(recommended_authorities))
    risk_drivers = list(dict.fromkeys(risk_drivers))
    recommended_actions = list(dict.fromkeys(recommended_actions))
    evidence = list(dict.fromkeys(evidence))

    return {
        "case_type": case_type,
        "evidence": evidence,
        "risk_drivers": risk_drivers,
        "recommended_authorities": recommended_authorities,
        "recommended_actions": recommended_actions
    }


def call_openai_json(prompt, fallback_payload):
    """
    Call OpenAI and attempt to parse JSON.
    Falls back safely if the response is not valid JSON.
    """
    ensure_openai_configured()

    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt
        )
        raw_text = (response.output_text or "").strip()

        if not raw_text:
            return fallback_payload

        try:
            return json.loads(raw_text)
        except Exception:
            logging.warning("OpenAI returned non-JSON output; using fallback wrapper.")
            fallback = fallback_payload.copy()
            if "summary" in fallback and not fallback.get("summary"):
                fallback["summary"] = raw_text
            elif "incident_summary" in fallback and not fallback.get("incident_summary"):
                fallback["incident_summary"] = raw_text
            elif "response" in fallback and not fallback.get("response"):
                fallback["response"] = raw_text
            else:
                fallback["raw_text"] = raw_text
            return fallback

    except Exception as e:
        logging.exception(f"OpenAI call failed: {e}")
        fallback = fallback_payload.copy()
        fallback["error"] = "OpenAI request failed; fallback response returned."
        return fallback


def get_cached_ai_result(transaction_id, task_type):
    doc = ai_cache_col.find_one({
        "transaction_id": transaction_id,
        "task_type": task_type
    })
    return serialize_document(doc) if doc else None


def save_cached_ai_result(transaction_id, task_type, payload):
    ai_cache_col.update_one(
        {"transaction_id": transaction_id, "task_type": task_type},
        {
            "$set": {
                "transaction_id": transaction_id,
                "task_type": task_type,
                "payload": payload,
                "updated_at": datetime.utcnow()
            }
        },
        upsert=True
    )


def generate_ai_transaction_explanation(txn):
    txn_context = build_transaction_context(txn)
    derived = derive_rule_based_evidence(txn_context)

    fallback = {
        "summary": "This transaction was flagged as suspicious and should be reviewed by an analyst.",
        "risk_drivers": derived["risk_drivers"],
        "recommendation": "Review the transaction manually and verify the supporting evidence before taking action.",
        "confidence_note": "Fallback response used."
    }

    prompt = f"""
You are FraudGuard AI, a financial fraud analysis assistant.

Your job is to explain why a transaction received its fraud score.
Use ONLY the transaction data and evidence provided below.
Do not invent facts.
Do not accuse any person of committing fraud.
Use careful wording such as:
- "flagged as suspicious"
- "possible fraud indicators"
- "requires analyst review"

Return ONLY valid JSON in this exact format:
{{
  "summary": "short paragraph",
  "risk_drivers": ["item1", "item2", "item3"],
  "recommendation": "short recommendation",
  "confidence_note": "short note"
}}

Transaction context:
{json.dumps(txn_context, indent=2)}

Derived grounded evidence:
{json.dumps(derived, indent=2)}
"""

    ai_json = call_openai_json(prompt, fallback)

    return {
        "transaction_id": txn_context.get("transaction_id"),
        "fraud_score": txn_context.get("fraud_score"),
        "risk_level": txn_context.get("risk_level"),
        "prediction": txn_context.get("prediction"),
        "sender_name": txn_context.get("sender_name"),
        "receiver_name": txn_context.get("receiver_name"),
        "summary": ai_json.get("summary"),
        "risk_drivers": ai_json.get("risk_drivers", derived["risk_drivers"]),
        "recommendation": ai_json.get("recommendation"),
        "confidence_note": ai_json.get("confidence_note"),
        "evidence": derived["evidence"],
        "recommended_authorities": derived["recommended_authorities"],
        "extra_fields": txn_context.get("extra_fields", {})
    }


def generate_ai_report(txn):
    txn_context = build_transaction_context(txn)
    derived = derive_rule_based_evidence(txn_context)

    fallback = {
        "case_type": derived["case_type"],
        "recommended_authority": derived["recommended_authorities"],
        "incident_summary": "This transaction was flagged as suspicious and is recommended for internal analyst review before any external reporting decision.",
        "reason_for_suspicion": derived["risk_drivers"],
        "evidence": derived["evidence"],
        "recommended_actions": derived["recommended_actions"],
        "human_review_required": True
    }

    prompt = f"""
You are FraudGuard AI, a cautious compliance and fraud reporting assistant.

Your task is to prepare a professional incident report draft.
Use ONLY the data provided.
Do not make legal conclusions.
Do not say the customer committed fraud.
State that the transaction was flagged as suspicious and requires human review.

Authority options:
- DCI
- FRC
- Internal Review Only

Return ONLY valid JSON in this exact format:
{{
  "case_type": "string",
  "recommended_authority": ["DCI"],
  "incident_summary": "string",
  "reason_for_suspicion": ["item1", "item2"],
  "evidence": ["item1", "item2"],
  "recommended_actions": ["item1", "item2"],
  "human_review_required": true
}}

Transaction context:
{json.dumps(txn_context, indent=2)}

Derived grounded evidence:
{json.dumps(derived, indent=2)}
"""

    ai_json = call_openai_json(prompt, fallback)

    return {
        "transaction_id": txn_context.get("transaction_id"),
        "fraud_score": txn_context.get("fraud_score"),
        "risk_level": txn_context.get("risk_level"),
        "sender_name": txn_context.get("sender_name"),
        "receiver_name": txn_context.get("receiver_name"),
        "report": {
            "case_type": ai_json.get("case_type", derived["case_type"]),
            "recommended_authority": ai_json.get("recommended_authority", derived["recommended_authorities"]),
            "incident_summary": ai_json.get("incident_summary"),
            "reason_for_suspicion": ai_json.get("reason_for_suspicion", derived["risk_drivers"]),
            "evidence": ai_json.get("evidence", derived["evidence"]),
            "recommended_actions": ai_json.get("recommended_actions", derived["recommended_actions"]),
            "human_review_required": ai_json.get("human_review_required", True)
        }
    }


def generate_ai_case_bundle(txn):
    """
    Combined output:
    - explanation
    - report
    Useful for one frontend call.
    """
    explanation = generate_ai_transaction_explanation(txn)
    report = generate_ai_report(txn)

    return {
        "transaction_id": explanation.get("transaction_id"),
        "fraud_score": explanation.get("fraud_score"),
        "risk_level": explanation.get("risk_level"),
        "prediction": explanation.get("prediction"),
        "sender_name": explanation.get("sender_name"),
        "receiver_name": explanation.get("receiver_name"),
        "explanation": explanation,
        "report": report.get("report")
    }


# -----------------------------
# Flask Base Endpoints
# -----------------------------
@app.route("/", methods=["GET", "OPTIONS"])
def root():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    return jsonify({"status": "Model API is running!"})


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    return jsonify({
        "status": "ok",
        "service": "ml-service",
        "openai_configured": bool(OPENAI_API_KEY)
    })


# -----------------------------
# User Registration
# -----------------------------
@app.route("/register", methods=["POST", "OPTIONS"])
def register_user():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    password = data.get("password")
    role = data.get("role", "user")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if users_col.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed = generate_password_hash(password)
    users_col.insert_one({
        "email": email,
        "password": hashed,
        "role": role,
        "login_attempts": [],
        "created_at": datetime.utcnow()
    })

    log_admin_action("register_user", {"email": email, "role": role})
    return jsonify({"message": "User registered successfully"}), 201


# -----------------------------
# Login with Password
# -----------------------------
@app.route("/login", methods=["POST", "OPTIONS"])
def login_user():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if check_password_hash(user["password"], password):
        users_col.update_one(
            {"email": email},
            {"$push": {"login_attempts": {"status": "success", "timestamp": datetime.utcnow()}}}
        )
        return jsonify({"message": "Login successful"}), 200

    users_col.update_one(
        {"email": email},
        {"$push": {"login_attempts": {"status": "failed", "timestamp": datetime.utcnow()}}}
    )
    return jsonify({"error": "Invalid credentials"}), 401


# -----------------------------
# OTP Login
# -----------------------------
@app.route("/request-otp", methods=["POST", "OPTIONS"])
def request_otp():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))

    if not email:
        return jsonify({"error": "Email required"}), 400

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    otp_code = generate_otp()
    expiry_time = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)

    users_col.update_one(
        {"email": email},
        {"$set": {"otp_code": otp_code, "otp_expiry": expiry_time}}
    )

    if send_email_otp(email, otp_code):
        return jsonify({"message": f"OTP sent to {email}"}), 200

    return jsonify({"error": "Failed to send OTP"}), 500


@app.route("/verify-otp", methods=["POST", "OPTIONS"])
def verify_otp():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = normalize_string(data.get("email"))
    otp = normalize_string(data.get("otp"))

    if not email or not otp:
        return jsonify({"error": "Email and OTP required"}), 400

    user = users_col.find_one({"email": email})
    if not user or "otp_code" not in user or "otp_expiry" not in user:
        return jsonify({"error": "No OTP found. Request a new one."}), 400

    if datetime.utcnow() > user["otp_expiry"]:
        return jsonify({"error": "OTP expired. Request a new one."}), 400

    if otp != user["otp_code"]:
        return jsonify({"error": "Invalid OTP"}), 400

    users_col.update_one(
        {"email": email},
        {"$unset": {"otp_code": "", "otp_expiry": ""}}
    )

    return jsonify({"message": "Login successful via OTP!"}), 200


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    transactions_list = data.get("transactions")

    if transactions_list is None:
        return jsonify({"error": "Missing 'transactions' key"}), 400

    if not isinstance(transactions_list, list) or len(transactions_list) == 0:
        return jsonify({"error": "'transactions' must be a non-empty list"}), 400

    try:
        temp_id_col = "__temp_transaction_id__"

        for idx, txn in enumerate(transactions_list):
            if not isinstance(txn, dict):
                return jsonify({"error": "Each transaction must be an object"}), 400
            txn[temp_id_col] = txn.get("transaction_id") or txn.get("id") or f"TXN_{idx + 1}"

        new_transactions_df = pd.DataFrame(transactions_list).set_index(temp_id_col)
        results_df = predict_internal(new_transactions_df)

        response_data = []

        for i in range(len(results_df)):
            txn_id = str(new_transactions_df.index[i])

            instance_data = {
                "transaction_id": txn_id,
                "prediction": int(results_df.iloc[i]["prediction"]),
                "fraud_score": float(results_df.iloc[i]["fraud_score"]),
                "risk_level": str(results_df.iloc[i]["risk_level"])
            }
            response_data.append(instance_data)

            record = transactions_list[i].copy()
            record.pop(temp_id_col, None)
            record.update(instance_data)
            record["created_at"] = datetime.utcnow()

            transactions_col.insert_one(record)

        return jsonify({"predictions": response_data}), 200

    except Exception as e:
        logging.exception(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed. Check input data."}), 500


# -----------------------------
# Process Dataset Endpoint
# -----------------------------
@app.route("/process-dataset", methods=["POST", "OPTIONS"])
def process_dataset():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        data = request.get_json(silent=True) or {}
        csv_content = data.get("csv_content")
        file_name = data.get("file_name", "dataset.csv")

        if not csv_content:
            return jsonify({"error": "Missing 'csv_content'"}), 400

        df = pd.read_csv(StringIO(csv_content))

        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        results_df = predict_internal(df)

        df["prediction"] = results_df["prediction"].values
        df["fraud_score"] = results_df["fraud_score"].values
        df["risk_level"] = results_df["risk_level"].values

        if "transaction_id" not in df.columns:
            df["transaction_id"] = [f"TXN_{i + 1}" for i in range(len(df))]

        inserted_count = 0
        for _, row in df.iterrows():
            record = {k: json_safe_value(v) for k, v in row.to_dict().items()}
            record["created_at"] = datetime.utcnow()
            transactions_col.insert_one(record)
            inserted_count += 1

        log_admin_action("process_dataset", {
            "file_name": file_name,
            "rows_processed": inserted_count
        })

        return jsonify({
            "message": f"{inserted_count} transactions processed successfully",
            "file_name": file_name,
            "predictions": serialize_documents(df.to_dict(orient="records"))
        }), 200

    except Exception as e:
        logging.exception(f"Dataset processing failed: {e}")
        return jsonify({"error": "Failed to process dataset"}), 500


# -----------------------------
# AI Explanation Endpoint
# -----------------------------
@app.route("/explain/<transaction_id>", methods=["GET", "OPTIONS"])
def explain_transaction(transaction_id):
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"error": "Transaction not found"}), 404

        cached = get_cached_ai_result(transaction_id, "explanation")
        if cached:
            return jsonify({
                "success": True,
                "cached": True,
                **cached["payload"]
            }), 200

        result = generate_ai_transaction_explanation(txn)
        save_cached_ai_result(transaction_id, "explanation", result)

        return jsonify({
            "success": True,
            "cached": False,
            **result
        }), 200

    except ValueError as e:
        logging.error(f"Explain transaction config error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.exception(f"Explain transaction failed: {e}")
        return jsonify({"error": "Failed to explain transaction"}), 500


# -----------------------------
# AI Report Endpoint
# -----------------------------
@app.route("/report/<transaction_id>", methods=["GET", "OPTIONS"])
def report_transaction(transaction_id):
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"error": "Transaction not found"}), 404

        cached = get_cached_ai_result(transaction_id, "report")
        if cached:
            return jsonify({
                "success": True,
                "cached": True,
                **cached["payload"]
            }), 200

        result = generate_ai_report(txn)
        save_cached_ai_result(transaction_id, "report", result)

        return jsonify({
            "success": True,
            "cached": False,
            **result
        }), 200

    except ValueError as e:
        logging.error(f"Report transaction config error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.exception(f"Report transaction failed: {e}")
        return jsonify({"error": "Failed to generate report"}), 500


# -----------------------------
# Combined AI Case Endpoint
# -----------------------------
@app.route("/ai-case/<transaction_id>", methods=["GET", "OPTIONS"])
def ai_case_bundle(transaction_id):
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"error": "Transaction not found"}), 404

        cached = get_cached_ai_result(transaction_id, "case_bundle")
        if cached:
            return jsonify({
                "success": True,
                "cached": True,
                **cached["payload"]
            }), 200

        result = generate_ai_case_bundle(txn)
        save_cached_ai_result(transaction_id, "case_bundle", result)

        return jsonify({
            "success": True,
            "cached": False,
            **result
        }), 200

    except ValueError as e:
        logging.error(f"AI case bundle config error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.exception(f"AI case bundle failed: {e}")
        return jsonify({"error": "Failed to generate AI case bundle"}), 500


# -----------------------------
# Admin Endpoints
# -----------------------------
@app.route("/admin/users", methods=["GET", "POST", "OPTIONS"])
def admin_users():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        if request.method == "GET":
            users = list(users_col.find().sort("created_at", -1))
            return jsonify({"users": serialize_documents(users)}), 200

        data = request.get_json(silent=True) or {}
        email = normalize_string(data.get("email"))
        password = data.get("password")
        role = data.get("role", "user")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        if users_col.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        hashed = generate_password_hash(password)
        users_col.insert_one({
            "email": email,
            "password": hashed,
            "role": role,
            "login_attempts": [],
            "created_at": datetime.utcnow()
        })

        log_admin_action("add_user", {"email": email, "role": role})
        return jsonify({"message": "User added successfully"}), 201

    except Exception as e:
        logging.exception(f"Admin users endpoint failed: {e}")
        return jsonify({"error": "Failed to process admin users request"}), 500


@app.route("/admin/transactions", methods=["GET", "OPTIONS"])
def admin_transactions():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        limit = safe_int(request.args.get("limit", 100), 100)
        transactions = list(transactions_col.find().sort("created_at", -1).limit(limit))
        return jsonify({"transactions": serialize_documents(transactions)}), 200
    except Exception as e:
        logging.exception(f"Admin transactions endpoint failed: {e}")
        return jsonify({"error": "Failed to load transactions"}), 500


@app.route("/admin/logs", methods=["GET", "OPTIONS"])
def admin_logs():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        limit = safe_int(request.args.get("limit", 100), 100)
        logs = list(admin_col.find().sort("timestamp", -1).limit(limit))
        return jsonify({"logs": serialize_documents(logs)}), 200
    except Exception as e:
        logging.exception(f"Admin logs endpoint failed: {e}")
        return jsonify({"error": "Failed to load admin logs"}), 500


@app.route("/admin/stats", methods=["GET", "OPTIONS"])
def admin_stats():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        total_users = users_col.count_documents({})
        total_transactions = transactions_col.count_documents({})
        total_logs = admin_col.count_documents({})
        flagged_transactions = transactions_col.count_documents({"prediction": 1})
        ai_cached_items = ai_cache_col.count_documents({})

        stats = {
            "total_users": total_users,
            "total_transactions": total_transactions,
            "total_logs": total_logs,
            "flagged_transactions": flagged_transactions,
            "ai_cached_items": ai_cached_items
        }

        return jsonify({"stats": stats}), 200
    except Exception as e:
        logging.exception(f"Admin stats endpoint failed: {e}")
        return jsonify({"error": "Failed to load dashboard stats"}), 500


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)