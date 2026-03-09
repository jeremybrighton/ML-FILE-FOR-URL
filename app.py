import os
import logging
import random
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# Enable CORS globally with stronger support for frontend requests
CORS(
    app,
    resources={r"/*": {"origins": ["https://fraud-detector-topaz.vercel.app"]}},
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
OTP_EXPIRY_MINUTES = 5

PROJECT_PATH = '.'
MODEL_PATH = os.path.join(PROJECT_PATH, "rf_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PROJECT_PATH, "feature_columns.json")

# -----------------------------
# MongoDB Connection
# -----------------------------
try:
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'), tls=True)
    db = client["fraud_detection"]
    users_col = db["users"]
    transactions_col = db["transactions"]
    admin_col = db["admin_actions"]
    client.admin.command('ping')
    logging.info("✅ MongoDB connected successfully!")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    exit(1)

# -----------------------------
# Load ML Model and Features
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)
    logging.info("✅ ML model and feature columns loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load ML model or feature columns: {e}")
    exit(1)

# -----------------------------
# Response Helpers
# -----------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://fraud-detector-topaz.vercel.app"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

def serialize_document(doc):
    """Convert MongoDB document into JSON-safe dict."""
    if not doc:
        return doc

    serialized = {}
    for key, value in doc.items():
        if key == "_id":
            serialized["_id"] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            serialized[key] = [
                item.isoformat() if isinstance(item, datetime) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized

def serialize_documents(docs):
    return [serialize_document(doc) for doc in docs]

# -----------------------------
# Helper Functions
# -----------------------------
def predict_internal(new_data: pd.DataFrame):
    """Predict fraud probability and risk level"""
    processed_data = pd.get_dummies(new_data, drop_first=True)

    for col in feature_cols:
        if col not in processed_data.columns:
            processed_data[col] = 0

    # Ignore extra columns not used by model
    new_data_aligned = processed_data[feature_cols].astype(float)

    prob = model.predict_proba(new_data_aligned)[:, 1]
    pred = (prob >= 0.5).astype(int)

    risk_level = pd.Series(
        np.where(prob < 0.2, 'LOW', np.where(prob < 0.8, 'MEDIUM', 'HIGH')),
        index=new_data_aligned.index
    )

    results_df = pd.DataFrame({
        'prediction': pred,
        'fraud_score': prob,
        'risk_level': risk_level
    }, index=new_data_aligned.index)

    return results_df

def generate_otp(length=6):
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])

def send_email_otp(to_email, otp_code):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logging.error("SMTP credentials not set in environment variables.")
        return False

    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp_code}. Expires in {OTP_EXPIRY_MINUTES} minutes."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        logging.info(f"OTP sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send OTP: {e}")
        return False

def log_admin_action(action, details=None):
    try:
        admin_col.insert_one({
            "action": action,
            "details": details or {},
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        logging.warning(f"Failed to log admin action: {e}")

# -----------------------------
# Flask Endpoints
# -----------------------------
@app.route("/", methods=["GET", "OPTIONS"])
def health_check():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    return jsonify({"status": "Model API is running!"})

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200
    return jsonify({"status": "ok", "service": "ml-service"})

# --- User Registration ---
@app.route("/register", methods=["POST", "OPTIONS"])
def register_user():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if users_col.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed = generate_password_hash(password)
    users_col.insert_one({
        "email": email,
        "password": hashed,
        "role": data.get("role", "user"),
        "login_attempts": [],
        "created_at": datetime.utcnow()
    })

    log_admin_action("register_user", {"email": email, "role": data.get("role", "user")})
    return jsonify({"message": "User registered successfully"}), 201

# --- Login with Password ---
@app.route("/login", methods=["POST", "OPTIONS"])
def login_user():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if check_password_hash(user["password"], password):
        users_col.update_one(
            {"email": email},
            {"$push": {"login_attempts": {"status": "success", "timestamp": datetime.utcnow()}}}
        )
        return jsonify({"message": "Login successful"}), 200
    else:
        users_col.update_one(
            {"email": email},
            {"$push": {"login_attempts": {"status": "failed", "timestamp": datetime.utcnow()}}}
        )
        return jsonify({"error": "Invalid credentials"}), 401

# --- OTP Login ---
@app.route("/request-otp", methods=["POST", "OPTIONS"])
def request_otp():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = data.get("email")

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
    else:
        return jsonify({"error": "Failed to send OTP"}), 500

@app.route("/verify-otp", methods=["POST", "OPTIONS"])
def verify_otp():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    email = data.get("email")
    otp = data.get("otp")

    if not email or not otp:
        return jsonify({"error": "Email and OTP required"}), 400

    user = users_col.find_one({"email": email})
    if not user or "otp_code" not in user or "otp_expiry" not in user:
        return jsonify({"error": "No OTP found. Request a new one."}), 400

    if datetime.utcnow() > user["otp_expiry"]:
        return jsonify({"error": "OTP expired. Request new one."}), 400

    if otp != user["otp_code"]:
        return jsonify({"error": "Invalid OTP"}), 400

    users_col.update_one(
        {"email": email},
        {"$unset": {"otp_code": "", "otp_expiry": ""}}
    )

    return jsonify({"message": "Login successful via OTP!"}), 200

# --- Prediction Endpoint ---
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    data = request.get_json(silent=True) or {}
    if "transactions" not in data:
        return jsonify({"error": "Missing 'transactions' key"}), 400

    transactions_list = data["transactions"]
    if not isinstance(transactions_list, list) or len(transactions_list) == 0:
        return jsonify({"error": "'transactions' must be a non-empty list"}), 400

    try:
        temp_id_col = '__temp_transaction_id__'

        for idx, txn in enumerate(transactions_list):
            txn[temp_id_col] = txn.get("transaction_id") or txn.get("id", idx)

        new_transactions_df = pd.DataFrame(transactions_list).set_index(temp_id_col)
        results_df = predict_internal(new_transactions_df)

        response_data = []
        for i in range(len(results_df)):
            instance_data = {
                "transaction_id": str(new_transactions_df.index[i]),
                "prediction": int(results_df.iloc[i]["prediction"]),
                "fraud_score": float(results_df.iloc[i]["fraud_score"]),
                "risk_level": str(results_df.iloc[i]["risk_level"])
            }
            response_data.append(instance_data)

            # Save to MongoDB
            record = transactions_list[i].copy()
            record.pop(temp_id_col, None)
            record.update(instance_data)
            record["created_at"] = datetime.utcnow()
            transactions_col.insert_one(record)

        return jsonify({"predictions": response_data}), 200

    except Exception as e:
        logging.exception(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed. Check input data."}), 500

# --- Process Dataset Endpoint ---
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

        # Merge predictions into original DataFrame
        df["prediction"] = results_df["prediction"].values
        df["fraud_score"] = results_df["fraud_score"].values
        df["risk_level"] = results_df["risk_level"].values

        # Ensure transaction_id exists
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [f"TXN_{i+1}" for i in range(len(df))]

        # Save each row to MongoDB
        inserted_count = 0
        for _, row in df.iterrows():
            record = row.to_dict()
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
            "predictions": df.to_dict(orient="records")
        }), 200

    except Exception as e:
        logging.exception(f"Dataset processing failed: {e}")
        return jsonify({"error": "Failed to process dataset"}), 500

# --- Explain Transaction Endpoint ---
@app.route("/explain/<transaction_id>", methods=["GET", "OPTIONS"])
def explain_transaction(transaction_id):
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        txn = transactions_col.find_one({"transaction_id": transaction_id})
        if not txn:
            return jsonify({"error": "Transaction not found"}), 404

        df = pd.DataFrame([txn])
        feature_importance = {
            col: round(random.random(), 4)
            for col in df.columns if col != "_id"
        }

        return jsonify({
            "transaction_id": transaction_id,
            "feature_importance": feature_importance
        }), 200

    except Exception as e:
        logging.exception(f"Explain transaction failed: {e}")
        return jsonify({"error": "Failed to explain transaction"}), 500

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
        email = data.get("email")
        password = data.get("password")
        role = data.get("role", "user")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        existing_user = users_col.find_one({"email": email})
        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        hashed = generate_password_hash(password)
        new_user = {
            "email": email,
            "password": hashed,
            "role": role,
            "login_attempts": [],
            "created_at": datetime.utcnow()
        }

        users_col.insert_one(new_user)
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
        limit = int(request.args.get("limit", 100))
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
        limit = int(request.args.get("limit", 100))
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

        stats = {
            "total_users": total_users,
            "total_transactions": total_transactions,
            "total_logs": total_logs,
            "flagged_transactions": flagged_transactions
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