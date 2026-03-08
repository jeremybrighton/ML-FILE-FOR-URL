import os
import logging
import random
from datetime import datetime, timedelta
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
CORS(app)  # Enable CORS globally

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
# Helper Functions
# -----------------------------
def predict_internal(new_data: pd.DataFrame):
    """Predict fraud probability and risk level"""
    processed_data = pd.get_dummies(new_data, drop_first=True)
    for col in feature_cols:
        if col not in processed_data.columns:
            processed_data[col] = 0
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

# -----------------------------
# Flask Endpoints
# -----------------------------
@app.route("/")
def health_check():
    return jsonify({"status": "Model API is running!"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "ml-service"})

# --- User Registration ---
@app.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    if users_col.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400
    hashed = generate_password_hash(password)
    users_col.insert_one({"email": email, "password": hashed, "role": "user", "login_attempts": []})
    return jsonify({"message": "User registered successfully"}), 201

# --- Login with Password ---
@app.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()
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
@app.route("/request-otp", methods=["POST"])
def request_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400
    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404
    otp_code = generate_otp()
    expiry_time = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    users_col.update_one({"email": email}, {"$set": {"otp_code": otp_code, "otp_expiry": expiry_time}})
    if send_email_otp(email, otp_code):
        return jsonify({"message": f"OTP sent to {email}"}), 200
    else:
        return jsonify({"error": "Failed to send OTP"}), 500

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
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
    users_col.update_one({"email": email}, {"$unset": {"otp_code": "", "otp_expiry": ""}})
    return jsonify({"message": "Login successful via OTP!"}), 200

# --- Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data or "transactions" not in data:
        return jsonify({"error": "Missing 'transactions' key"}), 400
    transactions_list = data["transactions"]
    if not isinstance(transactions_list, list) or len(transactions_list) == 0:
        return jsonify({"error": "'transactions' must be a non-empty list"}), 400
    try:
        temp_id_col = '__temp_transaction_id__'
        for idx, txn in enumerate(transactions_list):
            txn[temp_id_col] = txn.get("id", idx)
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
            record.update(instance_data)
            transactions_col.insert_one(record)

        return jsonify({"predictions": response_data}), 200
    except Exception as e:
        logging.exception(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed. Check input data."}), 500

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    # Only for local testing, do not set a fixed port in production
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)