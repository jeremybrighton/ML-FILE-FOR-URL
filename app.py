import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
import os
import logging
import os
import random
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from pymongo import MongoClient

# MongoDB URI from Render environment variable
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://FraudAdmin:Jeremy5195@detect.vn1wne0.mongodb.net/?appName=detect")
client = MongoClient(MONGO_URI)
db = client["fraud_detection"]  # Database name

# Collections
users_col = db["users"]  # Stores users and login attempts
transactions_col = db["transactions"]  # Stores predictions
admin_col = db["admin_actions"]  # Optional for admin logs

# Test connection
try:
    client.admin.command('ping')
    logging.info("✅ MongoDB connected successfully!")
except Exception as e:
    logging.error("❌ MongoDB connection failed:", e)
    exit(1)

    from werkzeug.security import generate_password_hash, check_password_hash

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
    users_col.insert_one({"email": email, "password": hashed, "role": "user", "login_attempts":[]})
    return jsonify({"message": "User registered successfully"}), 201

@app.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if check_password_hash(user["password"], password):
        users_col.update_one({"email": email}, {"$push": {"login_attempts": {"status":"success"}}})
        return jsonify({"message": "Login successful"}), 200
    else:
        users_col.update_one({"email": email}, {"$push": {"login_attempts": {"status":"failed"}}})
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Save predictions to MongoDB
for i, txn in enumerate(transactions_list):
    record = txn.copy()
    record.update({
        "prediction": int(results_df.iloc[i]["prediction"]),
        "fraud_score": float(results_df.iloc[i]["fraud_score"]),
        "risk_level": str(results_df.iloc[i]["risk_level"])
    })
    transactions_col.insert_one(record)

    import random
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# --- OTP Configuration ---
OTP_EXPIRY_MINUTES = 5  # OTP valid for 5 minutes

# --- Helper Functions ---
def generate_otp(length=6):
    """Generate a numeric OTP of given length."""
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])

def send_email_otp(to_email, otp_code):
    """
    Send OTP to user's email.
    Uses SMTP (example: Gmail). Replace credentials with environment variables in production!
    """
    sender_email = "your_email@gmail.com"
    sender_password = "your_email_password"  # use app password if using Gmail
    subject = "Your OTP Code"
    body = f"Your one-time login code is: {otp_code}. It will expire in {OTP_EXPIRY_MINUTES} minutes."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        logging.info(f"OTP sent successfully to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send OTP email to {to_email}: {e}")
        return False

# --- OTP Login Endpoints ---
@app.route("/request-otp", methods=["POST"])
def request_otp():
    """
    Step 1: User submits email. Backend generates OTP and sends via email.
    """
    data = request.get_json()
    if not data or "email" not in data:
        return jsonify({"error": "Email is required"}), 400

    user_email = data["email"]
    user_record = users_collection.find_one({"email": user_email})
    if not user_record:
        return jsonify({"error": "User not found"}), 404

    otp_code = generate_otp()
    expiry_time = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)

    # Save OTP to DB
    users_collection.update_one(
        {"email": user_email},
        {"$set": {"otp_code": otp_code, "otp_expiry": expiry_time}}
    )

    # Send OTP email
    if send_email_otp(user_email, otp_code):
        return jsonify({"message": f"OTP sent to {user_email}"}), 200
    else:
        return jsonify({"error": "Failed to send OTP"}), 500

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    """
    Step 2: User submits email + OTP. Backend verifies OTP and logs in user.
    """
    data = request.get_json()
    if not data or "email" not in data or "otp" not in data:
        return jsonify({"error": "Email and OTP are required"}), 400

    user_email = data["email"]
    submitted_otp = data["otp"]

    user_record = users_collection.find_one({"email": user_email})
    if not user_record or "otp_code" not in user_record or "otp_expiry" not in user_record:
        return jsonify({"error": "No OTP found for this user. Request a new one."}), 400

    otp_expiry = user_record["otp_expiry"]
    if datetime.utcnow() > otp_expiry:
        return jsonify({"error": "OTP expired. Please request a new one."}), 400

    if submitted_otp != user_record["otp_code"]:
        return jsonify({"error": "Invalid OTP"}), 400

    # OTP is valid; remove it from DB and allow login
    users_collection.update_one(
        {"email": user_email},
        {"$unset": {"otp_code": "", "otp_expiry": ""}}
    )

    return jsonify({"message": "Login successful via OTP!"}), 200

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_PATH = '.'

MODEL_PATH = os.path.join(PROJECT_PATH, "rf_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PROJECT_PATH, "feature_columns.json")

# --- Load Model and Features ---
model = None
feature_cols = []
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)
    logging.info("✅ Model and feature columns loaded successfully.")
except FileNotFoundError:
    logging.error(f"❌ Error: Model or feature columns file not found. "
                  f"Ensure '{MODEL_PATH}' and '{FEATURE_COLUMNS_PATH}' exist.")
    exit(1) # Exit if essential files are missing
except Exception as e:
    logging.error(f"❌ An unexpected error occurred during model loading: {e}")
    exit(1) # Exit if essential files are missing

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Prediction Function ---
def predict_internal(new_data: pd.DataFrame):
    """
    Internal prediction function for Random Forest,
    using pre-loaded model and feature_cols.
    Handles categorical variables in `new_data` and aligns features
    with the model's expected input. It returns prediction, probability, and risk_level.
    """
    if model is None or not feature_cols:
        raise RuntimeError("Model or feature columns not loaded. Cannot perform prediction.")

    processed_data = new_data.copy()

    # 1️⃣ Encode categorical variables (e.g., 'type' column if present)
    processed_data = pd.get_dummies(processed_data, drop_first=True)

    # 2️⃣ Add missing columns that were in training but not in new data
    for col in feature_cols:
        if col not in processed_data.columns:
            processed_data[col] = 0

    # 3️⃣ Keep only columns used in training, in the same order
    # Convert to float to match model's expected dtype.
    new_data_aligned = processed_data[feature_cols].astype(float)

    # 4️⃣ Predict probability and binary label
    prob = model.predict_proba(new_data_aligned)[:, 1]
    pred = (prob >= 0.5).astype(int) # Using 0.5 as default threshold

    # 5️⃣ Calculate fraud_score (same as probability) and risk_level
    fraud_score = prob
    risk_level = pd.Series(np.where(fraud_score < 0.2, 'LOW', np.where(fraud_score < 0.8, 'MEDIUM', 'HIGH')), index=new_data_aligned.index)

    # 6️⃣ Prepare results DataFrame
    results_df = pd.DataFrame({
        'prediction': pred,
        'probability': prob,
        'fraud_score': fraud_score,
        'risk_level': risk_level
    }, index=new_data_aligned.index)

    return results_df


# --- Flask Endpoints ---

@app.route("/")
def health_check():
    """Health check endpoint to confirm the API is running."""
    logging.info("Health check endpoint accessed.")
    return jsonify({"status": "Model API is running!"})

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Prediction endpoint that accepts transaction data and returns
    predictions, probabilities, fraud_score, and risk_level.
    """
    if not request.is_json:
        logging.warning("Received non-JSON request to /predict.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    logging.info(f"Received prediction request with data: {data}")

    if not data or "transactions" not in data:
        logging.warning("Missing 'transactions' key in JSON payload.")
        return jsonify({"error": "Missing 'transactions' key in JSON payload"}), 400

    transactions_list = data["transactions"]

    if not isinstance(transactions_list, list):
        logging.warning("'transactions' in payload is not a list.")
        return jsonify({"error": "'transactions' must be a list of dictionaries"}), 400

    if not transactions_list:
        logging.info("No transactions provided for prediction.")
        return jsonify({"message": "No transactions provided for prediction"}), 200

    try:
        # Convert list of dicts to DataFrame
        temp_id_col = '__temp_transaction_id__'
        for idx, txn in enumerate(transactions_list):
            if 'id' not in txn: # Add a unique ID if not present, otherwise use existing
                txn[temp_id_col] = idx
            else:
                txn[temp_id_col] = txn['id'] # Use existing 'id' for the temporary column

        new_transactions_df = pd.DataFrame(transactions_list).set_index(temp_id_col)
        new_transactions_df.index.name = None # Clear index name

        # Perform prediction
        results_df = predict_internal(new_transactions_df)

        # Combine results into a list of dictionaries for JSON response
        response_data = []
        for i in range(len(results_df)):
            instance_data = {
                "transaction_id": str(new_transactions_df.index[i]),
                "prediction": int(results_df.iloc[i]["prediction"]),
                "fraud_score": float(results_df.iloc[i]["fraud_score"]),
                "risk_level": str(results_df.iloc[i]["risk_level"])
            }
            response_data.append(instance_data)

        logging.info(f"Successfully processed {len(transactions_list)} transactions.")
        return jsonify({"predictions": response_data}), 200

    except KeyError as e:
        error_msg = (f"Missing expected feature in input data or malformed input: {e}. "
                     f"Please ensure all required features are present and correctly named. "
                     f"Expected features (after one-hot encoding if applicable): {', '.join(feature_cols)}. "
                     f"Received features: {list(new_transactions_df.columns) if 'new_transactions_df' in locals() else 'N/A'} ")
        logging.error(f"KeyError in predict_endpoint: {error_msg}")
        return jsonify({"error": error_msg}), 400
    except RuntimeError as e:
        logging.error(f"RuntimeError in predict_endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.exception(f"An unexpected error occurred in predict_endpoint: {e}") # Use exception for full traceback
        return jsonify({"error": f"An unexpected error occurred: {str(e)}. Please contact support."}), 500
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)