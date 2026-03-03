import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_PATH = "/content/drive/MyDrive/fraud_project"

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
    pass # Execution instructions are provided separately
