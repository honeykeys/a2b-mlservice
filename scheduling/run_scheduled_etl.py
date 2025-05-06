# scheduling/run_scheduled_etl.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import boto3
import traceback
import logging
import time
from pathlib import Path
from io import StringIO # Needed if saving JSON string directly to S3

# --- Add project root to sys.path to allow imports from sibling directories ---
# Assumes this script is in scheduling/ and others are in data_processing/, modeling/ etc.
# Adjust path depth if your structure is different
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    print(f"Project Root added to path: {PROJECT_ROOT}")

    # Now import your modules (adjust paths if needed)
    from scheduling.load_raw_data_scheduled import load_raw_data_via_https
    from data_processing.clean_and_merge_data import clean_and_merge_data
    from data_processing.feature_engineering import engineer_features
except ImportError as e:
     print(f"Error importing modules. Check paths and __init__.py files: {e}", file=sys.stderr)
     sys.exit(1)
except Exception as e:
     print(f"Unexpected error during import setup: {e}", file=sys.stderr)
     sys.exit(1)

# --- Configure Logging ---
# Sends logs to stdout/stderr -> CloudWatch Logs in Fargate
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration from Environment Variables ---
# These MUST be set in the ECS Task Definition
try:
    logging.info("Reading configuration from environment variables...")
    # Data Sources / Outputs
    RAW_DATA_BASE_URL = os.environ['RAW_DATA_BASE_URL']
    ARTIFACT_BUCKET = os.environ['ARTIFACT_BUCKET']
    # S3 Key for processed data (optional if kept in memory)
    # PROCESSED_DATA_S3_KEY = os.environ.get('PROCESSED_DATA_S3_KEY')
    PREDICTIONS_S3_KEY_LATEST = os.environ['PREDICTIONS_S3_KEY_LATEST']
    # S3 Keys for models
    MODEL_POINTS_KEY = os.environ['MODEL_POINTS_KEY']
    MODEL_PRICE_KEY = os.environ['MODEL_PRICE_KEY']
    # Seasons / GWs
    TARGET_SEASONS = os.environ.get('TARGET_SEASONS', '2023-24,2024-25').split(',')
    CURRENT_SEASON_MAX_GW = int(os.environ.get('CURRENT_SEASON_MAX_GW', '38')) # Default to full season if not set

    # --- CRITICAL: Define Feature lists EXACTLY as used for training ---
    POINTS_FEATURE_COLUMNS = [ # Features for points model (e.g., dt_model_v1)
        'minutes_lag_1', 'points_lag_1', 'fdr', 'was_home',
        # Add ALL others...
    ]
    PRICE_FEATURE_COLUMNS = [ # Features for price model (e.g., price_predictor_rf_v1)
        'transfers_balance_lag_1', 'net_transfers_roll_3', 'selected_lag_1',
        'points_lag_1', 'cost', 'chance_playing_prev_gw_forecast', # If available & used
        # Add ALL others...
    ]
    IDENTIFIER_COLUMNS = ['element', 'web_name', 'position', 'player_static_team', 'gameweek', 'season', 'cost']

    # Local paths within the container's temporary storage
    LOCAL_MODEL_POINTS_PATH = Path('/tmp/points_model.joblib')
    LOCAL_MODEL_PRICE_PATH = Path('/tmp/price_model.joblib')

    logging.info(f"Configuration loaded. Bucket: {ARTIFACT_BUCKET}")

except KeyError as e:
    logging.error(f"FATAL ERROR: Missing required environment variable: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"FATAL ERROR during configuration: {e}")
    sys.exit(1)


# --- Helper Function: Load Model from S3 ---
def load_model_from_s3_helper(bucket, key, local_path):
    """Downloads and loads a joblib model from S3."""
    s3_client = boto3.client('s3')
    try:
        # Skip download if already exists locally (e.g., from previous step in same run?)
        if not os.path.exists(local_path):
            logging.info(f"Downloading model from s3://{bucket}/{key} to {local_path}")
            s3_client.download_file(bucket, key, str(local_path))
            logging.info("Download complete.")
        else:
            logging.info(f"Model file already exists locally: {local_path}")

        logging.info(f"Loading model from {local_path}...")
        model = joblib.load(local_path)
        logging.info(f"Model loaded successfully: {type(model)}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from s3://{bucket}/{key}: {e.__class__.__name__} - {str(e)}")
        logging.error(traceback.format_exc())
        raise # Re-raise to be caught by main handler


# --- Main Orchestration Function ---
def run_etl_and_predict():
    logging.info("Starting ETL and Prediction Process...")
    start_time = time.time()

    # 1. Load Raw Data via HTTPS
    logging.info("--- Running Task 1: Load Raw Data ---")
    raw_data = load_raw_data_via_https(TARGET_SEASONS, CURRENT_SEASON_MAX_GW)
    if raw_data is None:
        raise Exception("Failed to load raw data.")

    # 2. Clean and Merge Data
    logging.info("--- Running Task 2: Clean and Merge Data ---")
    merged_data = clean_and_merge_data(raw_data)
    if merged_data is None or merged_data.empty:
        raise Exception("Failed to clean or merge data.")

    # 3. Engineer Features
    logging.info("--- Running Task 3: Feature Engineering ---")
    processed_df = engineer_features(merged_data) # Contains features AND targets
    if processed_df is None or processed_df.empty:
        raise Exception("Failed to engineer features.")
    logging.info(f"Feature engineering complete. Shape: {processed_df.shape}")
    # Optional: Save processed_df with features to S3 here if needed for other processes
    # e.g., processed_df.to_parquet(f's3://{ARTIFACT_BUCKET}/{PROCESSED_DATA_S3_KEY}', ...)

    # 4. Load Models from S3
    logging.info("--- Running Task 4: Load Models ---")
    points_model = load_model_from_s3_helper(ARTIFACT_BUCKET, MODEL_POINTS_KEY, LOCAL_MODEL_POINTS_PATH)
    price_model = load_model_from_s3_helper(ARTIFACT_BUCKET, MODEL_PRICE_KEY, LOCAL_MODEL_PRICE_PATH)

    # 5. Prepare Data for Prediction
    logging.info("--- Running Task 5: Prepare Prediction Input ---")
    # Assuming the processed_df contains rows ready for the *next* GW prediction
    target_gw = int(processed_df['gameweek'].min()) # Predict for earliest GW in file
    logging.info(f"Preparing features for GW {target_gw}...")
    prediction_input_df = processed_df[processed_df['gameweek'] == target_gw].copy()
    if prediction_input_df.empty:
         raise Exception(f"No data found for prediction target GW {target_gw}.")

    # Verify features and handle NaNs just before prediction
    missing_points_features = [col for col in POINTS_FEATURE_COLUMNS if col not in prediction_input_df.columns]
    missing_price_features = [col for col in PRICE_FEATURE_COLUMNS if col not in prediction_input_df.columns]
    if missing_points_features or missing_price_features:
        raise ValueError(f"Required features missing for prediction. Points: {missing_points_features}, Price: {missing_price_features}")

    X_predict_points = prediction_input_df[POINTS_FEATURE_COLUMNS].copy()
    X_predict_price = prediction_input_df[PRICE_FEATURE_COLUMNS].copy()

    # Basic NaN fill - replace with strategy matching training if needed
    X_predict_points.fillna(0, inplace=True)
    X_predict_price.fillna(0, inplace=True)

    # 6. Generate Predictions
    logging.info("--- Running Task 6: Generate Predictions ---")
    pred_start_time = time.time()
    y_pred_points = points_model.predict(X_predict_points)
    y_pred_price = price_model.predict(X_predict_price)
    logging.info(f"Predictions generated in {time.time() - pred_start_time:.2f}s.")

    # 7. Format and Save Predictions to S3
    logging.info("--- Running Task 7: Format and Save Predictions ---")
    output_df = prediction_input_df[IDENTIFIER_COLUMNS].copy()
    output_df['predicted_points'] = np.round(y_pred_points, 2)
    output_df['predicted_price_change'] = np.round(y_pred_price, 2)
    output_df['prediction_timestamp'] = pd.Timestamp.utcnow().isoformat() # Add timestamp

    # Save as JSON string
    output_json_string = output_df.to_json(orient='records', indent=2)

    # Upload to S3
    s3_client = boto3.client('s3')
    s3_key_specific = f"predictions/gw{target_gw}/predictions.json" # Example specific path
    s3_key_latest = PREDICTIONS_S3_KEY_LATEST # From env var

    logging.info(f"Uploading predictions to s3://{ARTIFACT_BUCKET}/{s3_key_specific}...")
    s3_client.put_object(Bucket=ARTIFACT_BUCKET, Key=s3_key_specific, Body=output_json_string, ContentType='application/json')
    logging.info(f"Uploading predictions to s3://{ARTIFACT_BUCKET}/{s3_key_latest}...")
    s3_client.put_object(Bucket=ARTIFACT_BUCKET, Key=s3_key_latest, Body=output_json_string, ContentType='application/json')

    total_time = time.time() - start_time
    logging.info(f"--- ETL and Prediction Process Completed Successfully in {total_time:.2f}s ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("="*60)
    logging.info(" Starting Scheduled ETL & Prediction Job ".center(60, "="))
    logging.info("="*60)

    try:
        run_etl_and_predict()
        logging.info("Job finished successfully.")
        sys.exit(0) # Explicit success exit code
    except Exception as e:
        logging.error(f"Job failed with unhandled exception: {e.__class__.__name__} - {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1) # Explicit failure exit code
