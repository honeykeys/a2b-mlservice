# prediction/generate_predictions.py

import pandas as pd
import joblib
import os
import sys
from pathlib import Path
import boto3
import numpy as np
import time # Added for load timing example
import traceback

print('Starting prediction script...')

# --- Configuration ---
try:
    # Define project root dynamically assuming this script is in prediction/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"Detected Project Root: {PROJECT_ROOT}")

    # --- Paths to Input Artifacts (MUST be correct) ---
    # Assumes ETL output this parquet file with features ready for prediction
    PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'fpl_analytics_mvp_v1.parquet'
    # Assumes model training saved this file
    MODEL_PATH = PROJECT_ROOT / 'models' / 'dt_model_v1.joblib' # <-- Point to your chosen model file

    # --- Output Configuration ---
    PREDICTIONS_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'predictions'
    # S3 Bucket where predictions will be uploaded
    S3_BUCKET_NAME = 'a2b-ml-artifacts-kn-euw2-20250504' # <-- Use YOUR bucket name

    # <<< --- CRITICAL: Update this list to match your trained model EXACTLY --- >>>
    FEATURE_COLUMNS = [
        'minutes_lag_1',
        'points_lag_1',
        'fdr',
        'was_home',
        # Add ALL other features your chosen model expects in the correct order!
    ]

    # Columns to keep in the final output file alongside predictions
    # Ensure these columns EXIST in your PROCESSED_DATA_PATH parquet file!
    IDENTIFIER_COLUMNS = ['element', 'web_name', 'position', 'player_static_team', 'gameweek', 'season']

except Exception as e:
    print(f"Error during initial configuration: {e}", file=sys.stderr)
    sys.exit(1)


def generate_predictions(processed_data_file, model_file, output_dir, s3_bucket):
    """
    Loads processed data & model, predicts points for the next available gameweek,
    saves predictions locally, and uploads to S3.
    """
    print("\n--- Starting Prediction Generation Function ---")
    start_time_total = time.time()

    # --- Load Processed Data ---
    try:
        print(f"Loading processed data from: {processed_data_file}")
        if not processed_data_file.is_file():
            raise FileNotFoundError(f"Processed data file not found: {processed_data_file}")
        df = pd.read_parquet(processed_data_file)
        print(f"Loaded processed data shape: {df.shape}")
        required_load_cols = ['gameweek'] + IDENTIFIER_COLUMNS + FEATURE_COLUMNS
        missing_load_cols = [col for col in required_load_cols if col not in df.columns]
        if missing_load_cols:
            raise KeyError(f"Required columns missing after load: {missing_load_cols}")
        # Ensure gameweek is numeric for filtering/finding max
        df['gameweek'] = pd.to_numeric(df['gameweek'], errors='coerce')
        df.dropna(subset=['gameweek'], inplace=True)
        df['gameweek'] = df['gameweek'].astype(int)

    except Exception as e:
        print(f"Error loading processed data: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Load Model ---
    try:
        print(f"Loading model from: {model_file}")
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model = joblib.load(str(model_file)) # joblib might prefer string path
        print(f"Successfully loaded model: {type(model)}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Determine Target Gameweek ---
    # Assumption: The Parquet file contains rows ready for prediction.
    # We predict for the EARLIEST gameweek present in this file.
    # (Alternatively, could find max historical GW and predict GW+1 if file structure differs)
    if df.empty:
        print("Error: Processed data frame is empty.", file=sys.stderr)
        return False
    try:
        target_gw = int(df['gameweek'].min())
        print(f"Identified target prediction Gameweek: {target_gw}")
        prediction_df = df[df['gameweek'] == target_gw].copy()
        if prediction_df.empty:
             print(f"Error: No data found for target Gameweek {target_gw} in the processed file.", file=sys.stderr)
             return False
    except Exception as e:
        print(f"Error determining target gameweek: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Prepare Features (X_predict) ---
    print(f"Preparing features for GW {target_gw}...")
    try:
        # Verify all needed feature columns are present in the filtered data
        missing_features = [col for col in FEATURE_COLUMNS if col not in prediction_df.columns]
        if missing_features:
            raise ValueError(f"Required feature columns missing from filtered data: {missing_features}")

        X_predict = prediction_df[FEATURE_COLUMNS]

        # Ensure data types match training (basic numeric check)
        for col in X_predict.columns:
            X_predict[col] = pd.to_numeric(X_predict[col], errors='coerce')

        if X_predict.isnull().sum().sum() > 0:
            print("Warning: NaNs found in features before prediction. Filling with 0 for now.", file=sys.stderr)
            print(X_predict.isnull().sum())
            # Basic Imputation - replace with more sophisticated strategy if needed
            X_predict.fillna(0, inplace=True)

    except Exception as e:
        print(f"Error preparing features: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Generate Predictions ---
    try:
        print(f"Generating predictions using {type(model).__name__}...")
        start_time_pred = time.time()
        y_pred = model.predict(X_predict)
        pred_time = time.time() - start_time_pred
        print(f"Predictions generated in {pred_time:.2f}s.")
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Format and Save Output ---
    try:
        print("Formatting and saving output...")
        output_df = prediction_df[IDENTIFIER_COLUMNS].copy()
        # Add predictions, ensuring index aligns if X_predict was filtered/modified
        output_df['predicted_points'] = y_pred
        output_df['predicted_points'] = np.round(output_df['predicted_points'], 2) # Round to 2 decimal places

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output filenames
        local_filename = output_dir / f"predictions_gw{target_gw}.json"
        s3_key_specific = f"predictions/gw{target_gw}/predictions.json"
        s3_key_latest = "predictions/latest_predictions.json"

        print(f"Saving predictions locally to: {local_filename}")
        output_df.to_json(local_filename, orient='records', indent=2)
        print("Local save successful.")

        # --- S3 Upload Logic ---
        print(f"\nAttempting to upload predictions to S3 bucket: {s3_bucket}...")
        s3_client = boto3.client('s3')

        print(f"Uploading {local_filename} to s3://{s3_bucket}/{s3_key_specific}...")
        s3_client.upload_file(str(local_filename), s3_bucket, s3_key_specific)
        print("  Upload to specific key successful.")

        print(f"Uploading {local_filename} to s3://{s3_bucket}/{s3_key_latest}...")
        s3_client.upload_file(str(local_filename), s3_bucket, s3_key_latest)
        print("  Upload to latest key successful.")
        # --- End S3 Upload Logic ---

        total_time = time.time() - start_time_total
        print(f"\nPredictions saved locally and uploaded successfully. Total time: {total_time:.2f}s.")
        return True

    except boto3.exceptions.NoCredentialsError:
         print("Error: AWS Credentials not found for S3 upload. Configure AWS CLI or environment variables.", file=sys.stderr)
         return False
    except Exception as e:
        print(f"Error formatting, saving, or uploading output: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False


# --- Main Execution ---
if __name__ == "__main__":
    print("="*50)
    print("Starting Offline Prediction Generation Job")
    print("="*50)

    # Pass the S3 bucket name to the function
    success = generate_predictions(
        processed_data_file=PROCESSED_DATA_PATH,
        model_file=MODEL_PATH,
        output_dir=PREDICTIONS_OUTPUT_DIR,
        s3_bucket=S3_BUCKET_NAME
    )

    print("="*50)
    if success:
        print("Prediction script finished successfully.")
        print(f"Output saved to {PREDICTIONS_OUTPUT_DIR} and uploaded to S3 bucket {S3_BUCKET_NAME}.")
    else:
        print("Prediction script finished with errors.")
        sys.exit(1)
    print("="*50)