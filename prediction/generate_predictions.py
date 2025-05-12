import pandas as pd
import joblib
import os
import sys
from pathlib import Path
import boto3
import numpy as np
import time
import traceback

print('Starting prediction script (Points & Price)...')

# --- Configuration ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"Detected Project Root: {PROJECT_ROOT}")
    PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'fpl_analytics_mvp_v1.parquet'
    POINTS_MODEL_PATH = PROJECT_ROOT / 'models' / 'dt_model_v1.joblib'
    PRICE_MODEL_PATH = PROJECT_ROOT / 'models' / 'price_predictor_rf_v1.joblib'
    PREDICTIONS_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'predictions'
    S3_BUCKET_NAME = 'a2b-ml-artifacts-kn-euw2-20250504'
    POINTS_FEATURE_COLUMNS = [
        'minutes_lag_1',
        'points_lag_1',
        'fdr',
        'was_home',
    ]
    PRICE_FEATURE_COLUMNS = [
        'transfers_balance_lag_1',
        'net_transfers_roll_3',
        'selected_lag_1',
        'points_lag_1',
        'cost',
        'chance_playing_prev_gw_forecast',
    ]
    IDENTIFIER_COLUMNS = ['element', 'web_name', 'position', 'player_static_team', 'gameweek', 'season', 'cost']

except Exception as e:
    print(f"Error during initial configuration: {e}", file=sys.stderr)
    sys.exit(1)


def generate_predictions(processed_data_file,
                         points_model_path, price_model_path,
                         output_dir, s3_bucket):
    """
    Loads processed data & models, predicts points & price changes for the
    next available gameweek, saves combined predictions locally, and uploads to S3.
    """
    print("\n--- Starting Prediction Generation Function (Points & Price) ---")
    start_time_total = time.time()

    # --- Load Processed Data ---
    try:
        print(f"Loading processed data from: {processed_data_file}")
        if not processed_data_file.is_file():
            raise FileNotFoundError(f"Processed data file not found: {processed_data_file}")
        df = pd.read_parquet(processed_data_file)
        print(f"Loaded processed data shape: {df.shape}")
        all_required_cols = list(set(
            POINTS_FEATURE_COLUMNS + PRICE_FEATURE_COLUMNS + IDENTIFIER_COLUMNS + ['gameweek', 'season']
        ))
        missing_load_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_load_cols:
            raise KeyError(f"Required columns missing after load: {missing_load_cols}")

        df['gameweek'] = pd.to_numeric(df['gameweek'], errors='coerce').fillna(0).astype(int)

    except Exception as e:
        print(f"Error loading processed data: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Load Models ---
    try:
        print(f"Loading points model from: {points_model_path}")
        if not points_model_path.is_file(): raise FileNotFoundError(f"Points model not found: {points_model_path}")
        points_model = joblib.load(str(points_model_path))
        print(f"Successfully loaded points model: {type(points_model)}")

        print(f"Loading price model from: {price_model_path}")
        if not price_model_path.is_file(): raise FileNotFoundError(f"Price model not found: {price_model_path}")
        price_model = joblib.load(str(price_model_path))
        print(f"Successfully loaded price model: {type(price_model)}")
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Determine Target Gameweek ---
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
        print(f"Error determining target gameweek or filtering data: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Prepare Features & Predict ---
    try:
        print(f"Preparing features and predicting for GW {target_gw}...")
        X_predict_points = prediction_df[POINTS_FEATURE_COLUMNS].copy()
        if X_predict_points.isnull().sum().sum() > 0:
             print("Warning: NaNs found in points features. Filling with 0.", file=sys.stderr)
             X_predict_points.fillna(0, inplace=True)
        print(f"Generating points predictions ({X_predict_points.shape[0]} players)...")
        y_pred_points = points_model.predict(X_predict_points)
        X_predict_price = prediction_df[PRICE_FEATURE_COLUMNS].copy()
        if X_predict_price.isnull().sum().sum() > 0:
             print("Warning: NaNs found in price features. Filling with 0.", file=sys.stderr)
             X_predict_price.fillna(0, inplace=True)
        print(f"Generating price change predictions ({X_predict_price.shape[0]} players)...")
        y_pred_price = price_model.predict(X_predict_price)

        print("Predictions generated successfully.")

    except Exception as e:
        print(f"Error preparing features or during prediction: {e}", file=sys.stderr)
        print(traceback.format_exc())
        return False

    # --- Format and Save Output ---
    try:
        print("Formatting and saving output...")
        output_df = prediction_df[IDENTIFIER_COLUMNS].copy()
        output_df['predicted_points'] = np.round(y_pred_points, 2)
        output_df['predicted_price_change'] = np.round(y_pred_price, 2)
        output_dir.mkdir(parents=True, exist_ok=True)
        local_filename = output_dir / f"predictions_gw{target_gw}.json"
        s3_key_specific = f"predictions/gw{target_gw}/predictions.json"
        s3_key_latest = "predictions/latest_predictions.json"
        print(f"Saving combined predictions locally to: {local_filename}")
        output_df.to_json(local_filename, orient='records', indent=2)
        print("Local save successful.")
        print(f"\nAttempting to upload predictions to S3 bucket: {s3_bucket}...")
        s3_client = boto3.client('s3')
        print(f"Uploading {local_filename} to s3://{s3_bucket}/{s3_key_specific}...")
        s3_client.upload_file(str(local_filename), s3_bucket, s3_key_specific)
        print("  Upload to specific key successful.")
        print(f"Uploading {local_filename} to s3://{s3_bucket}/{s3_key_latest}...")
        s3_client.upload_file(str(local_filename), s3_bucket, s3_key_latest)
        print("  Upload to latest key successful.")
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
if __name__ == "__main__":
    print("="*50)
    print("Starting Offline Prediction Generation Job (Points & Price)")
    print("="*50)
    success = generate_predictions(
        processed_data_file=PROCESSED_DATA_PATH,
        points_model_path=POINTS_MODEL_PATH,
        price_model_path=PRICE_MODEL_PATH,
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