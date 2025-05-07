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
from io import StringIO # For saving JSON to S3
import requests # For FPL API call

# --- Add project root to sys.path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    print(f"Project Root added to path: {PROJECT_ROOT}") # For local debugging

    # Import your processing functions
    # Assuming load_raw_data_via_https is in load_raw_data_scheduled.py now
    from scheduling.load_raw_data_scheduled import load_raw_data_via_https
    from data_processing.clean_and_merge_data import clean_and_merge_data
    from data_processing.feature_engineering import engineer_features
except ImportError as e:
     logging.critical(f"Error importing modules. Check paths/imports: {e}")
     logging.critical(traceback.format_exc())
     sys.exit(1)
except Exception as e:
     logging.critical(f"Unexpected error during import setup: {e}")
     logging.critical(traceback.format_exc())
     sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout for CloudWatch
)

# --- Configuration from Environment Variables (MUST be set in ECS Task Definition) ---
try:
    logging.info("Reading configuration from environment variables...")
    ARTIFACT_BUCKET = os.environ['ARTIFACT_BUCKET']
    MODEL_POINTS_KEY = os.environ['MODEL_POINTS_KEY']
    MODEL_PRICE_KEY = os.environ['MODEL_PRICE_KEY']
    PREDICTIONS_S3_KEY_LATEST = os.environ['PREDICTIONS_S3_KEY_LATEST']
    # Optional: If you also save processed data to S3, get its key
    # PROCESSED_DATA_S3_KEY = os.environ.get('PROCESSED_DATA_S3_KEY')
    CURRENT_FPL_SEASON_FOLDER = os.environ.get('CURRENT_FPL_SEASON_FOLDER', '2024-25') # Example

    # FPL API for GW detection
    FPL_BOOTSTRAP_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    REQUEST_TIMEOUT_API = 15

    # Feature columns (ensure these are exactly what your models were trained on)
    POINTS_FEATURE_COLUMNS = os.environ.get('POINTS_FEATURE_COLUMNS', 'minutes_lag_1,points_lag_1,fdr,was_home').split(',')
    PRICE_FEATURE_COLUMNS = os.environ.get('PRICE_FEATURE_COLUMNS', 'transfers_balance_lag_1,net_transfers_roll_3,selected_lag_1,points_lag_1,cost,chance_playing_prev_gw_forecast').split(',')
    IDENTIFIER_COLUMNS = os.environ.get('IDENTIFIER_COLUMNS', 'element,web_name,position,player_static_team,gameweek,season,cost').split(',')

    # Local paths within the container's temporary storage
    LOCAL_MODEL_POINTS_PATH = Path('/tmp/points_model.joblib')
    LOCAL_MODEL_PRICE_PATH = Path('/tmp/price_model.joblib')

    logging.info(f"Configuration loaded. Artifact Bucket: {ARTIFACT_BUCKET}")
except KeyError as e:
    logging.critical(f"FATAL ERROR: Missing required environment variable: {e}")
    logging.critical(traceback.format_exc())
    sys.exit(1)
except Exception as e_conf:
    logging.critical(f"FATAL ERROR during configuration: {e_conf}")
    logging.critical(traceback.format_exc())
    sys.exit(1)


# --- Helper Function: Get FPL Gameweek Info ---
def get_fpl_gameweek_info():
    logging.info("Fetching FPL gameweek information from API...")
    try:
        response = requests.get(FPL_BOOTSTRAP_API_URL, timeout=REQUEST_TIMEOUT_API)
        response.raise_for_status()
        data = response.json()
        current_gw, next_gw, latest_finished_gw = None, None, 0

        for event in data.get('events', []):
            if event.get('is_current') is True: current_gw = event.get('id')
            if event.get('is_next') is True: next_gw = event.get('id')
            if event.get('finished') is True and event.get('id', 0) > latest_finished_gw:
                latest_finished_gw = event.get('id')
        
        # Determine prediction target
        # Predict for the GW after the latest finished one for robust feature generation
        process_until_gw = latest_finished_gw if latest_finished_gw > 0 else current_gw
        predict_for_gw = (latest_finished_gw + 1) if latest_finished_gw > 0 else (current_gw if current_gw else None)

        if not process_until_gw or not predict_for_gw or predict_for_gw > 38 : # Basic sanity check
            logging.error(f"Could not reliably determine gameweeks. Current: {current_gw}, Next: {next_gw}, Finished: {latest_finished_gw}")
            return None
        
        logging.info(f"FPL API: Process data up to GW {process_until_gw}, Predict for GW {predict_for_gw}")
        return {'process_until_gw': process_until_gw, 'predict_for_gw': predict_for_gw}
    except Exception as e:
        logging.error(f"Error fetching/processing FPL bootstrap data: {e}")
        logging.error(traceback.format_exc())
        return None


# --- Helper Function: Load Model from S3 ---
def load_model_from_s3_helper(bucket, key, local_path):
    s3_client = boto3.client('s3')
    try:
        if not os.path.exists(local_path):
            logging.info(f"Downloading model from s3://{bucket}/{key} to {local_path}")
            s3_client.download_file(bucket, key, str(local_path))
            logging.info(f"Model {key} download complete.")
        else:
            logging.info(f"Model {key} already exists locally: {local_path}")
        model = joblib.load(local_path)
        logging.info(f"Model {key} loaded successfully: {type(model)}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from s3://{bucket}/{key}: {e.__class__.__name__} - {str(e)}")
        logging.error(traceback.format_exc())
        raise


# --- Main Orchestration Function ---
def main_etl_and_predict():
    logging.info("Starting ETL and Prediction Process...")
    overall_start_time = time.time()

    # 1. Determine Gameweeks
    logging.info("--- Determining Gameweeks from FPL API ---")
    fpl_gw_info = get_fpl_gameweek_info()
    if not fpl_gw_info:
        raise Exception("Failed to determine FPL gameweeks. Cannot proceed.")

    last_completed_gw = fpl_gw_info['process_until_gw']
    prediction_target_gw = fpl_gw_info['predict_for_gw'] # e.g., 36
    logging.info(f"Last completed GW for feature source: {last_completed_gw}, Prediction target GW: {prediction_target_gw}")

    seasons_to_process_for_raw_data = [CURRENT_FPL_SEASON_FOLDER]

    # 2. Load Raw Data via HTTPS
    logging.info(f"--- Loading Raw Data for {CURRENT_FPL_SEASON_FOLDER} up to GW {prediction_target_gw} ---")
    raw_data_dict = load_raw_data_via_https(seasons_to_process_for_raw_data, prediction_target_gw)
    if raw_data_dict is None:
        raise Exception("Failed to load raw data via HTTPS.")

    # <<< --- ADD DEBUG BLOCK 1 --- >>>
    logging.info("--- Verifying raw_data_dict contents ---")
    if raw_data_dict and CURRENT_FPL_SEASON_FOLDER in raw_data_dict:
        gws_df_raw = raw_data_dict[CURRENT_FPL_SEASON_FOLDER].get('gws')
        if gws_df_raw is not None and not gws_df_raw.empty:
            logging.info(f"Raw GWS for {CURRENT_FPL_SEASON_FOLDER} shape: {gws_df_raw.shape}")
            gw_target_raw = gws_df_raw[gws_df_raw['gameweek'] == prediction_target_gw]
            logging.info(f"Found {len(gw_target_raw)} rows for GW {prediction_target_gw} in RAW GWS data.")
            if not gw_target_raw.empty:
                logging.info(f"Sample of RAW GW {prediction_target_gw} data (first few rows):\n{gw_target_raw.head().to_string()}")
        else:
            logging.warning(f"No GWS data in raw_data_dict for {CURRENT_FPL_SEASON_FOLDER}")
    else:
        logging.warning("raw_data_dict or current season data missing after load_raw_data_via_https.")
    # --- END DEBUG BLOCK 1 --- >>>

    # 3. Clean and Merge Data
    logging.info("--- Cleaning and Merging Data ---")
    merged_df = clean_and_merge_data(raw_data_dict)
    if merged_df is None or merged_df.empty:
        raise Exception("Failed to clean or merge data.")

    # <<< --- ADD DEBUG BLOCK 2 --- >>>
    logging.info("--- Verifying merged_df contents ---")
    if merged_df is not None and not merged_df.empty:
        gw_target_merged = merged_df[merged_df['gameweek'] == prediction_target_gw]
        logging.info(f"Found {len(gw_target_merged)} rows for GW {prediction_target_gw} in MERGED data.")
        if not gw_target_merged.empty:
            logging.info(f"Sample of MERGED GW {prediction_target_gw} data (cols: {gw_target_merged.columns.tolist()[:10]}...):\n{gw_target_merged.head().to_string()}")
    else:
        logging.warning("merged_df is empty or None after clean_and_merge_data.")
    # --- END DEBUG BLOCK 2 --- >>>

    # 4. Engineer Features
    logging.info("--- Engineering Features ---")
    processed_df = engineer_features(merged_df) # This is the df that will be used for prediction
    if processed_df is None or processed_df.empty:
        raise Exception("Failed to engineer features.")
    logging.info(f"Feature engineering complete. Processed DF shape: {processed_df.shape}")

    # <<< --- ADD DEBUG BLOCK 3 --- >>>
    logging.info("--- Verifying processed_df contents (output of engineer_features) ---")
    if processed_df is not None and not processed_df.empty:
        gw_target_processed = processed_df[processed_df['gameweek'] == prediction_target_gw]
        logging.info(f"Found {len(gw_target_processed)} rows for GW {prediction_target_gw} in PROCESSED data.")
        if not gw_target_processed.empty:
            logging.info(f"Sample of PROCESSED GW {prediction_target_gw} data (cols: {gw_target_processed.columns.tolist()[:10]}...):\n{gw_target_processed.head().to_string()}")
            # Check for NaNs in key lagged features for these rows
            key_lagged_features = POINTS_FEATURE_COLUMNS + PRICE_FEATURE_COLUMNS # Use your actual feature lists
            actual_key_lagged = [f for f in key_lagged_features if f in gw_target_processed.columns and ('_lag_' in f or '_roll_' in f)]
            if actual_key_lagged:
                 logging.info(f"NaN check in key lagged features for GW {prediction_target_gw} in PROCESSED data:\n{gw_target_processed[actual_key_lagged].isnull().sum().to_string()}")
        else:
             logging.warning(f"Target GW {prediction_target_gw} rows ARE EMPTY in processed_df.")
    else:
        logging.warning("processed_df is empty or None after engineer_features.")
    # --- END DEBUG BLOCK 3 --- >>>

    # 5. Load Models from S3
    logging.info("--- Loading Models from S3 ---")
    points_model = load_model_from_s3_helper(ARTIFACT_BUCKET, MODEL_POINTS_KEY, LOCAL_MODEL_POINTS_PATH)
    price_model = load_model_from_s3_helper(ARTIFACT_BUCKET, MODEL_PRICE_KEY, LOCAL_MODEL_PRICE_PATH)

    # 6. Prepare Data for Prediction (for the target prediction gameweek)
    logging.info(f"--- Preparing Prediction Input for GW {prediction_target_gw} ---")
    # The processed_df should contain rows for future GWS with features ready for prediction
    prediction_input_df = processed_df[processed_df['gameweek'] == prediction_target_gw].copy()
    if prediction_input_df.empty:
         raise Exception(f"No data found in processed_df for target prediction GW {prediction_target_gw}.")

    # Verify features and handle NaNs (basic fill - match training if different)
    X_predict_points = prediction_input_df[POINTS_FEATURE_COLUMNS].copy()
    X_predict_points.fillna(0, inplace=True) # Ensure this matches training NaN strategy
    
    X_predict_price = prediction_input_df[PRICE_FEATURE_COLUMNS].copy()
    X_predict_price.fillna(0, inplace=True) # Ensure this matches training NaN strategy

    # 7. Generate Predictions
    logging.info("--- Generating Predictions ---")
    logging.info(f"DEBUG: Columns in X_predict_price JUST BEFORE PREDICTION: {X_predict_price.columns.tolist()}")
    y_pred_points = points_model.predict(X_predict_points)
    y_pred_price = price_model.predict(X_predict_price)
    logging.info("Predictions generated.")

    # 8. Format and Save Predictions to S3
    logging.info("--- Formatting and Saving Predictions to S3 ---")
    output_df = prediction_input_df[IDENTIFIER_COLUMNS].copy()
    output_df['predicted_points'] = np.round(y_pred_points, 2)
    output_df['predicted_price_change'] = np.round(y_pred_price, 2)
    output_df['prediction_timestamp_utc'] = pd.Timestamp.utcnow().isoformat()
    output_df['predicted_for_gameweek'] = prediction_target_gw # Add the target GW

    output_json_string = output_df.to_json(orient='records', indent=2)
    s3_client = boto3.client('s3')

    s3_key_gw_specific = f"predictions/gw{prediction_target_gw}/predictions.json"
    
    logging.info(f"Uploading predictions to s3://{ARTIFACT_BUCKET}/{s3_key_gw_specific}...")
    s3_client.put_object(Bucket=ARTIFACT_BUCKET, Key=s3_key_gw_specific, Body=output_json_string, ContentType='application/json')
    logging.info(f"Uploading predictions to s3://{ARTIFACT_BUCKET}/{PREDICTIONS_S3_KEY_LATEST}...")
    s3_client.put_object(Bucket=ARTIFACT_BUCKET, Key=PREDICTIONS_S3_KEY_LATEST, Body=output_json_string, ContentType='application/json')

    total_time = time.time() - overall_start_time
    logging.info(f"--- ETL and Prediction Process Completed Successfully for GW {prediction_target_gw} in {total_time:.2f}s ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("="*60)
    logging.info(" Starting Scheduled ETL & Prediction Job ".center(60, "="))
    logging.info("="*60)

    try:
        main_etl_and_predict()
        logging.info("Job finished successfully.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Job failed with unhandled exception: {e.__class__.__name__} - {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
