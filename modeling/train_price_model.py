import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import sys
import numpy as np

print("--- Starting Price Predictor Model Training Script ---")

# --- Configuration ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'fpl_analytics_mvp_v1.parquet'
    MODEL_OUTPUT_DIR = PROJECT_ROOT / 'models'
    MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / 'price_predictor_rf_v1.joblib'
    PRICE_FEATURE_COLUMNS = [
        'transfers_balance_lag_1',
        'net_transfers_roll_3',
        'selected_lag_1',
        'points_lag_1',
        'cost',
        'chance_playing_prev_gw_forecast', 
    ]
    TARGET_COLUMN_PRICE = 'price_change'
    TEST_SPLIT_WEEKS = 8

except Exception as e:
    print(f"Error during configuration: {e}", file=sys.stderr)
    sys.exit(1)

# --- Load Data ---
try:
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    if not PROCESSED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Processed data file not found: {PROCESSED_DATA_PATH}")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Data loaded shape: {df.shape}")
    required_cols_for_script = PRICE_FEATURE_COLUMNS + [TARGET_COLUMN_PRICE, 'gameweek', 'season']
    missing_cols = [col for col in required_cols_for_script if col not in df.columns]
    if missing_cols:
         raise KeyError(f"Required columns missing from Parquet file: {missing_cols}")

except Exception as e:
    print(f"Error loading data: {e}", file=sys.stderr)
    sys.exit(1)


# --- Prepare Data for Price Model ---
try:
    print("Preparing data for price model...")
    cols_to_keep = PRICE_FEATURE_COLUMNS + [TARGET_COLUMN_PRICE, 'gameweek', 'season', 'element']
    df_price = df[cols_to_keep].copy()
    initial_rows = len(df_price)
    df_price.dropna(subset=[TARGET_COLUMN_PRICE], inplace=True)
    print(f"Dropped {initial_rows - len(df_price)} rows with NaN target ('{TARGET_COLUMN_PRICE}').")
    initial_rows = len(df_price)
    df_price.dropna(subset=PRICE_FEATURE_COLUMNS, inplace=True)
    print(f"Dropped {initial_rows - len(df_price)} rows with NaN features.")

    print(f"Shape after handling NaNs: {df_price.shape}")

    if df_price.empty:
        print("Error: No data remaining after handling NaNs.")
        sys.exit(1)

    X = df_price[PRICE_FEATURE_COLUMNS]
    y = df_price[TARGET_COLUMN_PRICE]

except Exception as e:
     print(f"Error preparing data: {e}", file=sys.stderr)
     sys.exit(1)

# --- Train/Test Split (Time Series) ---
try:
    print(f"Performing time-series split (testing on last {TEST_SPLIT_WEEKS} GWs)...")
    last_season = df_price['season'].max()
    max_gw_in_data = df_price[df_price['season'] == last_season]['gameweek'].max()
    split_point_gw = max(0, max_gw_in_data - TEST_SPLIT_WEEKS)

    print(f"Identifying split point: Season '{last_season}', GW > {split_point_gw}")

    train_df = df_price[(df_price['season'] < last_season) |
                        ((df_price['season'] == last_season) & (df_price['gameweek'] <= split_point_gw))]
    test_df = df_price[(df_price['season'] == last_season) & (df_price['gameweek'] > split_point_gw)]
    if test_df.empty:
        print("Warning: Test set is empty based on split criteria. Adjust TEST_SPLIT_WEEKS or data range.", file=sys.stderr)
        X_train, y_train = train_df[PRICE_FEATURE_COLUMNS], train_df[TARGET_COLUMN_PRICE]
        X_test, y_test = pd.DataFrame(columns=PRICE_FEATURE_COLUMNS), pd.Series(dtype=y_train.dtype) # Empty test sets
    else:
        X_train = train_df[PRICE_FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN_PRICE]
        X_test = test_df[PRICE_FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN_PRICE]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    if X_train.empty:
         print("Error: Training split resulted in empty data.")
         sys.exit(1)

except Exception as e:
     print(f"Error during train/test split: {e}", file=sys.stderr)
     sys.exit(1)


# --- Train RandomForestRegressor ---
try:
    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,       
        max_depth=15,           
        min_samples_split=10,  
        min_samples_leaf=5,     
        random_state=42,       
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

except Exception as e:
     print(f"Error during model training: {e}", file=sys.stderr)
     sys.exit(1)


# --- Evaluate Model ---
if not X_test.empty:
    try:
        print("Evaluating model on test set...")
        y_pred = rf_model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Price Predictor Evaluation (RandomForestRegressor) ---")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}") 
        print(f"  R²:   {r2:.4f}")

    except Exception as e:
        print(f"Error during model evaluation: {e}", file=sys.stderr)
else:
    print("\nSkipping evaluation: Test set is empty.")


# --- Save Model ---
try:
    print(f"\nSaving trained model to {MODEL_OUTPUT_PATH}...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")

except Exception as e:
    print(f"Error saving model: {e}", file=sys.stderr)


print("\n--- Price Predictor Training Script Finished ---")