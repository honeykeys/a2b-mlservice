# modeling/train_price_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split # Using simple split for now
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import sys
import numpy as np # For RMSE calculation if needed, or other checks

print("--- Starting Price Predictor Model Training Script ---")

# --- Configuration ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'fpl_analytics_mvp_v1.parquet'
    MODEL_OUTPUT_DIR = PROJECT_ROOT / 'models'
    MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / 'price_predictor_rf_v1.joblib'

    # <<< --- CRITICAL: Define features relevant for PRICE prediction --- >>>
    # These columns MUST exist in your Parquet file after running run_etl.py
    PRICE_FEATURE_COLUMNS = [
        'transfers_balance_lag_1',
        'net_transfers_roll_3',
        'selected_lag_1',
        'points_lag_1', # Example: Using recent points
        # 'avg_points_last_3',
        'cost', # Current cost might influence change
        'chance_playing_prev_gw_forecast', # Availability forecast from previous week
        # Add other relevant lagged/rolling features you created
    ]
    TARGET_COLUMN_PRICE = 'price_change'

    # Define test set parameters (e.g., use last N GWs of most recent season)
    TEST_SPLIT_WEEKS = 8 # Number of most recent GWs to use for testing

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

    # Verify necessary columns exist
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
    # Select only potentially relevant columns first for efficiency
    cols_to_keep = PRICE_FEATURE_COLUMNS + [TARGET_COLUMN_PRICE, 'gameweek', 'season', 'element'] # Keep element/time for split
    df_price = df[cols_to_keep].copy()

    # CRITICAL: Handle NaNs specifically for price prediction
    # 1. Drop rows where the target variable ('price_change') is NaN
    #    (This removes the last gameweek for each player where change couldn't be calculated)
    initial_rows = len(df_price)
    df_price.dropna(subset=[TARGET_COLUMN_PRICE], inplace=True)
    print(f"Dropped {initial_rows - len(df_price)} rows with NaN target ('{TARGET_COLUMN_PRICE}').")

    # 2. Drop rows where any required *feature* is NaN (due to lagging at start of history)
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
    # Simple split based on last N gameweeks of the latest season available
    last_season = df_price['season'].max()
    max_gw_in_data = df_price[df_price['season'] == last_season]['gameweek'].max()
    # Ensure split point is valid even if max_gw is small
    split_point_gw = max(0, max_gw_in_data - TEST_SPLIT_WEEKS)

    print(f"Identifying split point: Season '{last_season}', GW > {split_point_gw}")

    train_df = df_price[(df_price['season'] < last_season) |
                        ((df_price['season'] == last_season) & (df_price['gameweek'] <= split_point_gw))]
    test_df = df_price[(df_price['season'] == last_season) & (df_price['gameweek'] > split_point_gw)]

    # Check if test set is empty (might happen if TEST_SPLIT_WEEKS is too large)
    if test_df.empty:
        print("Warning: Test set is empty based on split criteria. Adjust TEST_SPLIT_WEEKS or data range.", file=sys.stderr)
        # Optionally use a different split method (e.g., train on first N-1 seasons, test on last season)
        # For now, we'll continue but evaluation won't be possible.
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
    # Start with reasonable hyperparameters, tune later if needed
    # n_jobs=-1 uses all available CPU cores for training
    rf_model = RandomForestRegressor(
        n_estimators=100,       # Number of trees in the forest
        max_depth=15,           # Max depth of each tree (tune this)
        min_samples_split=10,   # Min samples required to split a node (tune this)
        min_samples_leaf=5,     # Min samples required at a leaf node (tune this)
        random_state=42,        # For reproducibility
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

        # Use metrics suitable for regression
        rmse = mean_squared_error(y_test, y_pred, squared=False) # Or use np.sqrt(mean_squared_error(...))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Price Predictor Evaluation (RandomForestRegressor) ---")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}") # Often more interpretable for price (£0.1m units)
        print(f"  R²:   {r2:.4f}")

    except Exception as e:
        print(f"Error during model evaluation: {e}", file=sys.stderr)
else:
    print("\nSkipping evaluation: Test set is empty.")


# --- Save Model ---
try:
    print(f"\nSaving trained model to {MODEL_OUTPUT_PATH}...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure models dir exists
    joblib.dump(rf_model, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")

except Exception as e:
    print(f"Error saving model: {e}", file=sys.stderr)


print("\n--- Price Predictor Training Script Finished ---")