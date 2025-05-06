import pandas as pd
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
import joblib 
from pathlib import Path
import numpy as np

# --- Import functions from other scripts within this package ---
# Assumes this script is run from within the data_processing directory OR
# that the fpl-ml-service root is in the Python path.
# If running from root, imports might be: from data_processing.load_raw_data import ...
try:
    from data_processing.load_raw_data import load_raw_fpl_data
    from clean_and_merge_data import clean_and_merge_data
    from feature_engineering import engineer_features # Assuming this function now returns the full processed_df
except ImportError as e:
     print(f"Error importing modules. Make sure scripts are in the same directory or PYTHONPATH is set.", file=sys.stderr)
     print(f"Ensure an empty __init__.py file exists in the data_processing directory.", file=sys.stderr)
     print(f"Original error: {e}", file=sys.stderr)
     sys.exit(1)


# --- Configuration ---
# Path to the cloned vaastav repo's 'data' folder
# Using Path object for better cross-platform compatibility
BASE_DATA_PATH = Path.home() / 'Desktop' / 'Fantasy-Premier-League' / 'data'

# Define the seasons and the GW limit for the current season
SEASONS = ['2023-24', '2024-25']
MAX_GW_CURRENT_SEASON = 33

# Define where to save the processed data
# Saves inside the current project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assumes run_etl.py is in data_processing/
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
PARQUET_OUTPUT_PATH = PROCESSED_DATA_DIR / 'fpl_analytics_mvp_v1.parquet'

# Define columns for final feature set (X) and target (y) for ML
# These are selected *after* loading the processed Parquet data
TARGET_COLUMN = 'total_points' # The actual points scored in the GW
FEATURE_COLUMNS = [ # Example initial features
    'minutes_lag_1',
    'points_lag_1',
    # 'ict_index_lag_1',
    # 'avg_points_last_3',
    'fdr',
    'was_home',
    # Add 'cost_lag_1' here if created
    # Add one-hot encoded position columns here if created
]
# Define the Gameweek to start the test set (adjust as needed)
TEST_SET_START_GW = 30 # Example: Use GW30-33 of current season for testing
MODEL_DIR = PROJECT_ROOT / 'models'
LR_MODEL_PATH = MODEL_DIR / 'lr_model_v1.joblib'
# --- Add this near your other Path definitions (MODEL_DIR should exist) ---
DT_MODEL_PATH = MODEL_DIR / 'dt_model_v1.joblib'


# --- Main ETL Pipeline Execution ---
if __name__ == "__main__":
    print("--- Starting ETL Pipeline ---")

    # --- Basic Path Check ---
    if not BASE_DATA_PATH.is_dir():
        print(f"Error: Base data path not found: {BASE_DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    # --- Task 1: Load Raw Data ---
    print("\n>>> Running Task 1: Load Raw Data...")
    raw_data = load_raw_fpl_data(str(BASE_DATA_PATH), SEASONS, MAX_GW_CURRENT_SEASON) # Convert Path to string if function expects string

    # --- Task 2: Clean and Merge Data ---
    merged_data = None
    if raw_data:
        print("\n>>> Running Task 2: Clean and Merge Data...")
        merged_data = clean_and_merge_data(raw_data)
    else:
        print("Error: Raw data loading failed. Skipping subsequent steps.", file=sys.stderr)
        sys.exit(1)

    # --- Task 3: Feature Engineering & Target Creation ---
    processed_df = None # Expecting a DataFrame with features + target
    if merged_data is not None and not merged_data.empty:
        print("\n>>> Running Task 3: Feature Engineering & Target Creation...")
        # Assuming engineer_features now returns the full DataFrame BEFORE selecting X, y
        processed_df = engineer_features(merged_data)
    else:
        print("Error: Merged data is empty or None. Skipping subsequent steps.", file=sys.stderr)
        sys.exit(1)

    # --- Task 4: Save Transformed Data to Parquet ---
    if processed_df is not None and not processed_df.empty:
        print("\n>>> Running Task 4: Save Transformed Data to Parquet...")
        try:
            # Create the output directory if it doesn't exist
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            processed_df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
            print(f"Successfully saved processed data to: {PARQUET_OUTPUT_PATH}")
        except Exception as e:
            print(f"Error saving data to Parquet: {e}", file=sys.stderr)
            sys.exit(1) # Treat failure to save as critical maybe?
    else:
        print("Error: Processed DataFrame is empty or None. Skipping save.", file=sys.stderr)
        sys.exit(1)

    # --- Task 5: Load Processed Data & Split for ML ---
    print("\n>>> Running Task 5: Load Processed Data & Split for ML...")
    X_train, y_train, X_test, y_test = (None, None, None, None) # Initialize
    try:
        print(f"Loading data from: {PARQUET_OUTPUT_PATH}")
        loaded_df = pd.read_parquet(PARQUET_OUTPUT_PATH)
        print(f"Loaded processed data shape: {loaded_df.shape}")

        # **Important:** Ensure the target column name matches what was saved
        # Adjust if your engineer_features function named it differently
        if TARGET_COLUMN not in loaded_df.columns:
             alt_target = 'actual_points_gw' # Check for alternative name used in previous examples
             if alt_target in loaded_df.columns:
                  TARGET_COLUMN = alt_target
             else:
                  raise ValueError(f"Target column '{TARGET_COLUMN}' not found in Parquet file.")

        # Verify feature columns exist
        missing_ml_features = [col for col in FEATURE_COLUMNS if col not in loaded_df.columns]
        if missing_ml_features:
             raise ValueError(f"Required feature columns missing in Parquet file: {missing_ml_features}")

        # Ensure 'gameweek' and 'season' columns exist for splitting
        if 'gameweek' not in loaded_df.columns or 'season' not in loaded_df.columns:
             raise ValueError("Columns 'gameweek' and 'season' required for time-series split.")

        # Time-Series Split (Example: Using last few GWs of the current season as test)
        current_season = SEASONS[-1]
        test_condition = (loaded_df['season'] == current_season) & (loaded_df['gameweek'] >= TEST_SET_START_GW)

        train_df = loaded_df[~test_condition].copy()
        test_df = loaded_df[test_condition].copy()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]

        print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

        if X_train.empty or X_test.empty:
             print("Warning: Train or Test set is empty after split!", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Processed Parquet file not found at {PARQUET_OUTPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during data loading/splitting: {e}", file=sys.stderr)
        sys.exit(1)
    # --- ML Modeling Starts Here (Tasks 6+) ---
    print("\n>>> ML Data Ready - Starting Task 6: Train/Evaluate Linear Regression...")
    model_lr = None # Initialize model variable

    # Check if training data is available and not empty
    if X_train is not None and not X_train.empty and y_train is not None and not y_train.empty:
        try:
            # 1. Initialize the Model
            model_lr = LinearRegression()
            print(f"Initialized {model_lr.__class__.__name__}")

            # 2. Train the Model
            print("Training model...")
            model_lr.fit(X_train, y_train)
            print("Training complete.")

            # 3. Make Predictions on Test Set
            # Check if test data is available and not empty
            if X_test is not None and not X_test.empty and y_test is not None and not y_test.empty:
                print("Predicting on test set...")
                y_pred_lr = model_lr.predict(X_test)

                # 4. Evaluate the Model
                print("Evaluating model...")
                rmse_lr = root_mean_squared_error(y_test, y_pred_lr)
                r2_lr = r2_score(y_test, y_pred_lr) # R-squared calculation remains the same


                print("\n--- Linear Regression Evaluation ---")
                print(f"  RMSE (Root Mean Squared Error): {rmse_lr:.4f}")
                print(f"  R-squared (Coefficient of Determination): {r2_lr:.4f}")

                # Optional: Print intercept and coefficients (can be many if features are numerous)
                # print(f"\n  Intercept: {model_lr.intercept_:.4f}")
                # print( "  Coefficients:")
                # for feature, coef in zip(FEATURE_COLUMNS, model_lr.coef_):
                #      print(f"    {feature}: {coef:.4f}")

            else:
                print("Warning: Test data is empty or None. Skipping prediction and evaluation.", file=sys.stderr)

        except Exception as e:
            print(f"Error during Linear Regression training/evaluation: {e}", file=sys.stderr)

    else:
        print("\n--- Skipping ML Model Training: No valid training data available. ---", file=sys.stderr)
# --- Task 7: Save Trained Linear Regression Model ---
    if model_lr: # Only save if the model object exists
        print(f"\n>>> Running Task 7: Save Trained Linear Regression Model to {LR_MODEL_PATH}...")
        try:
             # Ensure the models directory exists
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Save the model object
            joblib.dump(model_lr, LR_MODEL_PATH)
            print(f"Successfully saved Linear Regression model.")

        except Exception as e_save:
            print(f"Error saving Linear Regression model: {e_save}", file=sys.stderr)
        else:
            print("Skipping Task 7: Linear Regression model was not successfully trained.")
# --- Task 8: [Stretch Goal] Train/Eval Decision Tree Regressor ---
        print("\n>>> Running Task 8 (Stretch Goal): Train/Evaluate Decision Tree...")
        model_dt = None # Initialize model variable

        try:
            # 1. Initialize the Model
            # Set max_depth to prevent overfitting initially (e.g., 5 or 10)
            # random_state ensures reproducibility
            dt_max_depth = 7 # Example depth - you can experiment
            model_dt = DecisionTreeRegressor(max_depth=dt_max_depth, random_state=42)
            print(f"Initialized {model_dt.__class__.__name__}(max_depth={dt_max_depth})")

            # 2. Train the Model
            print("Training model...")
            model_dt.fit(X_train, y_train)
            print("Training complete.")

            # 3. Make Predictions on Test Set
            if X_test is not None and not X_test.empty and y_test is not None and not y_test.empty:
                print("Predicting on test set...")
                y_pred_dt = model_dt.predict(X_test)

                # 4. Evaluate the Model
                print("Evaluating model...")
                rmse_dt = root_mean_squared_error(y_test, y_pred_dt)
                r2_dt = r2_score(y_test, y_pred_dt) # R-squared calculation remains the same


                print("\n--- Decision Tree Evaluation ---")
                print(f"  Max Depth: {dt_max_depth}")
                print(f"  RMSE: {rmse_dt:.4f}")
                print(f"  R-squared: {r2_dt:.4f}")

                # 5. Compare with Linear Regression (assuming Task 6 ran successfully)
                print("\n--- Comparison ---")
                try:
                     # Assumes rmse_lr and r2_lr variables exist from Task 6
                     print(f"  Linear Regression: RMSE={rmse_lr:.4f}, R2={r2_lr:.4f}")
                     print(f"  Decision Tree    : RMSE={rmse_dt:.4f}, R2={r2_dt:.4f}")
                except NameError:
                     print("  (Linear Regression results not available for comparison)")
            
            # --- Task 9: [Stretch Goal] Save Trained Decision Tree Model ---
                if model_dt: # Only save if the model object exists and training succeeded
                    print(f"\n>>> Running Task 9 (Stretch Goal): Save Trained Decision Tree Model to {DT_MODEL_PATH}...")
                    try:
                        # Ensure the models directory exists (Task 7 code likely already did this)
                        MODEL_DIR.mkdir(parents=True, exist_ok=True)

                        # Save the model object
                        joblib.dump(model_dt, DT_MODEL_PATH)
                        print(f"Successfully saved Decision Tree model.")

                    except Exception as e_save_dt:
                        print(f"Error saving Decision Tree model: {e_save_dt}", file=sys.stderr)
                else:
                    print("Skipping Task 9: Decision Tree model was not successfully trained.")
            else:
                print("Warning: Test data is empty or None. Skipping prediction and evaluation.", file=sys.stderr)

        except Exception as e:
            print(f"Error during Decision Tree training/evaluation: {e}", file=sys.stderr)

    print("\n--- ML Model Training/Evaluation Block Complete ---")
    print("\n--- ETL Pipeline Finished ---") # This remains the final line of the __main__ block
