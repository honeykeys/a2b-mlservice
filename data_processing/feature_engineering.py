import pandas as pd
import numpy as np # Needed for np.select
import sys # Needed for error exit/prints

def engineer_features(df):
    """
    Performs feature engineering for FPL points and price prediction.
    Calculates targets, creates lagged/rolling features.

    Args:
        df (pandas.DataFrame): The cleaned and merged DataFrame from Task 2.

    Returns:
        pandas.DataFrame or None:
            A single DataFrame containing identifiers, both target variables
            (total_points, price_change), and all engineered features,
            ready to be saved to Parquet. Returns None if processing fails.
    """
    print("\n--- Starting Task 3: Feature Engineering & Target Creation ---")

    if df is None or df.empty:
        print("Error: Input DataFrame is empty or None.", file=sys.stderr)
        return None

    # --- 1. Ensure Required Base Columns Exist ---
    # Add all columns needed for target/feature calculation BEFORE processing
    base_required_cols = [
        'element', 'season', 'gameweek', 'total_points', 'minutes', 'cost',
        'transfers_in', 'transfers_out', 'transfers_balance', 'selected_by_percent',
        'was_home', 'team_h_difficulty', 'team_a_difficulty', 'position', 'web_name',
        'player_static_team', 'goals_scored', 'assists', 'ict_index' # Add others like goals_scored, assists, ict_index etc. if available and needed
    ]
    missing_cols = [col for col in base_required_cols if col not in df.columns]
    # Allow some to be missing but warn (e.g., maybe ICT index wasn't in all files)
    if missing_cols:
        print(f"Warning: Potentially missing base columns for feature engineering: {missing_cols}", file=sys.stderr)
        # Decide if any are absolutely critical and raise an error if so


    # --- 2. Sort Data Chronologically per Player ---
    print("Sorting data by player and time...")
    df_sorted = df.sort_values(by=['element', 'season', 'gameweek']).copy()


    # --- 3. Define Player Grouping --- # <<< MOVED UP & DEFINED ONCE >>>
    print("Grouping data by player ('element')...")
    # Assuming 'element' is the unique player ID across relevant seasons
    player_group = df_sorted.groupby('element')


    # --- 4. Calculate Target Variable 1: Price Change --- # <<< PP.1 >>>
    print("Calculating price_change target variable...")
    if 'cost' in df_sorted.columns:
        df_sorted['cost'] = pd.to_numeric(df_sorted['cost'], errors='coerce')
        df_sorted['cost_next_gw'] = player_group['cost'].shift(-1)
        df_sorted['price_change'] = df_sorted['cost_next_gw'] - df_sorted['cost']
        df_sorted.drop(columns=['cost_next_gw'], inplace=True)
        price_change_nans = df_sorted['price_change'].isnull().sum()
        print(f"  Calculated 'price_change'. Found {price_change_nans} NaNs (expected for last GW of players).")
    else:
        print("Warning: 'cost' column not found, cannot calculate price_change.", file=sys.stderr)


    # --- 5. Define Target Variable 2: Points ---
    # The 'total_points' column itself serves as the target for points prediction,
    # it just needs to be aligned with features from the *previous* gameweek.
    print("Identifying 'total_points' as points prediction target (alignment happens later).")
    target_col_points = 'total_points'
    if target_col_points not in df_sorted.columns:
         print(f"Error: Target column '{target_col_points}' not found.", file=sys.stderr)
         # Maybe return None here if points prediction is essential?

    # --- 6. Engineer Lagged/Rolling Features (MODIFIED to use .transform()) ---
    print("Creating lagged & rolling features using .transform()...")

    # Define player_group if not already defined just above
    # player_group = df_sorted.groupby('element')

    # Point Predictor Features (Examples using transform)
    if 'total_points' in df_sorted.columns:
        df_sorted['points_lag_1'] = player_group['total_points'].transform(lambda x: x.shift(1))
    if 'minutes' in df_sorted.columns:
        df_sorted['minutes_lag_1'] = player_group['minutes'].transform(lambda x: x.shift(1))
    if 'goals_scored' in df_sorted.columns:
        df_sorted['goals_lag_1'] = player_group['goals_scored'].transform(lambda x: x.shift(1))
    if 'assists' in df_sorted.columns:
        df_sorted['assists_lag_1'] = player_group['assists'].transform(lambda x: x.shift(1))
    if 'ict_index' in df_sorted.columns:
        df_sorted['ict_index_lag_1'] = player_group['ict_index'].transform(lambda x: x.shift(1))

    # Price Predictor Features (Examples using transform)
    if 'transfers_balance' in df_sorted.columns:
        df_sorted['transfers_balance_lag_1'] = player_group['transfers_balance'].transform(lambda x: x.shift(1))
        # Calculate rolling transfers using transform. Roll first, then shift result.
        df_sorted['net_transfers_roll_3'] = player_group['transfers_balance'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
        )
    else: print("Warning: 'transfers_balance' column not found for lagging/rolling.")

    if 'selected_by_percent' in df_sorted.columns:
        df_sorted['selected_lag_1'] = player_group['selected_by_percent'].transform(lambda x: x.shift(1))
    else: print("Warning: 'selected_by_percent' column not found for lagging.")

    # Lagged Status (if available)
    if 'chance_of_playing_next_round' in df_sorted.columns:
        # Ensure it's numeric first if needed, handle potential non-numeric entries
        df_sorted['chance_of_playing_next_round'] = pd.to_numeric(df_sorted['chance_of_playing_next_round'], errors='coerce')
        df_sorted['chance_playing_prev_gw_forecast'] = player_group['chance_of_playing_next_round'].transform(lambda x: x.shift(1))
        # Consider how NaNs introduced by coerce should be handled before lagging/using
    # else: print("Warning: 'chance_of_playing_next_round' not found for lagging.")

    # --- End of Feature Engineering Section ---

    # --- 7. Engineer Fixture Difficulty Feature ---
    print("Creating Fixture Difficulty Rating (FDR) feature...")
    # Ensure needed columns from merge exist
    if 'was_home' in df_sorted.columns and 'team_h_difficulty' in df_sorted.columns and 'team_a_difficulty' in df_sorted.columns:
        df_sorted['was_home'] = pd.to_numeric(df_sorted['was_home'], errors='coerce').fillna(-1).astype(int) # Ensure 0/1
        conditions_fdr = [
            df_sorted['was_home'] == 1, # Condition for home game
            df_sorted['was_home'] == 0  # Condition for away game
        ]
        choices_fdr = [
            df_sorted['team_h_difficulty'],
            df_sorted['team_a_difficulty']
        ]
        df_sorted['fdr'] = np.select(conditions_fdr, choices_fdr, default=np.nan) # Use NaN default for safety
    else:
        print("Warning: Missing columns needed for FDR calculation (was_home, team_h_difficulty, team_a_difficulty). Setting FDR to NaN.", file=sys.stderr)
        df_sorted['fdr'] = np.nan


    # --- 8. Handle NaNs Introduced ONLY by Lagging/Rolling ---
    # Identify all columns created by shifting or rolling (these define the initial rows to drop)
    # Ensure this list includes key features needed for BOTH model types later
    essential_lagged_cols = [col for col in df_sorted.columns if '_lag_' in col or '_roll_' in col]
    # Add specific essential lagged cols if not captured by name pattern:
    # essential_lagged_cols.extend(['chance_playing_prev_gw_forecast']) # Example

    # Only proceed if we actually created lagged columns
    if essential_lagged_cols:
        print(f"Dropping rows with NaN values in essential lagged/rolled features: {essential_lagged_cols}")
        initial_rows = len(df_sorted)
        # Use the identified list to drop rows missing crucial historical context
        df_processed = df_sorted.dropna(subset=essential_lagged_cols).copy()
        rows_dropped = initial_rows - len(df_processed)
        print(f"Dropped {rows_dropped} rows due to NaNs in lagged/rolled features.")
    else:
        print("Warning: No lagged/rolled features identified for NaN handling.")
        df_processed = df_sorted # Use df_sorted if no lagging was done

    if df_processed.empty:
        print("Error: DataFrame is empty after handling NaNs from lagging.", file=sys.stderr)
        return None

    # --- 9. Final Type Checks / Cleanup (Optional) ---
    # Example: Convert boolean was_home back if needed, ensure numeric types
    # df_processed['was_home'] = df_processed['was_home'].astype(bool) # If needed later
    # ...

    print(f"Final processed DataFrame shape: {df_processed.shape}")
    print("\n--- Feature Engineering Complete ---")
    # RETURN THE FULL DATAFRAME containing everything needed
    # Selection of X/y for specific models happens later in training scripts
    return df_processed