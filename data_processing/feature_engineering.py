import pandas as pd
import numpy as np
import sys

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
    base_required_cols = [
        'element', 'season', 'gameweek', 'total_points', 'minutes', 'cost',
        'transfers_in', 'transfers_out', 'transfers_balance', 'selected_by_percent',
        'was_home', 'team_h_difficulty', 'team_a_difficulty', 'position', 'web_name',
        'player_static_team', 'goals_scored', 'assists', 'ict_index'
    ]
    missing_cols = [col for col in base_required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Potentially missing base columns for feature engineering: {missing_cols}", file=sys.stderr)


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
    print("Identifying 'total_points' as points prediction target (alignment happens later).")
    target_col_points = 'total_points'
    if target_col_points not in df_sorted.columns:
         print(f"Error: Target column '{target_col_points}' not found.", file=sys.stderr)


    # --- 6. Engineer Lagged/Rolling Features (MODIFIED to use .transform()) ---
    print("Creating lagged & rolling features using .transform()...")

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

    # Price Predictor Features
    if 'transfers_balance' in df_sorted.columns:
        df_sorted['transfers_balance_lag_1'] = player_group['transfers_balance'].transform(lambda x: x.shift(1))
        df_sorted['net_transfers_roll_3'] = player_group['transfers_balance'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
        )
    else: print("Warning: 'transfers_balance' column not found for lagging/rolling.")

    if 'selected_by_percent' in df_sorted.columns:
        df_sorted['selected_lag_1'] = player_group['selected_by_percent'].transform(lambda x: x.shift(1))
    else: print("Warning: 'selected_by_percent' column not found for lagging.")

    # Lagged Status (if available)
    if 'chance_of_playing_next_round' in df_sorted.columns:
        df_sorted['chance_of_playing_next_round'] = pd.to_numeric(df_sorted['chance_of_playing_next_round'], errors='coerce')
        df_sorted['chance_playing_prev_gw_forecast'] = player_group['chance_of_playing_next_round'].transform(lambda x: x.shift(1))

    # --- End of Feature Engineering Section ---

    # --- 7. Engineer Fixture Difficulty Feature ---
    print("Creating Fixture Difficulty Rating (FDR) feature...")
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
    essential_lagged_cols = [col for col in df_sorted.columns if '_lag_' in col or '_roll_' in col]
    df_processed = df_sorted.copy() # Use the DataFrame with all rows
    print(f"Shape BEFORE any NaN drop in engineer_features: {df_processed.shape}")

    if df_processed.empty:
        print("Error: DataFrame is empty after potential minimal NaN handling.", file=sys.stderr)
        return None

    print(f"Final processed DataFrame shape (engineer_features): {df_processed.shape}")
    print("\n--- Feature Engineering Complete ---")
    return df_processed
