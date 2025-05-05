import pandas as pd
import os
import numpy as np # For potential NaN handling if needed
import sys
# Assume raw_data_dict is passed to this function or script
# raw_data_dict = load_raw_fpl_data(...) # From Task 1

def clean_and_merge_data(raw_data):
    """
    Cleans and merges raw FPL dataframes (gws, players, fixtures)
    across specified seasons.

    Args:
        raw_data (dict): The dictionary of raw dataframes loaded in Task 1.

    Returns:
        pandas.DataFrame or None: A single merged and cleaned DataFrame,
                                 or None if critical steps fail.
    """
    print("\n--- Starting Task 2: Cleaning & Merging ---")

    # --- 1. Combine Data Across Seasons ---
    gws_dfs = []
    players_dfs = []
    fixtures_dfs = []
    seasons_processed = list(raw_data.keys())

    print(f"Combining data for seasons: {seasons_processed}")

    for season in seasons_processed:
        if 'gws' in raw_data[season] and not raw_data[season]['gws'].empty:
            df = raw_data[season]['gws'].copy()
            df['season'] = season # Add season identifier
            gws_dfs.append(df)
        else:
            print(f"Warning: Missing or empty 'gws' data for season {season}", file=sys.stderr)

        if 'players' in raw_data[season] and not raw_data[season]['players'].empty:
            df = raw_data[season]['players'].copy()
            df['season'] = season # Add season identifier
            players_dfs.append(df)
        else:
            print(f"Warning: Missing or empty 'players' data for season {season}", file=sys.stderr)

        if 'fixtures' in raw_data[season] and not raw_data[season]['fixtures'].empty:
            df = raw_data[season]['fixtures'].copy()
            df['season'] = season # Add season identifier
            fixtures_dfs.append(df)
        else:
            print(f"Warning: Missing or empty 'fixtures' data for season {season}", file=sys.stderr)

    # Concatenate into single dataframes
    if not gws_dfs:
        print("Error: No gameweek data found to process.", file=sys.stderr)
        return None

    all_gws_df = pd.concat(gws_dfs, ignore_index=True)
    print(f"Combined GWS DataFrame shape: {all_gws_df.shape}")

    # Combine players and fixtures only if they exist
    all_players_df = pd.concat(players_dfs, ignore_index=True) if players_dfs else pd.DataFrame()
    all_fixtures_df = pd.concat(fixtures_dfs, ignore_index=True) if fixtures_dfs else pd.DataFrame()

    print(f"Combined Players DataFrame shape: {all_players_df.shape}")
    print(f"Combined Fixtures DataFrame shape: {all_fixtures_df.shape}")


    # --- 2. Merge DataFrames ---
    # IMPORTANT: Verify these key column names exist in your actual DataFrames!
    # Check using: all_gws_df.columns, all_players_df.columns, all_fixtures_df.columns
    player_gw_key = 'element'    # Common key for player ID in gw data (e.g., 'element')
    player_info_key = 'id'       # Common key for player ID in players_raw data (e.g., 'id')
    fixture_gw_key = 'fixture'   # Common key for fixture ID in gw data
    fixture_info_key = 'id'      # Common key for fixture ID in fixtures data

    merged_df = all_gws_df

    # --- Merge player information (position, name etc.) ---
    if not all_players_df.empty and player_info_key in all_players_df.columns and player_gw_key in merged_df.columns:
        print(f"Merging player info (using keys: GW '{player_gw_key}', PlayerInfo '{player_info_key}')...")

        # Define columns needed from the player info DataFrame
        # We will NOT include 'season' here as it's not in the raw files.
        player_cols_to_merge = [player_info_key, 'element_type', 'web_name', 'team'] # Add 'first_name', 'second_name' if needed

        # Select the necessary columns and drop duplicates based on the player ID key.
        # This uses the combined player DataFrame which might have multiple entries per player if they were in multiple seasons.
        # drop_duplicates keeps the *first* occurrence found. For stable info like position, this is usually okay.
        # If you needed the absolute latest season's info, you'd sort all_players_df by 'season' descending first.
        players_to_merge = all_players_df[player_cols_to_merge].drop_duplicates(subset=[player_info_key], keep='first')
        print(f"  Using {len(players_to_merge)} unique player entries for merge.")

        merged_df = pd.merge(
            merged_df,
            # Rename the key column in players_to_merge to match the key in merged_df (gw data)
            # Also rename the 'team' column from players_raw to avoid clashing with 'team' in gw data if it exists
            players_to_merge.rename(columns={player_info_key: player_gw_key, 'team': 'player_static_team'}),
            on=player_gw_key, # Merge on the player key column common to both
            how='left'        # Keep all rows from the gameweek data (left DataFrame)
        )
        print(f"Shape after merging player info: {merged_df.shape}")
        # Check how many rows didn't get player info (potential key mismatch or missing players in players_raw)
        print(f"  NaNs introduced in 'element_type' after merge: {merged_df['element_type'].isnull().sum()}")
    else:
        # Print detailed info if merge is skipped
        print("Warning: Skipping player info merge - required DFs or columns missing/empty.", file=sys.stderr)
        if all_players_df.empty: print("  Reason: Combined players DataFrame is empty.")
        if player_info_key not in all_players_df.columns: print(f"  Reason: Player key '{player_info_key}' not in player info columns: {all_players_df.columns}")
        if player_gw_key not in merged_df.columns: print(f"  Reason: Player key '{player_gw_key}' not in gameweek columns: {merged_df.columns}")


    # --- Merge fixture information (difficulty, scores, etc.) ---
    if not all_fixtures_df.empty and fixture_info_key in all_fixtures_df.columns and fixture_gw_key in merged_df.columns:
        print(f"Merging fixture info (using keys: GW '{fixture_gw_key}', FixtureInfo '{fixture_info_key}')...")

        # Define columns needed from the fixture info DataFrame
        fixture_cols_to_merge = [fixture_info_key, 'team_h_difficulty', 'team_a_difficulty', 'team_h_score', 'team_a_score'] # Add others if needed

        # Deduplicate fixture info - should be unique by ID already, but doesn't hurt
        fixtures_to_merge = all_fixtures_df[fixture_cols_to_merge].drop_duplicates(subset=[fixture_info_key], keep='first')
        print(f"  Using {len(fixtures_to_merge)} unique fixture entries for merge.")

        merged_df = pd.merge(
            merged_df,
            fixtures_to_merge.rename(columns={fixture_info_key: fixture_gw_key}), # Rename key to match gw key
            on=fixture_gw_key, # Merge only on fixture key
            how='left'         # Keep all rows from the gameweek data
        )
        print(f"Shape after merging fixture info: {merged_df.shape}")
        # Check for NaNs introduced (potential key mismatch or missing fixtures)
        print(f"  NaNs introduced in 'team_h_difficulty' after merge: {merged_df['team_h_difficulty'].isnull().sum()}")
    else:
        # Print detailed info if merge is skipped
        print("Warning: Skipping fixture info merge - required DFs or columns missing/empty.", file=sys.stderr)
        if all_fixtures_df.empty: print("  Reason: Combined fixtures DataFrame is empty.")
        if fixture_info_key not in all_fixtures_df.columns: print(f"  Reason: Fixture key '{fixture_info_key}' not in fixture info columns: {all_fixtures_df.columns}")
        if fixture_gw_key not in merged_df.columns: print(f"  Reason: Fixture key '{fixture_gw_key}' not in gameweek columns: {merged_df.columns}")


    # --- 3. Basic Cleaning on Merged DataFrame ---
    print("\nPerforming basic cleaning on merged data...")
    cleaned_df = merged_df.copy() # Work on a copy

    # Example: Map element_type to position names
    if 'element_type' in cleaned_df.columns:
        print("Mapping element_type to position names...")
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        cleaned_df['position'] = cleaned_df['element_type'].map(pos_map)
        print(cleaned_df['position'].value_counts(dropna=False)) # Show counts including potential NaNs

    # Example: Convert 'was_home' boolean to integer (1 for True, 0 for False)
    if 'was_home' in cleaned_df.columns:
        print("Converting 'was_home' to integer...")
        cleaned_df['was_home'] = cleaned_df['was_home'].astype(int)

    # Example: Handle NaNs in key numeric columns (ADAPT BASED ON YOUR EVALUATION!)
    # Very basic: fill points-related NaNs with 0 - ** re-evaluate this logic!**
    # Often better to fill NaNs based on minutes played or drop rows if essential IDs are missing.
    cols_to_fill_zero = ['total_points', 'goals_scored', 'assists', 'bonus', 'bps', 'clean_sheets']
    print(f"Attempting to fill NaNs with 0 for columns: {cols_to_fill_zero}")
    for col in cols_to_fill_zero:
        if col in cleaned_df.columns:
            original_nan_count = cleaned_df[col].isnull().sum()
            if original_nan_count > 0:
                cleaned_df[col].fillna(0, inplace=True)
                print(f"  Filled {original_nan_count} NaNs in '{col}' with 0.")

    # Example: Ensure key numeric columns are numeric type
    numeric_cols = ['minutes', 'value', 'influence', 'creativity', 'threat', 'ict_index'] # Add others
    print(f"Converting columns to numeric: {numeric_cols}")
    for col in numeric_cols:
        if col in cleaned_df.columns:
             # errors='coerce' turns non-numeric values into NaN - handle these!
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            nan_count = cleaned_df[col].isnull().sum()
            if nan_count > 0:
                print(f"  Warning: Found {nan_count} NaNs in '{col}' after numeric conversion (or they were already NaN). Consider imputation.", file=sys.stderr)
                # Maybe fill these new NaNs? e.g., cleaned_df[col].fillna(0, inplace=True) - BE CAREFUL!

    # Example: Convert cost to float value in millions
    if 'value' in cleaned_df.columns:
         print("Converting FPL 'value' column to millions...")
         cleaned_df['cost'] = cleaned_df['value'] / 10.0


    # --- Final Checks ---
    print("\n--- Cleaning & Merging Complete ---")
    print(f"Final DataFrame shape: {cleaned_df.shape}")
    print("\nFinal Info:")
    cleaned_df.info()
    print("\nFinal Missing values check (sum):")
    print(cleaned_df.isnull().sum().sort_values(ascending=False))

    return cleaned_df

# --- Example Usage (in a separate script or after Task 1) ---
if __name__ == "__main__":
    # Assume raw_data_dict is loaded from Task 1 execution
    # raw_data_dict = load_raw_fpl_data(BASE_DATA_PATH, SEASONS, MAX_GW_CURRENT_SEASON) # From Task 1

    # Placeholder check - replace with actual loading if running standalone
    if 'raw_data_dict' not in locals() or not raw_data_dict:
        print("Error: raw_data_dict not found. Please run Task 1 first.", file=sys.stderr)
        # In a real workflow, Task 1 would likely populate this variable
        # For testing this script standalone, you might load dummy data or run Task 1
        sys.exit(1)

    if raw_data_dict:
        merged_cleaned_df = clean_and_merge_data(raw_data_dict)

        if merged_cleaned_df is not None:
            print("\nSuccessfully created merged and cleaned DataFrame.")
            # Ready for Task 3: Feature Engineering
            # Display sample for verification
            print("\nSample of final cleaned & merged data:")
            print(merged_cleaned_df[['season', 'gameweek', 'name', 'position', 'cost', 'total_points', 'minutes', 'team_h_difficulty', 'team_a_difficulty']].head())
        else:
            print("\nFailed to create merged and cleaned DataFrame.", file=sys.stderr)