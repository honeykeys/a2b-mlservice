# clean_merge_data.py

import pandas as pd
import numpy as np
import os
import sys

def clean_and_merge_data(raw_data):
    """
    Cleans and merges raw FPL dataframes (gws, players, fixtures)
    across specified seasons into a single DataFrame ready for feature engineering.

    Args:
        raw_data (dict): The dictionary of raw dataframes loaded in Task 1.
                         Expected keys per season: 'gws', 'players', 'fixtures'.

    Returns:
        pandas.DataFrame or None: A single merged and cleaned DataFrame,
                                 or None if critical steps fail.
    """
    print("\n--- Starting Task 2: Cleaning & Merging ---")

    if not raw_data:
        print("Error: Input raw_data dictionary is empty or None.", file=sys.stderr)
        return None

    # --- Initial Debug Print of Raw Columns ---
    print("DEBUG: Checking raw input columns...")
    try:
        # Use last season loaded as example - assumes structure is consistent
        last_season = sorted(list(raw_data.keys()))[-1]
        raw_players_df = raw_data.get(last_season, {}).get('players')
        raw_gws_df = raw_data.get(last_season, {}).get('gws')
        raw_fixtures_df = raw_data.get(last_season, {}).get('fixtures')

        if raw_players_df is not None:
            print(f"  Raw 'players' columns ({last_season}):", raw_players_df.columns.tolist())
        else: print(f"  Raw 'players' data missing for season {last_season}.")
        if raw_gws_df is not None:
             print(f"  Raw 'gws' columns ({last_season}):", raw_gws_df.columns.tolist())
        else: print(f"  Raw 'gws' data missing for season {last_season}.")
        if raw_fixtures_df is not None:
             print(f"  Raw 'fixtures' columns ({last_season}):", raw_fixtures_df.columns.tolist())
        else: print(f"  Raw 'fixtures' data missing for season {last_season}.")
    except Exception as e_debug:
        print(f"  DEBUG Error checking raw columns: {e_debug}")
    # --- End Debug Prints ---


    # --- 1. Combine Data Across Seasons ---
    gws_dfs = []
    players_dfs = []
    fixtures_dfs = []
    seasons_processed = sorted(list(raw_data.keys())) # Process in order

    print(f"\nCombining data for seasons: {seasons_processed}")

    for season in seasons_processed:
        print(f"  Processing {season}...")
        if 'gws' in raw_data[season] and isinstance(raw_data[season]['gws'], pd.DataFrame) and not raw_data[season]['gws'].empty:
            df_gws = raw_data[season]['gws'].copy()
            df_gws['season'] = season
            gws_dfs.append(df_gws)
        else: print(f"  Warning: Missing or empty 'gws' data for season {season}", file=sys.stderr)

        # We likely only need one definitive source for player/fixture info, often the latest season's is sufficient
        # unless merging season-specific details. Let's plan to use the *last* season's players/fixtures for merging.
        # Alternatively, concat all and deduplicate if IDs are consistent. Concat/dedupe is safer.
        if 'players' in raw_data[season] and isinstance(raw_data[season]['players'], pd.DataFrame) and not raw_data[season]['players'].empty:
            df_players = raw_data[season]['players'].copy()
            df_players['season'] = season
            players_dfs.append(df_players)
        else: print(f"  Warning: Missing or empty 'players' data for season {season}", file=sys.stderr)

        if 'fixtures' in raw_data[season] and isinstance(raw_data[season]['fixtures'], pd.DataFrame) and not raw_data[season]['fixtures'].empty:
            df_fixtures = raw_data[season]['fixtures'].copy()
            df_fixtures['season'] = season
            fixtures_dfs.append(df_fixtures)
        else: print(f"  Warning: Missing or empty 'fixtures' data for season {season}", file=sys.stderr)

    # Concatenate into single dataframes
    if not gws_dfs:
        print("Error: No gameweek data found to process.", file=sys.stderr)
        return None

    all_gws_df = pd.concat(gws_dfs, ignore_index=True)
    print(f"Combined GWS DataFrame shape: {all_gws_df.shape}")

    all_players_df = pd.concat(players_dfs, ignore_index=True) if players_dfs else pd.DataFrame()
    all_fixtures_df = pd.concat(fixtures_dfs, ignore_index=True) if fixtures_dfs else pd.DataFrame()

    print(f"Combined Players DataFrame shape: {all_players_df.shape}")
    print(f"Combined Fixtures DataFrame shape: {all_fixtures_df.shape}")


    # --- 2. Merge DataFrames ---
    print("\nMerging datasets...")
    # --- CHECKPOINT: Verify these key column names match your RAW data output ---
    player_gw_key = 'element'           # Key for player in gameweek data
    player_info_key = 'id'              # Key for player in players_raw data
    fixture_gw_key = 'fixture'          # Key for fixture in gameweek data
    fixture_info_key = 'id'             # Key for fixture in fixtures data
    ownership_col_raw = 'selected_by_percent' # Source column for ownership % (Check raw players cols)
    play_chance_col_raw = 'chance_of_playing_next_round' # Source column for play chance (Check raw players cols)
    player_team_col_raw = 'team'        # Source column for team ID in players_raw
    player_name_col_raw = 'web_name'    # Source column for player display name
    position_col_raw = 'element_type'   # Source column for player position ID

    merged_df = all_gws_df

    # --- Merge player information ---
    if not all_players_df.empty and player_info_key in all_players_df.columns and player_gw_key in merged_df.columns:
        print(f"  Preparing player info merge...")

        # --- CHECKPOINT: Ensure ALL columns needed exist in all_players_df and are listed here ---
        player_cols_to_select = [
            player_info_key,
            position_col_raw,
            player_name_col_raw,
            player_team_col_raw,
            ownership_col_raw, # Make sure this exact name exists
            play_chance_col_raw # Make sure this exact name exists
        ]
        # Filter list based on actual columns present to avoid KeyErrors
        actual_player_cols_to_select = [col for col in player_cols_to_select if col in all_players_df.columns]
        if len(actual_player_cols_to_select) < len(player_cols_to_select):
             print(f"Warning: Could not find all desired player columns! Missing: {set(player_cols_to_select) - set(actual_player_cols_to_select)}", file=sys.stderr)

        if player_info_key not in actual_player_cols_to_select:
             print(f"Error: Player info key '{player_info_key}' not found in player data columns.", file=sys.stderr)
             return None # Cannot merge without the key

        player_info_subset = all_players_df[actual_player_cols_to_select]

        # Deduplicate: Get the most recent info for each player ID if seasons were combined
        if 'season' in player_info_subset.columns:
             player_info_subset = player_info_subset.sort_values(by='season', ascending=False)
        players_to_merge = player_info_subset.drop_duplicates(subset=[player_info_key], keep='first')

        print(f"  DEBUG: Columns in players_to_merge:", players_to_merge.columns.tolist())

        # Rename columns before merge to avoid clashes
        rename_dict = {player_info_key: player_gw_key, player_team_col_raw: 'player_static_team'}
        # Add renames for ownership/chance if their raw names differ from desired final names
        # Example: if raw name is 'selected', but we want 'selected_by_percent'
        # if ownership_col_raw == 'selected' and 'selected' in players_to_merge.columns :
        #     rename_dict['selected'] = 'selected_by_percent'
        # if play_chance_col_raw == 'status' and 'status' in players_to_merge.columns:
        #      rename_dict['status'] = 'chance_of_playing_next_round' # Careful, status might mean something else

        players_to_merge = players_to_merge.rename(columns=rename_dict)

        merged_df = pd.merge(
            merged_df,
            players_to_merge, # Use the renamed, deduplicated, selected columns
            on=player_gw_key,
            how='left'
        )
        print(f"Shape after merging player info: {merged_df.shape}")
        print(f"  DEBUG: Columns AFTER player merge:", merged_df.columns.tolist())
        # Check if the target columns (using final names) are now present
        print(f"    'selected_by_percent' present: {'selected_by_percent' in merged_df.columns}")
        print(f"    'chance_of_playing_next_round' present: {'chance_of_playing_next_round' in merged_df.columns}")

    else:
        print("Warning: Skipping player info merge - required DFs or columns missing/empty.", file=sys.stderr)

    # --- Merge fixture information ---
    if not all_fixtures_df.empty and fixture_info_key in all_fixtures_df.columns and fixture_gw_key in merged_df.columns:
        print(f"\n  Merging fixture info...")
        # --- CHECKPOINT: Add any other fixture columns needed ---
        fixture_cols_to_select = [fixture_info_key, 'team_h_difficulty', 'team_a_difficulty', 'team_h_score', 'team_a_score']
        actual_fixture_cols = [col for col in fixture_cols_to_select if col in all_fixtures_df.columns]
        if len(actual_fixture_cols) < len(fixture_cols_to_select):
             print(f"Warning: Could not find all desired fixture columns! Missing: {set(fixture_cols_to_select) - set(actual_fixture_cols)}", file=sys.stderr)

        if fixture_info_key not in actual_fixture_cols:
             print(f"Error: Fixture info key '{fixture_info_key}' not found.", file=sys.stderr)
        else:
            fixtures_to_merge = all_fixtures_df[actual_fixture_cols].drop_duplicates(subset=[fixture_info_key], keep='first')
            merged_df = pd.merge(
                merged_df,
                fixtures_to_merge.rename(columns={fixture_info_key: fixture_gw_key}),
                on=fixture_gw_key,
                how='left'
            )
            print(f"Shape after merging fixture info: {merged_df.shape}")
            print(f"  DEBUG: Columns AFTER fixture merge:", merged_df.columns.tolist())
    else:
        print("Warning: Skipping fixture info merge - required DFs or columns missing/empty.", file=sys.stderr)


    # --- 3. Basic Cleaning on Merged DataFrame ---
    # Assign final DataFrame name
    cleaned_merged_df = merged_df.copy()
    print("\nPerforming basic cleaning on merged data...")

    # Map position ID to Name
    if position_col_raw in cleaned_merged_df.columns: # Use the raw name before potential rename/drop
        print("Mapping element_type to position names...")
        pos_map = {1.0: 'GK', 2.0: 'DEF', 3.0: 'MID', 4.0: 'FWD'} # Use floats if type is float
        # Ensure column is numeric before mapping
        cleaned_merged_df[position_col_raw] = pd.to_numeric(cleaned_merged_df[position_col_raw], errors='coerce')
        cleaned_merged_df['position'] = cleaned_merged_df[position_col_raw].map(pos_map)
        print(f"  Position counts:\n{cleaned_merged_df['position'].value_counts(dropna=False)}")

    # Convert 'was_home' boolean/object to integer
    if 'was_home' in cleaned_merged_df.columns:
        print("Converting 'was_home' to integer...")
        # Handle potential True/False strings or bools
        cleaned_merged_df['was_home'] = cleaned_merged_df['was_home'].replace({True: 1, False: 0})
        # Convert anything else non-numeric to NaN, then fill NaNs (e.g., with -1 or median?), then convert to int
        cleaned_merged_df['was_home'] = pd.to_numeric(cleaned_merged_df['was_home'], errors='coerce').fillna(-1).astype(int)

    # Convert FPL 'value'/'cost' to float value in millions
    # 'value' usually from GW data, 'cost' added previously from raw player data
    if 'value' in cleaned_merged_df.columns:
         print("Converting FPL 'value' column to millions and renaming to 'cost'...")
         cleaned_merged_df['cost'] = pd.to_numeric(cleaned_merged_df['value'], errors='coerce') / 10.0
         # Drop original 'value' maybe? Or keep it? Let's keep both for now unless name clash
         # If 'cost' already exists from player merge, decide which one to keep or rename 'value' to 'gw_cost'
         # Let's assume we prioritize 'value' from GW data if present
         if 'cost' in cleaned_merged_df.columns and 'value' in cleaned_merged_df.columns:
             print("  'cost' column also exists, overwriting with value/10.")

    # Basic NaN handling (EXAMPLE ONLY - MUST CUSTOMIZE)
    cols_to_fill_zero = ['total_points', 'goals_scored', 'assists', 'bonus', 'bps', 'clean_sheets', 'minutes'] # Add more if 0 makes sense
    print(f"Attempting basic NaN fill (0) for columns: {cols_to_fill_zero}")
    for col in cols_to_fill_zero:
        if col in cleaned_merged_df.columns:
            original_nan_count = cleaned_merged_df[col].isnull().sum()
            if original_nan_count > 0:
                cleaned_merged_df[col].fillna(0, inplace=True)
                print(f"  Filled {original_nan_count} NaNs in '{col}' with 0.")

    # Ensure other key numeric columns are numeric type
    numeric_cols = ['ict_index', 'influence', 'creativity', 'threat', 'selected_by_percent', 'chance_of_playing_next_round'] # Add others
    print(f"Ensuring columns are numeric: {numeric_cols}")
    for col in numeric_cols:
        if col in cleaned_merged_df.columns:
            original_dtype = cleaned_merged_df[col].dtype
            cleaned_merged_df[col] = pd.to_numeric(cleaned_merged_df[col], errors='coerce')
            new_dtype = cleaned_merged_df[col].dtype
            nan_count = cleaned_merged_df[col].isnull().sum()
            if nan_count > 0:
                print(f"  Warning: Found {nan_count} NaNs in '{col}' after numeric conversion. Consider imputation.", file=sys.stderr)
            if original_dtype != new_dtype:
                 print(f"  Converted '{col}' dtype from {original_dtype} to {new_dtype}")


    # --- Final Checks ---
    print("\n--- Cleaning & Merging Complete ---")
    print(f"Final DataFrame shape: {cleaned_merged_df.shape}")
    print("\nColumns AFTER clean/merge (BEFORE RETURN):", cleaned_merged_df.columns.tolist())
    print(f"  'selected_by_percent' present: {'selected_by_percent' in cleaned_merged_df.columns}")
    print(f"  'chance_of_playing_next_round' present: {'chance_of_playing_next_round' in cleaned_merged_df.columns}")

    return cleaned_merged_df


# --- Example Usage Block (if running this file directly for testing) ---
# You would typically call this function from run_etl.py
if __name__ == '__main__':
     # This block is for testing this script directly
     # You would need to load raw data first or use sample data
     print("This script is intended to be called with loaded raw_data_dict.")
     print("Running with placeholder data structure for demonstration.")

     # Create dummy data structure similar to Task 1 output
     dummy_raw_data = {
         '2023-24': {
             'players': pd.DataFrame({
                 'id': [1, 2, 3], 'element_type': [1, 2, 3], 'web_name': ['PlayerA', 'PlayerB', 'PlayerC'],
                 'team': [10, 11, 10], 'selected_by_percent': [5.5, 10.1, 2.3], 'chance_of_playing_next_round': [100, 75, None],
                 'now_cost': [50, 65, 70] # Original cost/value before division
             }),
             'fixtures': pd.DataFrame({
                 'id': [101, 102, 103], 'team_h_difficulty': [2, 3, 4], 'team_a_difficulty': [4, 3, 2],
                 'team_h_score': [1, 2, 0], 'team_a_score': [1, 0, 2]
             }),
             'gws': pd.DataFrame({
                 'element': [1, 2, 3, 1, 2, 3], 'fixture': [101, 101, 101, 102, 102, 102], 'gameweek': [1, 1, 1, 2, 2, 2],
                 'total_points': [2, 6, 1, 3, 0, 5], 'minutes': [90, 90, 20, 90, 0, 75], 'cost': [50, 65, 70, 51, 65, 71],
                 'transfers_balance': [100, 500, -10, 200, -500, 150], 'selected': [55000, 101000, 23000, 60000, 95000, 25000], # Example raw selected count
                 'was_home': [True, False, False, False, True, True], 'opponent_team': [11, 10, 11, 10, 11, 10],
                 'value': [50, 65, 70, 51, 65, 71] # Example GW value
             })
         }
     }
     # Add more checks if needed
     if 'value' in dummy_raw_data['2023-24']['gws'].columns:
        print("Using 'value' from GW data as primary 'cost' source.")
        dummy_raw_data['2023-24']['gws']['cost'] = dummy_raw_data['2023-24']['gws']['value']


     merged_cleaned_df = clean_and_merge_data(dummy_raw_data)

     if merged_cleaned_df is not None:
         print("\nDummy run successful.")
         print(merged_cleaned_df.head())
     else:
         print("\nDummy run failed.")