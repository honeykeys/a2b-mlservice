import pandas as pd
import os
import sys # Optional: for exiting or writing to stderr

BASE_DATA_PATH = '/Users/karlnuyda/Desktop/Fantasy-Premier-League/data'

# Define the seasons and the GW limit for the current season
SEASONS = ['2023-24', '2024-25']
MAX_GW_CURRENT_SEASON = 33

def load_raw_fpl_data(base_path, seasons_to_load, current_season_gw_limit):
    """
    Loads raw FPL data (gameweeks, fixtures, players) for specified seasons.

    Args:
        base_path (str): The path to the main 'data' directory containing season folders.
        seasons_to_load (list): A list of season strings (e.g., ['2023-24', '2024-25']).
        current_season_gw_limit (int): The max gameweek to load for the most recent season in the list.

    Returns:
        dict: A dictionary containing DataFrames keyed by season and file type
              (e.g., data['2023-24']['gws'], data['2023-24']['fixtures']).
              Returns None or partially filled dict if critical errors occur.
    """
    all_data = {}
    current_season = seasons_to_load[-1] # Assume last season in list is the current one

    print("--- Starting Raw Data Loading ---")

    for season in seasons_to_load:
        print(f"\nProcessing season: {season}")
        all_data[season] = {}
        season_path = os.path.join(base_path, season)

        # 1. Load players_raw.csv
        players_file = os.path.join(season_path, 'players_raw.csv')
        try:
            print(f"  Loading {players_file}...")
            all_data[season]['players'] = pd.read_csv(players_file)
            print(f"  Successfully loaded 'players_raw.csv'. Shape: {all_data[season]['players'].shape}")
            # print(all_data[season]['players'].head(2)) # Optional: view head
            # all_data[season]['players'].info()       # Optional: view info
        except FileNotFoundError:
            print(f"  Error: File not found: {players_file}", file=sys.stderr)
            # Decide if this is critical - perhaps continue for other files?
        except Exception as e:
            print(f"  Error loading {players_file}: {e}", file=sys.stderr)

        # 2. Load fixtures.csv
        fixtures_file = os.path.join(season_path, 'fixtures.csv')
        try:
            print(f"  Loading {fixtures_file}...")
            all_data[season]['fixtures'] = pd.read_csv(fixtures_file)
            print(f"  Successfully loaded 'fixtures.csv'. Shape: {all_data[season]['fixtures'].shape}")
            # print(all_data[season]['fixtures'].head(2))
            # all_data[season]['fixtures'].info()
        except FileNotFoundError:
            print(f"  Error: File not found: {fixtures_file}", file=sys.stderr)
        except Exception as e:
            print(f"  Error loading {fixtures_file}: {e}", file=sys.stderr)

        # --- Replace the EXISTING section #3 in your script with THIS block ---

        # 3. Load Gameweek Data
        gws_list_for_season = [] # Use a list to collect GW DataFrames
        final_gw_df_for_season = None # Will hold the final GW DataFrame for the season

        # --- Try loading merged_gw.csv first ---
        gws_merged_file = os.path.join(season_path, 'gws', 'merged_gw.csv')
        try:
            print(f"  Attempting to load merged file: {gws_merged_file}...")
            gw_df_merged = pd.read_csv(gws_merged_file)
            print(f"  Successfully loaded 'merged_gw.csv'. Initial Shape: {gw_df_merged.shape}")
            # If merged file loads successfully, we'll use it
            final_gw_df_for_season = gw_df_merged
        except FileNotFoundError:
            # This is expected if the merged file doesn't exist yet for the current season
            print(f"  Info: 'merged_gw.csv' not found for {season}. Will try loading individual GW files.")
        except Exception as e:
            # Catch other errors like the 'tokenizing data' error
            print(f"  Warning: Error loading {gws_merged_file}: {e}. Will try loading individual GW files.", file=sys.stderr)

        # --- If merged file failed or didn't exist, load individual GW files ---
        if final_gw_df_for_season is None:
            print(f"  Loading individual gameweek files for {season}...")
            # Determine max GW for this season (full season or limit)
            max_gw = 38 if season != current_season else current_season_gw_limit
            print(f"    (Up to GW {max_gw})")

            for gw_num in range(1, max_gw + 1):
                gw_file = os.path.join(season_path, 'gws', f'gw{gw_num}.csv')
                try:
                    # print(f"      Loading {gw_file}...") # Optional: uncomment for verbose loading
                    temp_df = pd.read_csv(gw_file)
                    # Add gameweek column if it doesn't exist (sometimes needed)
                    if 'GW' not in temp_df.columns and 'gameweek' not in temp_df.columns:
                        temp_df['gameweek'] = gw_num
                    elif 'GW' in temp_df.columns and 'gameweek' not in temp_df.columns:
                        # Standardize column name if needed
                        temp_df.rename(columns={'GW': 'gameweek'}, inplace=True)

                    # Ensure gameweek column is integer
                    if 'gameweek' in temp_df.columns:
                        temp_df['gameweek'] = pd.to_numeric(temp_df['gameweek'], errors='coerce').fillna(gw_num).astype(int)

                    gws_list_for_season.append(temp_df)
                except FileNotFoundError:
                    # It's possible individual GW files might also be missing, especially for very recent GWs
                    print(f"    Warning: Individual file not found: {gw_file}. Skipping.", file=sys.stderr)
                except Exception as e:
                    print(f"    Warning: Error loading {gw_file}: {e}. Skipping.", file=sys.stderr)

            # Concatenate all loaded individual GW DataFrames
            if gws_list_for_season:
                print(f"  Concatenating {len(gws_list_for_season)} loaded gameweek DataFrames for {season}...")
                try:
                    final_gw_df_for_season = pd.concat(gws_list_for_season, ignore_index=True)
                    print(f"  Concatenated GW data shape for {season}: {final_gw_df_for_season.shape}")
                except Exception as e:
                    print(f"  Error during concatenation for {season}: {e}", file=sys.stderr)
                    final_gw_df_for_season = None # Ensure it's None if concat fails
            else:
                print(f"  Error: No individual gameweek files could be loaded for {season}.", file=sys.stderr)


        # --- Filter the final DataFrame if it's the current season ---
        # Make sure we actually have a DataFrame before filtering
        if final_gw_df_for_season is not None and season == current_season:
            # Ensure 'gameweek' column exists before filtering
            if 'gameweek' in final_gw_df_for_season.columns:
                print(f"  Filtering {season} concatenated data up to GW {current_season_gw_limit}...")
                initial_rows = len(final_gw_df_for_season)
                # Ensure the column is numeric before comparison
                final_gw_df_for_season['gameweek'] = pd.to_numeric(final_gw_df_for_season['gameweek'], errors='coerce')
                final_gw_df_for_season.dropna(subset=['gameweek'], inplace=True) # Drop rows where conversion failed
                final_gw_df_for_season['gameweek'] = final_gw_df_for_season['gameweek'].astype(int)

                final_gw_df_for_season = final_gw_df_for_season[final_gw_df_for_season['gameweek'] <= current_season_gw_limit].copy()
                print(f"  Shape after filtering GWs (Initial: {initial_rows}): {final_gw_df_for_season.shape}")
            else:
                print(f"  Warning: Could not find 'gameweek' column in the final DataFrame for {season} for filtering.", file=sys.stderr)


        # --- Store the final result ---
        if final_gw_df_for_season is not None and not final_gw_df_for_season.empty:
            all_data[season]['gws'] = final_gw_df_for_season
        else:
            # Key will be missing if no GW data loaded, verification step will catch this
            print(f"  Error: No valid gameweek data loaded or DataFrame empty for season {season}.", file=sys.stderr)

        # --- End of Replacement Block ---

    print("\n--- Raw Data Loading Finished ---")
    return all_data

# --- Main execution block ---
if __name__ == "__main__":
    # Basic check if the base path exists
    if not os.path.isdir(BASE_DATA_PATH):
        print(f"Error: Base data path not found or not a directory: {BASE_DATA_PATH}", file=sys.stderr)
        print("Please update the BASE_DATA_PATH variable in the script.", file=sys.stderr)
        sys.exit(1) # Exit if base path is wrong

    # Load the data
    raw_data_dict = load_raw_fpl_data(BASE_DATA_PATH, SEASONS, MAX_GW_CURRENT_SEASON)

    # Check if data was loaded and potentially proceed
    if raw_data_dict:
        print("\nData loaded into 'raw_data_dict'. Example access:")
        if '2023-24' in raw_data_dict and 'gws' in raw_data_dict['2023-24']:
             print(f"  Shape of 2023-24 gameweek data: {raw_data_dict['2023-24']['gws'].shape}")
        if '2024-25' in raw_data_dict and 'gws' in raw_data_dict['2024-25']:
             print(f"  Shape of 2024-25 gameweek data (up to GW{MAX_GW_CURRENT_SEASON}): {raw_data_dict['2024-25']['gws'].shape}")
        # You can now pass raw_data_dict to your cleaning/merging function (Task 2)
    else:
        print("\nData loading encountered critical errors.", file=sys.stderr)

# ... (keep the imports and the load_raw_fpl_data function as before) ...

# --- Main execution block ---
if __name__ == "__main__":
    # Basic check if the base path exists
    if not os.path.isdir(BASE_DATA_PATH):
        print(f"Error: Base data path not found or not a directory: {BASE_DATA_PATH}", file=sys.stderr)
        print("Please update the BASE_DATA_PATH variable in the script.", file=sys.stderr)
        sys.exit(1) # Exit if base path is wrong

    # Load the data
    raw_data_dict = load_raw_fpl_data(BASE_DATA_PATH, SEASONS, MAX_GW_CURRENT_SEASON)

    # --- Add Verification Logic Here ---
    print("\n--- Verifying Loaded Data ---")
    if raw_data_dict and isinstance(raw_data_dict, dict):
        print("Successfully received a dictionary.")
        overall_success = True
        for season in SEASONS:
            print(f"\nChecking season: {season}")
            if season in raw_data_dict and isinstance(raw_data_dict[season], dict):
                for file_key in ['gws', 'fixtures', 'players']:
                    print(f"  Checking '{file_key}' data...")
                    if file_key in raw_data_dict[season]:
                        df = raw_data_dict[season][file_key]
                        if isinstance(df, pd.DataFrame):
                            if not df.empty:
                                print(f"    ✅ DataFrame found and is not empty. Shape: {df.shape}")
                                # Optional: Print first row to visually inspect
                                # print("      First row sample:")
                                # print(df.head(1))
                            else:
                                print(f"    ⚠️ DataFrame found but it IS EMPTY.")
                                overall_success = False # Treat empty as possible issue?
                        else:
                            print(f"    ❌ Error: Expected a Pandas DataFrame for '{file_key}', but got {type(df)}.")
                            overall_success = False
                    else:
                        print(f"    ❌ Error: Data for '{file_key}' not found in season {season}.")
                        overall_success = False # Mark as failure if essential data is missing
            else:
                print(f"  ❌ Error: Data for season {season} not found or not a dictionary.")
                overall_success = False

        if overall_success:
             print("\n--- Verification Complete: All expected DataFrames seem to be loaded correctly! ---")
        else:
             print("\n--- Verification Complete: Some issues found during verification. Check messages above. ---")

    else:
        print("\n--- Verification Failed: Script did not return a valid dictionary. ---", file=sys.stderr)

    # Now you know if raw_data_dict is populated correctly before potentially passing it on.