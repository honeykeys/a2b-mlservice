# data_processing/load_raw_data.py (Refactored for HTTPS)

import pandas as pd
import requests
from io import StringIO # Used to read string data as a file
import os
import sys
import time # For potential retries/backoff

# --- Configuration ---
# Read Base URL from environment variable set in Task Definition
RAW_DATA_BASE_URL = os.environ.get(
    'RAW_DATA_BASE_URL', # Matches Task Def env var
    'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data' # Default fallback
)
# Define request timeout
REQUEST_TIMEOUT = 30 # seconds

# --- Helper Function ---
def fetch_csv_from_url(url):
    """Fetches CSV content from a URL and returns a Pandas DataFrame."""
    print(f"    Fetching: {url} ...")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check if content is empty before parsing
        if not response.text:
             print(f"    Warning: Empty content received from {url}", file=sys.stderr)
             return pd.DataFrame() # Return empty DataFrame

        # Use StringIO to read the text content as if it were a file
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        print(f"    Successfully fetched and parsed. Shape: {df.shape}")
        return df
    except requests.exceptions.Timeout:
        print(f"    Error: Timeout occurred while fetching {url}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as http_err:
         # Specifically check for 404 Not Found
         if http_err.response.status_code == 404:
              print(f"    Info: File not found (404) at {url}. Skipping.")
              return None
         else:
              print(f"    Error: HTTP error occurred: {http_err} for {url}", file=sys.stderr)
              return None
    except requests.exceptions.RequestException as e:
        print(f"    Error: Failed to fetch {url}: {e}", file=sys.stderr)
        return None
    except pd.errors.ParserError as e_parse:
         print(f"    Error: Failed to parse CSV from {url}: {e_parse}", file=sys.stderr)
         return None # Indicates bad data in the CSV itself
    except Exception as e_gen:
         print(f"    Error: Unexpected error processing {url}: {e_gen}", file=sys.stderr)
         return None

# --- Main Loading Function ---
def load_raw_data_via_https(seasons_to_load, current_season_gw_limit):
    """
    Loads raw FPL data (gameweeks, fixtures, players) for specified seasons
    directly from the vaastav GitHub repo via HTTPS.

    Args:
        seasons_to_load (list): List of season strings (e.g., ['2023-24', '2024-25']).
        current_season_gw_limit (int): Max gameweek for the current/latest season.

    Returns:
        dict: A dictionary containing DataFrames keyed by season and file type,
              or empty dict/None if critical errors occur.
    """
    all_data = {}
    if not seasons_to_load:
        print("Error: No seasons specified for loading.", file=sys.stderr)
        return None

    current_season = seasons_to_load[-1]
    print(f"--- Starting Raw Data Loading via HTTPS (Base URL: {RAW_DATA_BASE_URL}) ---")

    for season in seasons_to_load:
        print(f"\nProcessing season: {season}")
        all_data[season] = {}

        # 1. Load players_raw.csv
        print("  Loading players_raw...")
        players_df = fetch_csv_from_url(f"{RAW_DATA_BASE_URL}/{season}/players_raw.csv")
        if players_df is not None:
             all_data[season]['players'] = players_df
        else:
             print(f"  Critical Error: Failed to load players_raw for {season}", file=sys.stderr)
             # Decide if processing can continue without player data
             # return None # Or maybe continue? Depends on requirements

        # 2. Load fixtures.csv
        print("  Loading fixtures...")
        fixtures_df = fetch_csv_from_url(f"{RAW_DATA_BASE_URL}/{season}/fixtures.csv")
        if fixtures_df is not None:
             all_data[season]['fixtures'] = fixtures_df
        else:
             print(f"  Warning: Failed to load fixtures for {season}", file=sys.stderr)


        # 3. Load Gameweek Data
        print("  Loading gameweek data...")
        gws_list_for_season = []
        # Determine max GW for this season
        max_gw = 38 if season != current_season else current_season_gw_limit

        # Loop through individual gameweek files
        print(f"    Fetching individual GW files up to GW {max_gw}...")
        for gw_num in range(1, max_gw + 1):
            # Short delay between requests to be polite to the source server
            # time.sleep(0.1)
            gw_df = fetch_csv_from_url(f"{RAW_DATA_BASE_URL}/{season}/gws/gw{gw_num}.csv")
            if gw_df is not None:
                # Add gameweek column if missing (less likely with individual files)
                if 'GW' not in gw_df.columns and 'gameweek' not in gw_df.columns:
                    gw_df['gameweek'] = gw_num
                elif 'GW' in gw_df.columns and 'gameweek' not in gw_df.columns:
                    gw_df.rename(columns={'GW': 'gameweek'}, inplace=True)
                # Ensure gameweek column is integer
                if 'gameweek' in gw_df.columns:
                     gw_df['gameweek'] = pd.to_numeric(gw_df['gameweek'], errors='coerce').fillna(gw_num).astype(int)

                gws_list_for_season.append(gw_df)
            # else: fetch_csv_from_url already printed warning/error

        # Concatenate all loaded individual GW DataFrames for the season
        if gws_list_for_season:
            print(f"  Concatenating {len(gws_list_for_season)} loaded gameweek DataFrames for {season}...")
            try:
                all_data[season]['gws'] = pd.concat(gws_list_for_season, ignore_index=True)
                print(f"  Concatenated GW data shape for {season}: {all_data[season]['gws'].shape}")
            except Exception as e_concat:
                print(f"  Error during gameweek concatenation for {season}: {e_concat}", file=sys.stderr)
                all_data[season]['gws'] = pd.DataFrame() # Assign empty DF on error
        else:
             print(f"  Error: No individual gameweek files could be loaded for {season}.", file=sys.stderr)
             all_data[season]['gws'] = pd.DataFrame() # Assign empty DF

    print("\n--- Raw Data Loading via HTTPS Finished ---")
    # Filter out seasons where essential data (like GWS) might be missing entirely
    valid_data = {s: d for s, d in all_data.items() if 'gws' in d and not d['gws'].empty}
    if not valid_data:
         print("Error: Failed to load essential gameweek data for any specified season.", file=sys.stderr)
         return None

    return valid_data

# Example of how run_scheduled_etl.py might use this:
if __name__ == '__main__':
     print("Running load_raw_data_via_https directly for testing...")
     # These would normally come from the calling script or env vars
     test_seasons = ['2023-24'] # Test with just one season
     test_gw_limit = 38 # Example limit for testing

     # You need to set the environment variable for testing locally:
     # export RAW_DATA_BASE_URL='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
     if 'RAW_DATA_BASE_URL' not in os.environ:
          print("Warning: RAW_DATA_BASE_URL environment variable not set for test.")

     test_raw_data = load_raw_data_via_https(test_seasons, test_gw_limit)

     if test_raw_data:
          print("\nTest load successful. Data dictionary structure:")
          for season, data_types in test_raw_data.items():
               print(f"  Season {season}:")
               for dtype, df in data_types.items():
                    print(f"    {dtype}: {df.shape}")
     else:
          print("\nTest load failed.")