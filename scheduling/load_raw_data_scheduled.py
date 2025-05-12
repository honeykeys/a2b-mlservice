import pandas as pd
import requests
from io import StringIO
import os
import sys
import time
import logging
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
RAW_DATA_BASE_URL = os.environ.get(
    'RAW_DATA_BASE_URL',
    'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
)
REQUEST_TIMEOUT = 30
if not RAW_DATA_BASE_URL:
    logging.critical("FATAL: RAW_DATA_BASE_URL environment variable not set.")
    sys.exit("Exiting: Missing RAW_DATA_BASE_URL.")
elif not RAW_DATA_BASE_URL.startswith(('http://', 'https://')):
     logging.critical(f"FATAL: RAW_DATA_BASE_URL is missing scheme (http/https): {RAW_DATA_BASE_URL}")
     sys.exit("Exiting: Invalid RAW_DATA_BASE_URL configuration.")
cleaned_base_url = RAW_DATA_BASE_URL.rstrip('/')


# --- Helper Function ---
def fetch_csv_from_url(file_path_suffix):
    """Fetches CSV content from a URL (base_url + suffix) and returns a Pandas DataFrame."""
    url = f"{cleaned_base_url}/{file_path_suffix.lstrip('/')}"
    logging.info(f"    Attempting to fetch: {url} ...")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        if not response.text:
             logging.warning(f"    Warning: Empty content received from {url}")
             return pd.DataFrame()

        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        logging.info(f"    Successfully fetched and parsed {file_path_suffix}. Shape: {df.shape}")
        return df
    except requests.exceptions.Timeout:
        logging.error(f"    Error: Timeout occurred while fetching {url}")
        return None
    except requests.exceptions.HTTPError as http_err:
         if http_err.response.status_code == 404:
              logging.info(f"    Info: File not found (404) at {url}. Will attempt shell creation if needed.")
              return None
         else:
              logging.error(f"    Error: HTTP error occurred: {http_err} for {url}")
              return None
    except requests.exceptions.RequestException as e:
        logging.error(f"    Error: Failed to fetch {url}: {e}")
        return None
    except pd.errors.ParserError as e_parse:
         logging.error(f"    Error: Failed to parse CSV from {url}: {e_parse}")
         return None
    except Exception as e_gen:
         logging.error(f"    Error: Unexpected error processing {url}: {e_gen}")
         return None

# --- Main Loading Function ---
def load_raw_data_via_https(seasons_to_load, current_season_gw_limit):
    """
    Loads raw FPL data for specified seasons.
    For gameweeks where actual data CSV is missing (e.g., future GWs),
    it constructs a "shell" DataFrame with player and fixture info.
    """
    all_data = {}
    if not seasons_to_load:
        logging.error("Error: No seasons specified for loading.")
        return None
    current_season = seasons_to_load[-1]
    logging.info(f"--- Starting Raw Data Loading via HTTPS (Base URL: {cleaned_base_url}) ---")

    for season in seasons_to_load:
        logging.info(f"\nProcessing season: {season}")
        all_data[season] = {}

        # 1. Load players_raw.csv for the season (needed for player list for shells)
        logging.info("  Loading players_raw...")
        players_df_season = fetch_csv_from_url(f"{season}/players_raw.csv")
        if players_df_season is None or players_df_season.empty:
             logging.critical(f"  Critical Error: Failed to load players_raw for {season}. Cannot create GW shells.")
             all_data[season]['players'] = pd.DataFrame()
             all_data[season]['fixtures'] = pd.DataFrame()
             all_data[season]['gws'] = pd.DataFrame()
             continue
        all_data[season]['players'] = players_df_season

        # 2. Load fixtures.csv for the season (needed for fixture info for shells)
        logging.info("  Loading fixtures...")
        fixtures_df_season = fetch_csv_from_url(f"{season}/fixtures.csv")
        if fixtures_df_season is None or fixtures_df_season.empty:
             logging.warning(f"  Warning: Failed to load fixtures for {season}. GW shells might be incomplete.")
             all_data[season]['fixtures'] = pd.DataFrame()
        else:
             all_data[season]['fixtures'] = fixtures_df_season

        # 3. Load or Construct Gameweek Data
        logging.info("  Loading/Constructing gameweek data...")
        gws_list_for_season = []
        max_gw = 38 if season != current_season else current_season_gw_limit
        logging.info(f"    Targeting GW data up to GW {max_gw} for season {season}...")

        for gw_num in range(1, max_gw + 1):
            gw_data_path_suffix = f"{season}/gws/gw{gw_num}.csv"
            gw_df_actual = fetch_csv_from_url(gw_data_path_suffix)

            if gw_df_actual is not None and not gw_df_actual.empty:
                if 'GW' not in gw_df_actual.columns and 'gameweek' not in gw_df_actual.columns:
                    gw_df_actual['gameweek'] = gw_num
                elif 'GW' in gw_df_actual.columns and 'gameweek' not in gw_df_actual.columns:
                    gw_df_actual.rename(columns={'GW': 'gameweek'}, inplace=True)
                if 'gameweek' in gw_df_actual.columns:
                     gw_df_actual['gameweek'] = pd.to_numeric(gw_df_actual['gameweek'], errors='coerce').fillna(gw_num).astype(int)
                gws_list_for_season.append(gw_df_actual)
            else:
                logging.info(f"    Actual data for {gw_data_path_suffix} not found. Constructing shell for GW {gw_num}, Season {season}.")
                
                if players_df_season.empty or fixtures_df_season is None or fixtures_df_season.empty:
                    logging.warning(f"    Cannot construct shell for GW {gw_num}: missing players or fixtures base data for season {season}.")
                    continue

                shell_gw_fixtures = fixtures_df_season[fixtures_df_season['event'] == gw_num].copy()
                if shell_gw_fixtures.empty:
                    logging.warning(f"    No fixtures found for GW {gw_num} in season {season}. Cannot create shell rows for this GW.")
                    continue

                current_gw_shell_rows = []
                for _, player_row in players_df_season.iterrows():
                    player_id = player_row['id'] # Player's main FPL ID
                    player_team_id = player_row['team'] # Player's current team ID

                    player_fixture_details = shell_gw_fixtures[
                        (shell_gw_fixtures['team_h'] == player_team_id) |
                        (shell_gw_fixtures['team_a'] == player_team_id)
                    ]

                    if not player_fixture_details.empty:
                        fixture_info = player_fixture_details.iloc[0]
                        was_home = (fixture_info['team_h'] == player_team_id)
                        opponent_team_id = fixture_info['team_a'] if was_home else fixture_info['team_h']
                        
                        shell_row = {
                            'element': player_id,
                            'name': player_row.get('web_name', f"Player_{player_id}"),
                            'fixture': fixture_info['id'],
                            'opponent_team': opponent_team_id,
                            'was_home': was_home,
                            'gameweek': gw_num,
                            'season': season,
                            'value': player_row.get('now_cost', np.nan),
                            'cost': player_row.get('now_cost', np.nan) / 10.0 if pd.notna(player_row.get('now_cost')) else np.nan,
                            'total_points': np.nan, 'minutes': np.nan, 'goals_scored': np.nan,
                            'assists': np.nan, 'clean_sheets': np.nan, 'goals_conceded': np.nan,
                            'own_goals': np.nan, 'penalties_saved': np.nan, 'penalties_missed': np.nan,
                            'yellow_cards': np.nan, 'red_cards': np.nan, 'saves': np.nan,
                            'bonus': np.nan, 'bps': np.nan, 'influence': np.nan,
                            'creativity': np.nan, 'threat': np.nan, 'ict_index': np.nan,
                            'starts': np.nan, 'expected_goals': np.nan, 'expected_assists': np.nan,
                            'expected_goal_involvements': np.nan, 'expected_goals_conceded': np.nan,
                            'transfers_in': np.nan, 'transfers_out': np.nan, 'transfers_balance': np.nan,
                            'selected': np.nan,
                        }
                        current_gw_shell_rows.append(shell_row)
                
                if current_gw_shell_rows:
                    shell_df_for_gw = pd.DataFrame(current_gw_shell_rows)
                    gws_list_for_season.append(shell_df_for_gw)
                    logging.info(f"    Constructed shell for GW {gw_num}, Season {season} with {len(shell_df_for_gw)} players.")
                else:
                    logging.warning(f"    No player fixtures matched for GW {gw_num}, Season {season}. Shell not created.")
        if gws_list_for_season:
            logging.info(f"  Concatenating {len(gws_list_for_season)} GW DataFrames (actuals and shells) for {season}...")
            try:
                all_data[season]['gws'] = pd.concat(gws_list_for_season, ignore_index=True)
                logging.info(f"  Concatenated GW data shape for {season}: {all_data[season]['gws'].shape}")
            except Exception as e_concat:
                logging.error(f"  Error during GW concatenation for {season}: {e_concat}")
                all_data[season]['gws'] = pd.DataFrame()
        else:
             logging.error(f"  Error: No gameweek data (actual or shell) could be processed for {season}.")
             all_data[season]['gws'] = pd.DataFrame()

    logging.info("\n--- Raw Data Loading via HTTPS Finished ---")
    valid_data = {s: d for s, d in all_data.items() if d.get('gws') is not None and not d['gws'].empty}
    if not valid_data:
         logging.error("Error: Failed to load essential gameweek data for any specified season.")
         return None

    return valid_data
if __name__ == '__main__':
     logging.info("Running load_raw_data_via_https directly for testing...")
     test_seasons = ['2024-25']
     test_target_prediction_gw = 36

     if 'RAW_DATA_BASE_URL' not in os.environ:
          os.environ['RAW_DATA_BASE_URL'] = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
          logging.info(f"Set RAW_DATA_BASE_URL for test: {os.environ['RAW_DATA_BASE_URL']}")

     test_raw_data = load_raw_data_via_https(test_seasons, test_target_prediction_gw)

     if test_raw_data and test_raw_data.get(test_seasons[0], {}).get('gws') is not None:
          logging.info("\nTest load successful. Data dictionary structure:")
          for season, data_types in test_raw_data.items():
               logging.info(f"  Season {season}:")
               for dtype, df_content in data_types.items():
                    if df_content is not None:
                        logging.info(f"    {dtype}: {df_content.shape}")
                        if dtype == 'gws' and not df_content.empty:
                            target_gw_data = df_content[df_content['gameweek'] == test_target_prediction_gw]
                            if not target_gw_data.empty:
                                logging.info(f"    Data for target GW {test_target_prediction_gw} (shell or actual):")
                                logging.info(target_gw_data.head())
                            else:
                                logging.warning(f"    No data found for target GW {test_target_prediction_gw} in test output.")
                    else:
                        logging.info(f"    {dtype}: DataFrame is None")
     else:
          logging.error("\nTest load failed or returned no GWS data.")