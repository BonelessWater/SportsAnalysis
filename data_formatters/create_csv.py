import sqlite3
import csv
import logging
from datetime import datetime
import os
import sys
import json
import pandas as pd
import numpy as np

# -----------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,  # maximum verbosity for debugging
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("nba_data_processing.log")
    ]
)

# -----------------------------------------------------------
# Team Mapping: Full team names from gamelogs to three-letter tickers from players table
# -----------------------------------------------------------
TEAM_NAME_TO_TICKER = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "New Jersey Nets": "NJN",  # For 2004-2011
    "Charlotte Bobcats": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Hornets": "NOH",
    "New York Knicks": "NYK",
    "Seattle SuperSonics": "SEA",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

def get_ticker(team_full_name, mapping):
    """
    Convert a full team name from gamelogs to its ticker using the provided mapping.
    """
    ticker = mapping.get(team_full_name)
    if not ticker:
        logging.warning("No ticker found for team: %s", team_full_name)
    return ticker

# -----------------------------------------------------------
# Utility functions for date, height, and age conversion
# -----------------------------------------------------------
def convert_date_to_int(date_str):
    """
    Convert a date string formatted as "Weekday, Month Day, Year" or "Month Day, Year"
    to an integer (YYYYMMDD) and return the datetime object.
    """
    try:
        logging.debug("Original date string: %s", date_str)
        # Remove weekday prefix (e.g., "Mon, ") if present.
        if len(date_str) > 4 and date_str[3] == ',':
            date_str = date_str[4:].strip()
        try:
            dt = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            dt = datetime.strptime(date_str, "%b %d, %Y")
        date_int = int(dt.strftime("%Y%m%d"))
        return date_int, dt
    except Exception as e:
        logging.error("Error converting date '%s': %s", date_str, e)
        raise

def convert_height_to_cm(height_str):
    """
    Convert a height string in the format 'feet-inches' (e.g., '6-1') to centimeters.
    """
    try:
        feet, inches = height_str.split('-')
        feet = float(feet)
        inches = float(inches)
        cm = feet * 30.48 + inches * 2.54
        return round(cm, 2)
    except Exception as e:
        logging.error("Error converting height '%s': %s", height_str, e)
        raise

def compute_age(birth_date_str, reference_date):
    """
    Compute a player's age at the time of a game.
    birth_date_str is expected in the format "Month Day, Year" (e.g., "September 19, 1998").
    """
    try:
        try:
            birth_date = datetime.strptime(birth_date_str, "%B %d, %Y")
        except ValueError:
            birth_date = datetime.strptime(birth_date_str, "%b %d, %Y")
        age = reference_date.year - birth_date.year - (
            (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day)
        )
        return age
    except Exception as e:
        logging.error("Error computing age for birth_date '%s': %s", birth_date_str, e)
        raise

# -----------------------------------------------------------
# Data fetching functions
# -----------------------------------------------------------
def fetch_all_players(db_path):
    """
    Fetch all players from the players table and return a dictionary mapping team ticker
    to a list of player dictionaries with selected and converted features.
    """
    players_dict = {}
    logging.info("Fetching all players from the players table.")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """
        SELECT name, team, position, height, weight, birth_date, experience
        FROM players;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        logging.info("Fetched %d players.", len(rows))
    except Exception as e:
        logging.error("Error fetching players: %s", e)
        raise

    for row in rows:
        name, team, position, height, weight, birth_date, experience = row
        try:
            height_cm = convert_height_to_cm(height)
        except Exception as e:
            logging.error("Error converting height for player %s: %s", name, e)
            height_cm = None
        try:
            if experience.strip().upper() == "R":
                exp_years = 0
            else:
                exp_years = float(experience)
        except Exception as e:
            logging.error("Error processing experience for player %s: %s", name, e)
            exp_years = None

        player_data = {
            "name": name,
            "team": team,  # This should be a ticker as stored in the players table.
            "position": position,
            "height_cm": height_cm,
            "weight": weight,
            "birth_date": birth_date,
            "experience": exp_years
        }
        if team not in players_dict:
            players_dict[team] = []
        players_dict[team].append(player_data)
    conn.close()
    logging.info("Finished processing players.")
    return players_dict

def fetch_gamelogs(db_path):
    """
    Connect to the SQLite database and fetch data from the gamelogs table.
    Excludes columns: id, month, box_score_link, and notes.
    """
    logging.info("Fetching gamelog data.")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """
        SELECT season_year, game_date, start_et, visitor_team, visitor_pts,
               home_team, home_pts, overtime, attendance
        FROM gamelogs;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        logging.info("Fetched %d gamelog rows.", len(rows))
    except Exception as e:
        logging.error("Error fetching gamelogs: %s", e)
        raise
    conn.close()
    return rows

# -----------------------------------------------------------
# Data processing function: Combine game logs with player data
# -----------------------------------------------------------
def process_data(gamelog_rows, players_dict):
    """
    Process gamelog rows by:
      - Converting game_date to an integer and datetime.
      - Computing the target (home_pts - visitor_pts).
      - Mapping full team names to tickers.
      - Adding player JSON data for visitor and home teams.
    """
    processed_rows = []
    for idx, row in enumerate(gamelog_rows):
        try:
            season_year, game_date, start_et, visitor_team, visitor_pts, home_team, home_pts, overtime, attendance = row
            game_date_int, game_date_dt = convert_date_to_int(game_date)
            target = home_pts - visitor_pts

            # Map full team names to tickers.
            visitor_ticker = get_ticker(visitor_team, TEAM_NAME_TO_TICKER)
            home_ticker = get_ticker(home_team, TEAM_NAME_TO_TICKER)
            if not visitor_ticker or not home_ticker:
                logging.warning("Skipping row %d due to missing team mapping.", idx)
                continue

            # Process visitor players.
            visitor_players_raw = players_dict.get(visitor_ticker, [])
            visitor_players = []
            for player in visitor_players_raw:
                try:
                    age = compute_age(player["birth_date"], game_date_dt)
                except Exception as e:
                    logging.error("Error computing age for visitor player %s: %s", player["name"], e)
                    age = None
                visitor_players.append({
                    "name": player["name"],
                    "team": player["team"],
                    "position": player["position"],
                    "age": age,
                    "height_cm": player["height_cm"],
                    "weight": player["weight"],
                    "experience": player["experience"]
                })

            # Process home players.
            home_players_raw = players_dict.get(home_ticker, [])
            home_players = []
            for player in home_players_raw:
                try:
                    age = compute_age(player["birth_date"], game_date_dt)
                except Exception as e:
                    logging.error("Error computing age for home player %s: %s", player["name"], e)
                    age = None
                home_players.append({
                    "name": player["name"],
                    "team": player["team"],
                    "position": player["position"],
                    "age": age,
                    "height_cm": player["height_cm"],
                    "weight": player["weight"],
                    "experience": player["experience"]
                })

            processed_record = {
                "season_year": season_year,
                "game_date": game_date_int,
                "start_et": start_et,
                "visitor_team": visitor_team,
                "visitor_pts": visitor_pts,
                "home_team": home_team,
                "home_pts": home_pts,
                "overtime": overtime,
                "attendance": attendance,
                "target": target,
                "visitor_players": json.dumps(visitor_players),
                "home_players": json.dumps(home_players)
            }
            processed_rows.append(processed_record)
        except Exception as e:
            logging.error("Error processing gamelog row %d: %s", idx, e)
    logging.info("Processed %d gamelog rows.", len(processed_rows))
    return processed_rows

# -----------------------------------------------------------
# Extract aggregated numeric features from JSON columns
# -----------------------------------------------------------
def extract_player_features(json_str, prefix):
    """
    Given a JSON string of player data, compute aggregate features (average, min, max)
    for age, height, and experience. The prefix distinguishes home vs. visitor.
    """
    try:
        players = json.loads(json_str)
    except Exception as e:
        logging.error("Error parsing JSON for %s: %s", prefix, e)
        return {
            f"{prefix}_avg_age": np.nan,
            f"{prefix}_min_age": np.nan,
            f"{prefix}_max_age": np.nan,
            f"{prefix}_avg_height": np.nan,
            f"{prefix}_min_height": np.nan,
            f"{prefix}_max_height": np.nan,
            f"{prefix}_avg_exp": np.nan,
            f"{prefix}_min_exp": np.nan,
            f"{prefix}_max_exp": np.nan
        }
    ages = [player.get("age") for player in players if player.get("age") is not None]
    heights = [player.get("height_cm") for player in players if player.get("height_cm") is not None]
    exps = [player.get("experience") for player in players if player.get("experience") is not None]

    def safe_agg(arr, func):
        return func(arr) if arr else np.nan

    features = {
        f"{prefix}_avg_age": safe_agg(ages, np.mean),
        f"{prefix}_min_age": safe_agg(ages, np.min),
        f"{prefix}_max_age": safe_agg(ages, np.max),
        f"{prefix}_avg_height": safe_agg(heights, np.mean),
        f"{prefix}_min_height": safe_agg(heights, np.min),
        f"{prefix}_max_height": safe_agg(heights, np.max),
        f"{prefix}_avg_exp": safe_agg(exps, np.mean),
        f"{prefix}_min_exp": safe_agg(exps, np.min),
        f"{prefix}_max_exp": safe_agg(exps, np.max)
    }
    return features

# -----------------------------------------------------------
# Main processing flow: fetch, process, extract features, and write CSV
# -----------------------------------------------------------
def main():
    # Path to SQLite database
    db_path = os.path.join("Gambling-Project-", "data collector", "nba_players.db")
    logging.info("Starting NBA data processing.")
    
    # Fetch player and gamelog data
    players_dict = fetch_all_players(db_path)
    gamelog_rows = fetch_gamelogs(db_path)
    
    # Process gamelog rows by integrating player data
    processed_data = process_data(gamelog_rows, players_dict)
    
    # Convert processed data into a DataFrame
    df = pd.DataFrame(processed_data)
    logging.info("Converted processed data to DataFrame with shape: %s", df.shape)
    
    # Extract aggregate features from the JSON columns
    home_features_series = df["home_players"].apply(lambda x: extract_player_features(x, "home"))
    home_features_df = pd.DataFrame(home_features_series.tolist())
    
    visitor_features_series = df["visitor_players"].apply(lambda x: extract_player_features(x, "visitor"))
    visitor_features_df = pd.DataFrame(visitor_features_series.tolist())
    
    # Concatenate the new aggregated features with the original DataFrame
    final_df = pd.concat([df, home_features_df, visitor_features_df], axis=1)
    logging.info("Final DataFrame shape after adding aggregated features: %s", final_df.shape)
    
    # Define output directory and file path
    output_dir = os.path.join("..", "outputs", "data", "nba")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "nba_data.csv")
    
    # Write the final DataFrame to CSV
    final_df.to_csv(output_file, index=False)
    logging.info("Final CSV saved to %s", output_file)

if __name__ == "__main__":
    main()
