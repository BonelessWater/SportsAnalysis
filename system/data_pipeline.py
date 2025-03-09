import sqlite3
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_db():
    # This function creates the necessary tables in our tennis database.
    conn = sqlite3.connect('tennis_data.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        full_name TEXT,
        country TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY,
        match_date TEXT,
        player1_id INTEGER,
        player2_id INTEGER,
        score TEXT,
        winner_id INTEGER,
        FOREIGN KEY (player1_id) REFERENCES players (player_id),
        FOREIGN KEY (player2_id) REFERENCES players (player_id),
        FOREIGN KEY (winner_id) REFERENCES players (player_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS odds (
        odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sport_key TEXT,
        event_id TEXT,
        event_name TEXT,
        bookmaker TEXT,
        market TEXT,
        odds_player1 REAL,
        odds_player2 REAL,
        region TEXT,
        timestamp TEXT
    )
    ''')

    print("Database and tables initialized successfully.")
    conn.commit()
    conn.close()

def fetch_data(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

def fetch_odds():
    API_KEY = os.getenv("ODDS_API_KEY")
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    # For tennis, use the appropriate sport key.
    # (Check the API docs; here we assume "tennis_atp" for ATP matches.)
    SPORT = "tennis_atp"
    REGIONS = "us,uk,eu"
    MARKETS = "h2h"  # head-to-head market for match winners
    ODDS_FORMAT = "decimal"
    DATE_FORMAT = "iso"

    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        "api_key": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }

    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")

    try:
        data = response.json()
    except Exception as e:
        print("Error parsing JSON:", e)
        return None

    if response.status_code == 200:
        return data
    else:
        print(f"Error fetching odds: {response.status_code} - {response.text}")
        return None

def insert_tennis_odds(odds_data):
    """
    Inserts tennis odds into the database. Assumes each event corresponds to a match with two players.
    """
    conn = sqlite3.connect('tennis_data.db')
    cursor = conn.cursor()

    for event in odds_data:
        sport_key = event.get("sport_key")
        event_id = event.get("id", None)  # if provided by the API
        commence_time = event.get("commence_time")

        # Construct an event name. Many tennis events list competitors in a "teams" field.
        teams = event.get("teams", [])
        if len(teams) == 2:
            event_name = " vs ".join(teams)
        else:
            event_name = "Unknown vs Unknown"

        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title")
            for market in bookmaker.get("markets", []):
                market_type = market.get("key")
                outcomes = market.get("outcomes", [])
                odds_player1 = None
                odds_player2 = None
                if len(outcomes) == 2:
                    odds_player1 = outcomes[0].get("price")
                    odds_player2 = outcomes[1].get("price")

                # Insert the odds record into the database
                cursor.execute('''
                    INSERT INTO odds (
                        sport_key, event_id, event_name, bookmaker, market, odds_player1, odds_player2, region, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sport_key, event_id, event_name, bookmaker_name, market_type,
                    odds_player1, odds_player2, bookmaker.get("region", "unknown"), commence_time
                ))

    conn.commit()
    print("Tennis odds data inserted successfully.")
    conn.close()

if __name__ == "__main__":
    # Optionally, initialize the DB (if not already done via database.py)
    initialize_db()

    odds_data = fetch_odds()
    if odds_data is None or len(odds_data) == 0:
        print("No data fetched from the API. Check the API response or parameters.")
    else:
        print("Fetched odds data:")
        print(odds_data)
        insert_tennis_odds(odds_data)
