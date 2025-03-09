import pandas as pd
import requests
import json
import sqlite3
from dotenv import load_dotenv
import os
from datetime import datetime, timezone

load_dotenv()

API_KEY = os.getenv("SPORTS_DEVS_API")

# -------------------------------------------------------------------
# Step 0: Configuration and API headers.
# -------------------------------------------------------------------

class TennisFetcher:
    def __init__(self, API_KEY):
        self.base_url = "https://tennis.sportdevs.com/"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }

    def get_matches(self):
        matches_url = self.base_url + "matches/"

        # We set limit to 10 so that we only fetch ten matches.
        params_matches = {
            'offset': 0,
            'limit': 10
        }

        print("Fetching matches...")
        response = requests.get(matches_url, headers=self.headers, params=params_matches)
        if response.status_code != 200:
            print(f"Error fetching matches list: {response.status_code} - {response.text}")
            exit(1)

        matches_data = response.json()
        if not matches_data:
            print("No matches data returned!")
            exit(1)

        print("Fetched Matches List:")
        print(json.dumps(matches_data, indent=2))

        # -------------------------------------------------------------------
        # Step 2: Create an SQLite database and a table for the match data.
        # -------------------------------------------------------------------
        # This example uses a database file named "matches.db".
        conn = sqlite3.connect("matches.db")
        cursor = conn.cursor()

        # Create a table named "matches". The table structure below is based on the sample JSON.
        # Adjust or remove columns as needed.
        create_table_query = """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            name TEXT,
            first_to_serve INTEGER,
            ground_type TEXT,
            tournament_id INTEGER,
            tournament_name TEXT,
            tournament_importance INTEGER,
            season_id INTEGER,
            season_name TEXT,
            round_id INTEGER,
            round_name TEXT,
            round_round INTEGER,
            round_end_time TEXT,
            round_start_time TEXT,
            status_type TEXT,
            status_reason TEXT,
            arena_id INTEGER,
            arena_name TEXT,
            arena_hash_image TEXT,
            home_team_id INTEGER,
            home_team_name TEXT,
            home_team_hash_image TEXT,
            away_team_id INTEGER,
            away_team_name TEXT,
            away_team_hash_image TEXT,
            home_team_score_current INTEGER,
            home_team_score_display INTEGER,
            home_team_score_period_1 INTEGER,
            home_team_score_period_2 INTEGER,
            home_team_score_default_time INTEGER,
            away_team_score_current INTEGER,
            away_team_score_display INTEGER,
            away_team_score_period_1 INTEGER,
            away_team_score_period_2 INTEGER,
            away_team_score_default_time INTEGER,
            times_period_1 INTEGER,
            times_period_2 INTEGER,
            times_specific_start_time TEXT,
            specific_start_time TEXT,
            start_time TEXT,
            duration INTEGER,
            class_id INTEGER,
            class_name TEXT,
            class_hash_image TEXT,
            league_id INTEGER,
            league_name TEXT,
            league_hash_image TEXT
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        # -------------------------------------------------------------------
        # Step 3: Insert each match into the SQLite table.
        # -------------------------------------------------------------------
        insert_query = """
        INSERT OR REPLACE INTO matches (
            id, name, first_to_serve, ground_type, tournament_id, tournament_name, tournament_importance,
            season_id, season_name, round_id, round_name, round_round, round_end_time, round_start_time,
            status_type, status_reason, arena_id, arena_name, arena_hash_image,
            home_team_id, home_team_name, home_team_hash_image, away_team_id, away_team_name, away_team_hash_image,
            home_team_score_current, home_team_score_display, home_team_score_period_1, home_team_score_period_2, home_team_score_default_time,
            away_team_score_current, away_team_score_display, away_team_score_period_1, away_team_score_period_2, away_team_score_default_time,
            times_period_1, times_period_2, times_specific_start_time, specific_start_time, start_time, duration,
            class_id, class_name, class_hash_image, league_id, league_name, league_hash_image
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        );
        """

        for match in matches_data:
            # Flatten nested objects
            # Round object
            round_obj = match.get("round", {})
            round_name = round_obj.get("name")
            round_round = round_obj.get("round")
            round_end_time = round_obj.get("end_time")
            round_start_time = round_obj.get("start_time")
            
            # Status object
            status_obj = match.get("status", {})
            status_reason = status_obj.get("reason")
            
            # Home team score object
            home_score = match.get("home_team_score", {})
            home_team_score_current = home_score.get("current")
            home_team_score_display = home_score.get("display")
            home_team_score_period_1 = home_score.get("period_1")
            home_team_score_period_2 = home_score.get("period_2")
            home_team_score_default_time = home_score.get("default_time")
            
            # Away team score object
            away_score = match.get("away_team_score", {})
            away_team_score_current = away_score.get("current")
            away_team_score_display = away_score.get("display")
            away_team_score_period_1 = away_score.get("period_1")
            away_team_score_period_2 = away_score.get("period_2")
            away_team_score_default_time = away_score.get("default_time")
            
            # Times object
            times_obj = match.get("times", {})
            times_period_1 = times_obj.get("period_1")
            times_period_2 = times_obj.get("period_2")
            times_specific_start_time = times_obj.get("specific_start_time")
            
            # Build the record tuple. Note that some fields are taken directly from the top-level JSON,
            # while others come from the nested objects.
            record = (
                match.get("id"),
                match.get("name"),
                match.get("first_to_serve"),
                match.get("ground_type"),
                match.get("tournament_id"),
                match.get("tournament_name"),
                match.get("tournament_importance"),
                match.get("season_id"),
                match.get("season_name"),
                match.get("round_id"),  # top-level round_id field (may be redundant)
                round_name,
                round_round,
                round_end_time,
                round_start_time,
                match.get("status_type"),  # top-level field
                status_reason,
                match.get("arena_id"),
                match.get("arena_name"),
                match.get("arena_hash_image"),
                match.get("home_team_id"),
                match.get("home_team_name"),
                match.get("home_team_hash_image"),
                match.get("away_team_id"),
                match.get("away_team_name"),
                match.get("away_team_hash_image"),
                home_team_score_current,
                home_team_score_display,
                home_team_score_period_1,
                home_team_score_period_2,
                home_team_score_default_time,
                away_team_score_current,
                away_team_score_display,
                away_team_score_period_1,
                away_team_score_period_2,
                away_team_score_default_time,
                times_period_1,
                times_period_2,
                times_specific_start_time,
                match.get("specific_start_time"),
                match.get("start_time"),
                match.get("duration"),
                match.get("class_id"),
                match.get("class_name"),
                match.get("class_hash_image"),
                match.get("league_id"),
                match.get("league_name"),
                match.get("league_hash_image")
            )
            
            cursor.execute(insert_query, record)

        conn.commit()
        conn.close()

        print("Data inserted into SQLite database 'matches.db' successfully.")
    
    def get_players(self):
        """
        Fetch player data for a given team_id from the players-by-team endpoint,
        extract key fields, compute the player's age, and insert the data into
        an SQLite table named 'player_stats'.
        
        The table will include the following columns:
        - player_id (primary key)
        - team_id
        - team_name
        - player_name
        - country_name
        - player_height
        - age (computed from date_of_birth)
        - win_rate (empty for now)
        - court_win_rate (empty for now)
        - weather_win_rate (empty for now)
        """
        url = "https://tennis.sportdevs.com/players-by-team"
        headers = {
            'Accept': 'application/json'
        }
        params = {
            'team_id': f'eq.{team_id}',
            'limit': 50,
            'offset': 0,
            'lang': 'en'
        }
        
        print(f"Fetching player data for team_id {team_id} ...")
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching players for team {team_id}: {response.status_code} - {response.text}")
            return
        
        teams_data = response.json()
        if not teams_data:
            print(f"No player data returned for team {team_id}")
            return
        
        cursor = db_conn.cursor()
        
        # Create the player_stats table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS player_stats (
            player_id INTEGER PRIMARY KEY,
            team_id INTEGER,
            team_name TEXT,
            player_name TEXT,
            country_name TEXT,
            player_height INTEGER,
            age INTEGER,
            win_rate REAL,
            court_win_rate REAL,
            weather_win_rate REAL
        );
        """
        cursor.execute(create_table_query)
        db_conn.commit()
        
        # Each team record contains team information and an array of player objects.
        for team_record in teams_data:
            current_team_id = team_record.get("team_id")
            team_name = team_record.get("team_name")
            players = team_record.get("players", [])
            
            for player in players:
                player_id = player.get("id")
                player_name = player.get("name")
                country_name = player.get("country_name")
                player_height = player.get("player_height")
                dob_str = player.get("date_of_birth")
                age = self.calculate_age(dob_str) if dob_str else None
                
                # Additional columns to be filled later are set to None for now.
                win_rate = None
                court_win_rate = None
                weather_win_rate = None
                
                insert_query = """
                INSERT OR REPLACE INTO player_stats (
                    player_id, team_id, team_name, player_name, country_name, player_height, age,
                    win_rate, court_win_rate, weather_win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    player_id,
                    current_team_id,
                    team_name,
                    player_name,
                    country_name,
                    player_height,
                    age,
                    win_rate,
                    court_win_rate,
                    weather_win_rate
                ))
                db_conn.commit()
                print(f"Inserted/Updated player '{player_name}' (ID: {player_id}) from team '{team_name}'.")
    
    def calculate_age(dob_str):
    """
    Given an ISO 8601 date string (e.g., "1998-08-31T00:00:00+00:00"),
    return the age in years.
    """
    try:
        # Replace "Z" if present and parse the ISO formatted date.
        dob = datetime.fromisoformat(dob_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age = now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day))
        return age
    except Exception as e:
        print(f"Error calculating age from '{dob_str}': {e}")
        return None