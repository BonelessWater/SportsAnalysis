import sqlite3

class Database:
    def __init__(self, db_name="tennis_data.db"):
        self.db_name = db_name

    def initialize(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Create players table (for tennis players)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT,
            country TEXT
        )
        ''')

        # Create matches table (for tennis matches)
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

        # Create odds table (for tennis betting odds)
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

        conn.commit()
        conn.close()
        print("Database initialized with tables for tennis data.")
