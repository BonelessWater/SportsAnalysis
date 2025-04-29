import sqlite3
import os

def migrate_database():
    # Connect to databases
    print("Connecting to databases...")
    old_conn = sqlite3.connect('nba.db')
    
    # Create a fresh new database
    if os.path.exists('nba_organized.db'):
        os.remove('nba_organized.db')
    
    new_conn = sqlite3.connect('nba_organized.db')
    
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Disable foreign key constraints during creation
    new_cursor.execute("PRAGMA foreign_keys = OFF")
    
    # Create the new tables - all at once before inserting data
    print("Creating new database structure...")
    
    # Teams table
    new_cursor.execute("""
        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_code TEXT,
            UNIQUE(team_name)
        )
    """)
    
    # Players table
    new_cursor.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            position TEXT,
            height TEXT,
            weight REAL,
            birth_date TEXT,
            birth_country TEXT,
            college TEXT,
            url TEXT,
            UNIQUE(player_name, birth_date)
        )
    """)
    
    # Games table
    new_cursor.execute("""
        CREATE TABLE games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            season TEXT,
            start_time TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_points INTEGER,
            away_points INTEGER,
            overtime TEXT,
            attendance INTEGER,
            box_score_link TEXT NOT NULL,
            notes TEXT,
            UNIQUE(box_score_link)
        )
    """)
    
    # Player seasons table
    new_cursor.execute("""
        CREATE TABLE player_seasons (
            player_season_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            team_id INTEGER,
            season TEXT,
            experience TEXT,
            games INTEGER,
            games_started INTEGER,
            minutes_per_game REAL,
            field_goals REAL,
            field_goal_attempts REAL,
            field_goal_percentage REAL,
            three_pointers REAL,
            three_point_attempts REAL,
            three_point_percentage REAL,
            two_pointers REAL,
            two_point_attempts REAL,
            two_point_percentage REAL,
            effective_fg_percentage REAL,
            free_throws REAL,
            free_throw_attempts REAL,
            free_throw_percentage REAL,
            offensive_rebounds REAL,
            defensive_rebounds REAL,
            total_rebounds REAL,
            assists REAL,
            steals REAL,
            blocks REAL,
            turnovers REAL,
            personal_fouls REAL,
            points_per_game REAL
        )
    """)
    
    # Box scores table
    new_cursor.execute("""
        CREATE TABLE box_scores (
            box_score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            player_id INTEGER,
            team_id INTEGER,
            is_starter INTEGER,
            minutes_played TEXT,
            field_goals INTEGER,
            field_goal_attempts INTEGER,
            field_goal_pct REAL,
            three_pointers INTEGER,
            three_point_attempts INTEGER,
            three_point_pct REAL,
            free_throws INTEGER,
            free_throw_attempts INTEGER,
            free_throw_pct REAL,
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            total_rebounds INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            personal_fouls INTEGER,
            points INTEGER,
            plus_minus INTEGER
        )
    """)
    
    # Step 1: Migrate teams
    print("Migrating teams...")
    
    # Extract unique teams from various sources
    teams_query = """
    SELECT DISTINCT team_name FROM (
        SELECT DISTINCT team AS team_name FROM players
        UNION
        SELECT DISTINCT home_team AS team_name FROM gamelogs
        UNION
        SELECT DISTINCT visitor_team AS team_name FROM gamelogs
    ) WHERE team_name IS NOT NULL
    """
    
    try:
        old_cursor.execute(teams_query)
        teams = old_cursor.fetchall()
    except:
        # Alternative approach if the query doesn't work
        teams = []
        
        try:
            old_cursor.execute("SELECT DISTINCT team FROM players")
            teams.extend([(t[0],) for t in old_cursor.fetchall() if t[0]])
        except:
            print("Could not get teams from players table")
        
        try:
            old_cursor.execute("SELECT DISTINCT home_team FROM gamelogs")
            teams.extend([(t[0],) for t in old_cursor.fetchall() if t[0]])
        except:
            print("Could not get teams from gamelogs (home_team)")
        
        try:
            old_cursor.execute("SELECT DISTINCT visitor_team FROM gamelogs")
            teams.extend([(t[0],) for t in old_cursor.fetchall() if t[0]])
        except:
            print("Could not get teams from gamelogs (visitor_team)")
    
    # Also try to get teams from box scores tables
    old_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'box_scores_per_game%'")
    box_tables = old_cursor.fetchall()
    
    for table in box_tables:
        try:
            old_cursor.execute(f"SELECT DISTINCT team FROM {table[0]}")
            teams.extend([(t[0],) for t in old_cursor.fetchall() if t[0]])
        except:
            print(f"Could not get teams from {table[0]}")
    
    # Remove duplicates
    unique_teams = set()
    for team in teams:
        if team[0]:
            unique_teams.add(team[0])
    
    print(f"Found {len(unique_teams)} unique teams")
    
    # Insert teams
    for team in unique_teams:
        # Try to extract team code from box score links
        team_code = None
        try:
            old_cursor.execute("SELECT box_score_link FROM gamelogs WHERE home_team = ? LIMIT 1", (team,))
            result = old_cursor.fetchone()
            if result and result[0]:
                link = result[0]
                if link.endswith('.html'):
                    team_code = link[-8:-5]
        except:
            pass
        
        try:
            new_cursor.execute("INSERT INTO teams (team_name, team_code) VALUES (?, ?)", (team, team_code))
        except sqlite3.IntegrityError:
            pass  # Skip duplicates
    
    new_conn.commit()
    
    # Step 2: Migrate players
    print("Migrating players...")
    
    # Get players from players table
    try:
        old_cursor.execute("""
            SELECT DISTINCT name, position, height, weight, birth_date, 
            birth_country, college, url FROM players
        """)
        players_data = old_cursor.fetchall()
        
        for player in players_data:
            if player[0]:
                try:
                    new_cursor.execute("""
                        INSERT OR IGNORE INTO players 
                        (player_name, position, height, weight, birth_date, birth_country, college, url) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, player)
                except:
                    pass
        
        print(f"Migrated {len(players_data)} players from players table")
    except:
        print("Could not migrate from players table")
    
    # Get additional players from box scores
    players_added = 0
    for table in box_tables:
        try:
            old_cursor.execute(f"SELECT DISTINCT player_name FROM {table[0]}")
            box_players = old_cursor.fetchall()
            
            for player in box_players:
                if player[0]:
                    try:
                        new_cursor.execute("""
                            INSERT OR IGNORE INTO players (player_name) VALUES (?)
                        """, (player[0],))
                        if new_cursor.rowcount > 0:
                            players_added += 1
                    except:
                        pass
        except:
            pass
    
    print(f"Added {players_added} additional players from box scores")
    new_conn.commit()
    
    # Step 3: Migrate games
    print("Migrating games...")
    
    try:
        old_cursor.execute("""
            SELECT season_year, game_date, start_et, visitor_team, visitor_pts, 
            home_team, home_pts, box_score_link, overtime, attendance, notes
            FROM gamelogs
        """)
        
        games_data = old_cursor.fetchall()
        games_inserted = 0
        
        for game in games_data:
            try:
                if game[3] and game[5] and game[7]:  # Make sure critical fields exist
                    # Format the season (e.g., "2023-24")
                    season = None
                    if game[0]:  # season_year
                        season = f"{game[0]}-{str(game[0] + 1)[-2:]}"
                    
                    # Get team IDs
                    home_team_id = None
                    away_team_id = None
                    
                    new_cursor.execute("SELECT team_id FROM teams WHERE team_name = ?", (game[5],))
                    result = new_cursor.fetchone()
                    if result:
                        home_team_id = result[0]
                    
                    new_cursor.execute("SELECT team_id FROM teams WHERE team_name = ?", (game[3],))
                    result = new_cursor.fetchone()
                    if result:
                        away_team_id = result[0]
                    
                    # Parse attendance
                    attendance = None
                    if game[9]:
                        try:
                            attendance = int(game[9].replace(',', ''))
                        except:
                            pass
                    
                    new_cursor.execute("""
                        INSERT OR IGNORE INTO games 
                        (game_date, season, start_time, home_team_id, away_team_id, 
                        home_points, away_points, box_score_link, overtime, attendance, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (game[1], season, game[2], home_team_id, away_team_id, 
                         game[6], game[4], game[7], game[8], attendance, game[10]))
                    
                    if new_cursor.rowcount > 0:
                        games_inserted += 1
            except:
                pass
        
        print(f"Inserted {games_inserted} games")
        new_conn.commit()
    except:
        print("Could not migrate games")
    
    # Step 4: Migrate player seasons
    print("Migrating player seasons...")
    
    try:
        old_cursor.execute("""
            SELECT p.name, p.team, p.season, p.experience, p.games, p.games_started, 
            p.minutes_per_game, p.field_goals, p.field_goal_attempts, p.field_goal_percentage,
            p.three_pointers, p.three_point_attempts, p.three_point_percentage,
            p.two_pointers, p.two_point_attempts, p.two_point_percentage,
            p.effective_fg_percentage, p.free_throws, p.free_throw_attempts,
            p.free_throw_percentage, p.offensive_rebounds, p.defensive_rebounds,
            p.total_rebounds, p.assists, p.steals, p.blocks, p.turnovers,
            p.personal_fouls, p.points_per_game
            FROM players p
        """)
        
        player_seasons_data = old_cursor.fetchall()
        seasons_inserted = 0
        
        for ps in player_seasons_data:
            try:
                if ps[0] and ps[1]:  # Make sure name and team exist
                    # Get player ID
                    player_id = None
                    new_cursor.execute("SELECT player_id FROM players WHERE player_name = ?", (ps[0],))
                    result = new_cursor.fetchone()
                    if result:
                        player_id = result[0]
                    else:
                        continue
                    
                    # Get team ID
                    team_id = None
                    new_cursor.execute("SELECT team_id FROM teams WHERE team_name = ?", (ps[1],))
                    result = new_cursor.fetchone()
                    if result:
                        team_id = result[0]
                    else:
                        continue
                    
                    new_cursor.execute("""
                        INSERT INTO player_seasons 
                        (player_id, team_id, season, experience, games, games_started, 
                        minutes_per_game, field_goals, field_goal_attempts, field_goal_percentage,
                        three_pointers, three_point_attempts, three_point_percentage,
                        two_pointers, two_point_attempts, two_point_percentage,
                        effective_fg_percentage, free_throws, free_throw_attempts,
                        free_throw_percentage, offensive_rebounds, defensive_rebounds,
                        total_rebounds, assists, steals, blocks, turnovers,
                        personal_fouls, points_per_game)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (player_id, team_id, ps[2], ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], 
                          ps[9], ps[10], ps[11], ps[12], ps[13], ps[14], ps[15], ps[16], ps[17], 
                          ps[18], ps[19], ps[20], ps[21], ps[22], ps[23], ps[24], ps[25], ps[26], 
                          ps[27], ps[28]))
                    
                    seasons_inserted += 1
            except:
                pass
        
        print(f"Inserted {seasons_inserted} player seasons")
        new_conn.commit()
    except:
        print("Could not migrate player seasons")
    
    # Step 5: Migrate box scores (the largest part)
    print("Migrating box scores (this may take a while)...")
    
    total_box_scores = 0
    
    for table in box_tables:
        table_name = table[0]
        print(f"Processing {table_name}...")
        
        try:
            # Get count of records
            old_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = old_cursor.fetchone()[0]
            print(f"  Found {count} records")
            
            # Process in batches to avoid memory issues
            batch_size = 5000
            for offset in range(0, count, batch_size):
                old_cursor.execute(f"""
                    SELECT box_score_link, team, player_name, starter, mp, 
                    fg, fga, fg_pct, threep, threepa, threep_pct, 
                    ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, 
                    tov, pf, pts, plus_minus
                    FROM {table_name}
                    LIMIT {batch_size} OFFSET {offset}
                """)
                
                batch_data = old_cursor.fetchall()
                batch_inserted = 0
                
                for bs in batch_data:
                    try:
                        if bs[0] and bs[1] and bs[2]:  # Make sure critical fields exist
                            # Get game ID from box_score_link
                            game_id = None
                            new_cursor.execute("SELECT game_id FROM games WHERE box_score_link = ?", (bs[0],))
                            result = new_cursor.fetchone()
                            if result:
                                game_id = result[0]
                            else:
                                # Game not found, skip this record
                                continue
                            
                            # Get player ID
                            player_id = None
                            new_cursor.execute("SELECT player_id FROM players WHERE player_name = ?", (bs[2],))
                            result = new_cursor.fetchone()
                            if result:
                                player_id = result[0]
                            else:
                                # Create player if not found
                                new_cursor.execute("INSERT INTO players (player_name) VALUES (?)", (bs[2],))
                                player_id = new_cursor.lastrowid
                            
                            # Get team ID
                            team_id = None
                            new_cursor.execute("SELECT team_id FROM teams WHERE team_name = ?", (bs[1],))
                            result = new_cursor.fetchone()
                            if result:
                                team_id = result[0]
                            else:
                                # Create team if not found
                                new_cursor.execute("INSERT INTO teams (team_name) VALUES (?)", (bs[1],))
                                team_id = new_cursor.lastrowid
                            
                            new_cursor.execute("""
                                INSERT INTO box_scores 
                                (game_id, player_id, team_id, is_starter, minutes_played, 
                                field_goals, field_goal_attempts, field_goal_pct, 
                                three_pointers, three_point_attempts, three_point_pct, 
                                free_throws, free_throw_attempts, free_throw_pct, 
                                offensive_rebounds, defensive_rebounds, total_rebounds, 
                                assists, steals, blocks, turnovers, personal_fouls, points, plus_minus)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (game_id, player_id, team_id, bs[3], bs[4], bs[5], bs[6], bs[7], 
                                 bs[8], bs[9], bs[10], bs[11], bs[12], bs[13], bs[14], bs[15], 
                                 bs[16], bs[17], bs[18], bs[19], bs[20], bs[21], bs[22], bs[23]))
                            
                            batch_inserted += 1
                    except:
                        pass
                
                total_box_scores += batch_inserted
                print(f"  Inserted {batch_inserted} records (batch {offset//batch_size + 1})")
                new_conn.commit()
        except Exception as e:
            print(f"  Error processing {table_name}: {str(e)}")
    
    print(f"Total box scores inserted: {total_box_scores}")
    
    # Now add foreign key constraints
    print("Adding foreign key constraints...")
    
    # Create a new database with constraints
    constraint_conn = sqlite3.connect('nba_final.db')
    new_cursor = new_conn.cursor()
    constraint_cursor = constraint_conn.cursor()
    
    # Enable foreign keys
    constraint_cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create tables with constraints
    constraint_cursor.execute("""
        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_code TEXT,
            UNIQUE(team_name)
        )
    """)
    
    constraint_cursor.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            position TEXT,
            height TEXT,
            weight REAL,
            birth_date TEXT,
            birth_country TEXT,
            college TEXT,
            url TEXT,
            UNIQUE(player_name, birth_date)
        )
    """)
    
    constraint_cursor.execute("""
        CREATE TABLE games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            season TEXT,
            start_time TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_points INTEGER,
            away_points INTEGER,
            overtime TEXT,
            attendance INTEGER,
            box_score_link TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams(team_id),
            UNIQUE(box_score_link)
        )
    """)
    
    constraint_cursor.execute("""
        CREATE TABLE player_seasons (
            player_season_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            team_id INTEGER,
            season TEXT,
            experience TEXT,
            games INTEGER,
            games_started INTEGER,
            minutes_per_game REAL,
            field_goals REAL,
            field_goal_attempts REAL,
            field_goal_percentage REAL,
            three_pointers REAL,
            three_point_attempts REAL,
            three_point_percentage REAL,
            two_pointers REAL,
            two_point_attempts REAL,
            two_point_percentage REAL,
            effective_fg_percentage REAL,
            free_throws REAL,
            free_throw_attempts REAL,
            free_throw_percentage REAL,
            offensive_rebounds REAL,
            defensive_rebounds REAL,
            total_rebounds REAL,
            assists REAL,
            steals REAL,
            blocks REAL,
            turnovers REAL,
            personal_fouls REAL,
            points_per_game REAL,
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
    """)
    
    constraint_cursor.execute("""
        CREATE TABLE box_scores (
            box_score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            player_id INTEGER,
            team_id INTEGER,
            is_starter INTEGER,
            minutes_played TEXT,
            field_goals INTEGER,
            field_goal_attempts INTEGER,
            field_goal_pct REAL,
            three_pointers INTEGER,
            three_point_attempts INTEGER,
            three_point_pct REAL,
            free_throws INTEGER,
            free_throw_attempts INTEGER,
            free_throw_pct REAL,
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            total_rebounds INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            personal_fouls INTEGER,
            points INTEGER,
            plus_minus INTEGER,
            FOREIGN KEY (game_id) REFERENCES games(game_id),
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
    """)
    
    # Copy all data to the new database
    print("Copying teams...")
    new_cursor.execute("SELECT * FROM teams")
    for row in new_cursor.fetchall():
        constraint_cursor.execute("INSERT INTO teams VALUES (?, ?, ?)", row)
    
    print("Copying players...")
    new_cursor.execute("SELECT * FROM players")
    for row in new_cursor.fetchall():
        constraint_cursor.execute("INSERT INTO players VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    
    print("Copying games...")
    new_cursor.execute("SELECT * FROM games")
    for row in new_cursor.fetchall():
        constraint_cursor.execute("""
            INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
    
    print("Copying player_seasons...")
    new_cursor.execute("SELECT * FROM player_seasons")
    for row in new_cursor.fetchall():
        constraint_cursor.execute("""
            INSERT INTO player_seasons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
    
    print("Copying box_scores (this may take a while)...")
    new_cursor.execute("SELECT COUNT(*) FROM box_scores")
    count = new_cursor.fetchone()[0]
    
    batch_size = 10000
    for offset in range(0, count, batch_size):
        new_cursor.execute(f"SELECT * FROM box_scores LIMIT {batch_size} OFFSET {offset}")
        for row in new_cursor.fetchall():
            constraint_cursor.execute("""
                INSERT INTO box_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
        constraint_conn.commit()
        print(f"  Copied {min(offset + batch_size, count)}/{count} box scores")
    
    # Create useful indexes
    print("Creating indexes...")
    constraint_cursor.execute("CREATE INDEX idx_games_date ON games(game_date)")
    constraint_cursor.execute("CREATE INDEX idx_games_season ON games(season)")
    constraint_cursor.execute("CREATE INDEX idx_box_scores_game ON box_scores(game_id)")
    constraint_cursor.execute("CREATE INDEX idx_box_scores_player ON box_scores(player_id)")
    constraint_cursor.execute("CREATE INDEX idx_player_seasons_player ON player_seasons(player_id)")
    constraint_cursor.execute("CREATE INDEX idx_player_seasons_season ON player_seasons(season)")
    
    constraint_conn.commit()
    
    # Print final counts
    print("\nFinal table counts:")
    constraint_cursor.execute("SELECT COUNT(*) FROM teams")
    print(f"Teams: {constraint_cursor.fetchone()[0]}")
    
    constraint_cursor.execute("SELECT COUNT(*) FROM players")
    print(f"Players: {constraint_cursor.fetchone()[0]}")
    
    constraint_cursor.execute("SELECT COUNT(*) FROM games")
    print(f"Games: {constraint_cursor.fetchone()[0]}")
    
    constraint_cursor.execute("SELECT COUNT(*) FROM player_seasons")
    print(f"Player seasons: {constraint_cursor.fetchone()[0]}")
    
    constraint_cursor.execute("SELECT COUNT(*) FROM box_scores")
    print(f"Box scores: {constraint_cursor.fetchone()[0]}")
    
    # Close connections
    old_conn.close()
    new_conn.close()
    constraint_conn.close()
    
    # Rename the final database
    os.remove('nba_organized.db')
    os.rename('nba_final.db', 'nba_organized.db')
    
    print("\nMigration complete! New database saved as 'nba_organized.db'")

if __name__ == "__main__":
    migrate_database()