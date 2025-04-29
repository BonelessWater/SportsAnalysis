import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Connect to the database
print("Connecting to database...")
db_path = 'data/processed/nba_organized.db'
conn = sqlite3.connect(db_path)

# Let's inspect the game_id ranges in both tables
cursor = conn.cursor()
cursor.execute("SELECT MIN(game_id), MAX(game_id) FROM games")
games_range = cursor.fetchone()
print(f"Games table ID range: {games_range}")

cursor.execute("SELECT MIN(game_id), MAX(game_id) FROM box_scores")
box_scores_range = cursor.fetchone()
print(f"Box scores table ID range: {box_scores_range}")

# Since the IDs don't match, let's just work with the games table
print("\nBuilding model using games data only...")

# Load games data
query = """
SELECT 
    g.game_id,
    g.game_date,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    g.home_points,
    g.away_points,
    (g.home_points - g.away_points) AS point_diff
FROM 
    games g
JOIN 
    teams ht ON g.home_team_id = ht.team_id
JOIN 
    teams at ON g.away_team_id = at.team_id
WHERE 
    g.home_points IS NOT NULL AND g.away_points IS NOT NULL
"""

games_df = pd.read_sql_query(query, conn)
print(f"Loaded {len(games_df)} games with score information")

# Close connection
conn.close()

# Convert date columns and create temporal features
games_df['game_date'] = pd.to_datetime(games_df['game_date'], errors='coerce')
games_df = games_df.dropna(subset=['game_date'])

# Feature engineering
games_df['day_of_week'] = games_df['game_date'].dt.dayofweek
games_df['month'] = games_df['game_date'].dt.month
games_df['year'] = games_df['game_date'].dt.year

# Add team win rate features
print("Calculating team statistics...")

# Sort games by date
games_df = games_df.sort_values('game_date')

# Initialize team record tracking
team_records = {}
team_stats = {}

for team in pd.concat([games_df['home_team'], games_df['away_team']]).unique():
    team_records[team] = {'wins': 0, 'losses': 0}
    team_stats[team] = {
        'points_for': [], 
        'points_against': [],
        'last_5_results': []  # 1 for win, 0 for loss
    }

# Calculate rolling statistics
home_win_rates = []
away_win_rates = []
home_last_5 = []
away_last_5 = []
home_avg_points = []
away_avg_points = []
home_avg_points_allowed = []
away_avg_points_allowed = []

for _, game in games_df.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']
    
    # Get current records
    home_wins = team_records[home_team]['wins']
    home_losses = team_records[home_team]['losses']
    away_wins = team_records[away_team]['wins']
    away_losses = team_records[away_team]['losses']
    
    # Calculate win rates
    home_win_rate = home_wins / max(1, home_wins + home_losses)
    away_win_rate = away_wins / max(1, away_wins + away_losses)
    
    home_win_rates.append(home_win_rate)
    away_win_rates.append(away_win_rate)
    
    # Calculate last 5 win percentage
    home_last_5.append(sum(team_stats[home_team]['last_5_results'][-5:]) / max(1, len(team_stats[home_team]['last_5_results'][-5:])))
    away_last_5.append(sum(team_stats[away_team]['last_5_results'][-5:]) / max(1, len(team_stats[away_team]['last_5_results'][-5:])))
    
    # Calculate average points
    home_avg_points.append(np.mean(team_stats[home_team]['points_for'][-10:]) if team_stats[home_team]['points_for'] else 0)
    away_avg_points.append(np.mean(team_stats[away_team]['points_for'][-10:]) if team_stats[away_team]['points_for'] else 0)
    
    # Calculate average points allowed
    home_avg_points_allowed.append(np.mean(team_stats[home_team]['points_against'][-10:]) if team_stats[home_team]['points_against'] else 0)
    away_avg_points_allowed.append(np.mean(team_stats[away_team]['points_against'][-10:]) if team_stats[away_team]['points_against'] else 0)
    
    # Update records after the game
    home_points = game['home_points']
    away_points = game['away_points']
    
    if home_points > away_points:
        team_records[home_team]['wins'] += 1
        team_records[away_team]['losses'] += 1
        team_stats[home_team]['last_5_results'].append(1)
        team_stats[away_team]['last_5_results'].append(0)
    else:
        team_records[home_team]['losses'] += 1
        team_records[away_team]['wins'] += 1
        team_stats[home_team]['last_5_results'].append(0)
        team_stats[away_team]['last_5_results'].append(1)
    
    # Update points stats
    team_stats[home_team]['points_for'].append(home_points)
    team_stats[home_team]['points_against'].append(away_points)
    team_stats[away_team]['points_for'].append(away_points)
    team_stats[away_team]['points_against'].append(home_points)

# Add the features to the dataframe
games_df['home_win_rate'] = home_win_rates
games_df['away_win_rate'] = away_win_rates
games_df['home_last_5'] = home_last_5
games_df['away_last_5'] = away_last_5
games_df['home_avg_points'] = home_avg_points
games_df['away_avg_points'] = away_avg_points
games_df['home_avg_points_allowed'] = home_avg_points_allowed
games_df['away_avg_points_allowed'] = away_avg_points_allowed

# Create differential features
games_df['win_rate_diff'] = games_df['home_win_rate'] - games_df['away_win_rate']
games_df['last_5_diff'] = games_df['home_last_5'] - games_df['away_last_5']
games_df['avg_points_diff'] = games_df['home_avg_points'] - games_df['away_avg_points']
games_df['avg_points_allowed_diff'] = games_df['home_avg_points_allowed'] - games_df['away_avg_points_allowed']

# Now we'll add home court advantage (historical)
home_court_advantage = {}
for team in games_df['home_team'].unique():
    team_home_games = games_df[games_df['home_team'] == team]
    if len(team_home_games) >= 10:
        home_court_advantage[team] = team_home_games['point_diff'].mean()
    else:
        home_court_advantage[team] = 3.0  # Default home court advantage

games_df['home_court_advantage'] = games_df['home_team'].map(home_court_advantage)

# Define features to use
numeric_features = [
    'day_of_week', 'month', 'year',
    'home_win_rate', 'away_win_rate', 
    'home_last_5', 'away_last_5',
    'home_avg_points', 'away_avg_points',
    'home_avg_points_allowed', 'away_avg_points_allowed',
    'win_rate_diff', 'last_5_diff', 'avg_points_diff', 
    'avg_points_allowed_diff', 'home_court_advantage'
]

categorical_features = ['home_team', 'away_team']

print(f"Using {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")

# Filter to games with full feature set (after initial games)
games_df = games_df[games_df['home_avg_points'] > 0]
print(f"Final dataset: {len(games_df)} games")

# Prepare input data
X = games_df[numeric_features + categorical_features].copy()
y = games_df['point_diff']

# Fill any missing values
X = X.fillna(X.mean(numeric_only=True))
for cat in categorical_features:
    X[cat] = X[cat].fillna('Unknown')

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Create Ridge and Lasso pipelines
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge())
])

lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(max_iter=10000))
])

# Set up hyperparameter grid
ridge_param_grid = {
    'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

lasso_param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}

# Perform grid search for Ridge
print("Training Ridge model...")
ridge_grid = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

# Perform grid search for Lasso
print("Training Lasso model...")
lasso_grid = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

# Get best models
best_ridge = ridge_grid.best_estimator_
best_lasso = lasso_grid.best_estimator_

# Evaluate models
ridge_pred = best_ridge.predict(X_test)
lasso_pred = best_lasso.predict(X_test)

print(f"Ridge best alpha: {ridge_grid.best_params_['model__alpha']}")
print(f"Ridge RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred))}")
print(f"Ridge R²: {r2_score(y_test, ridge_pred)}")

print(f"Lasso best alpha: {lasso_grid.best_params_['model__alpha']}")
print(f"Lasso RMSE: {np.sqrt(mean_squared_error(y_test, lasso_pred))}")
print(f"Lasso R²: {r2_score(y_test, lasso_pred)}")

# Plot actual vs predicted
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.scatter(y_test, ridge_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Ridge Regression: Actual vs Predicted Point Differential')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(2, 1, 2)
plt.scatter(y_test, lasso_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Lasso Regression: Actual vs Predicted Point Differential')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.savefig('nba_prediction_results.png')
plt.close()

# Examine betting performance with various thresholds
print("\nBetting strategy evaluation:")
for threshold in [1, 2, 3, 4, 5]:
    # Apply threshold to identify betting opportunities
    bet_on = np.where(ridge_pred > threshold, 'Home', 
                      np.where(ridge_pred < -threshold, 'Away', 'No Bet'))
    
    # Calculate results
    bets_placed = bet_on != 'No Bet'
    total_bets = np.sum(bets_placed)
    
    if total_bets > 0:
        # For home team bets, a win is when point_diff > 0
        # For away team bets, a win is when point_diff < 0
        home_bets = np.sum((bet_on == 'Home') & bets_placed)
        away_bets = np.sum((bet_on == 'Away') & bets_placed)
        
        home_wins = np.sum((bet_on == 'Home') & (y_test > 0))
        away_wins = np.sum((bet_on == 'Away') & (y_test < 0))
        
        total_wins = home_wins + away_wins
        win_rate = total_wins / total_bets
        
        # Expected value using standard -110 odds (0.91 return)
        ev = (win_rate * 0.91) - (1 - win_rate)
        
        print(f"\nThreshold: {threshold} points")
        print(f"Total bets: {total_bets} ({total_bets/len(y_test):.1%} of games)")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Home/Away split: {home_bets}/{away_bets}")
        print(f"Expected value: {ev:.4f}")

# Save the best model
import joblib
joblib.dump(best_ridge, 'nba_ridge_model.pkl')
print("\nBest model saved as 'nba_ridge_model.pkl'")

# Function to use for predicting upcoming games
def predict_point_spread(model, home_team, away_team, home_win_rate, away_win_rate, 
                         home_last_5, away_last_5, home_avg_points, away_avg_points,
                         home_avg_points_allowed, away_avg_points_allowed):
    """
    Predict the point spread for an upcoming game
    
    Returns: predicted home team margin (positive = home team win)
    """
    # Prepare features
    features = pd.DataFrame({
        'day_of_week': [pd.Timestamp.now().dayofweek],
        'month': [pd.Timestamp.now().month],
        'year': [pd.Timestamp.now().year],
        'home_win_rate': [home_win_rate],
        'away_win_rate': [away_win_rate],
        'home_last_5': [home_last_5],
        'away_last_5': [away_last_5],
        'home_avg_points': [home_avg_points],
        'away_avg_points': [away_avg_points],
        'home_avg_points_allowed': [home_avg_points_allowed],
        'away_avg_points_allowed': [away_avg_points_allowed],
        'win_rate_diff': [home_win_rate - away_win_rate],
        'last_5_diff': [home_last_5 - away_last_5],
        'avg_points_diff': [home_avg_points - away_avg_points],
        'avg_points_allowed_diff': [home_avg_points_allowed - away_avg_points_allowed],
        'home_court_advantage': [3.0],  # Default or look up specific team value
        'home_team': [home_team],
        'away_team': [away_team]
    })
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return prediction

print("\nExample prediction function included - use 'predict_point_spread()' to predict new games")