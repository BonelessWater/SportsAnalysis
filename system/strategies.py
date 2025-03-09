# strategies_xgboost.py
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# --- Monkey Patch Start ---
# Define a simple __sklearn_tags__ function that ignores any parent calls.
def custom_sklearn_tags(estimator):
    return {
        'binary_only': False,
        'multilabel': False,
        'multioutput': False,
        'requires_y': True,
        'poor_score': False,
    }

# Patch the XGBClassifier class by assigning our custom function.
XGBClassifier.__sklearn_tags__ = custom_sklearn_tags
# --- Monkey Patch End ---

class XGBoostStrategy:
    def __init__(self, risk_target, capital):
        self.risk_target = risk_target  # Fraction of capital to risk per trade
        self.capital = capital          # Total available capital
        self.model = None
        self._prepare_model()

    def _prepare_model(self):
        # For demonstration, we create synthetic yet structured data.
        np.random.seed(42)
        data_size = 1000
        
        # Example features:
        # - odds_diff: difference between our estimated odds and market odds.
        # - player_form: recent performance metric.
        # - head_to_head: historical head-to-head win rate difference.
        df = pd.DataFrame({
            'odds_diff': np.random.randn(data_size) * 0.5 + 0.1,
            'player_form': np.random.randn(data_size) * 0.5,
            'head_to_head': np.random.randn(data_size) * 0.3
        })
        # Define target variable: a trade is profitable if odds_diff > 0.2 and player_form > 0.
        df['profitable'] = ((df['odds_diff'] > 0.2) & (df['player_form'] > 0)).astype(int)

        # Split the data into training and testing sets.
        X = df[['odds_diff', 'player_form', 'head_to_head']]
        y = df['profitable']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize an XGBoost classifier.
        # (Setting use_label_encoder=False and specifying eval_metric avoids warnings.)
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        # Define a parameter grid for hyperparameter tuning.
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        # Use grid search with cross-validation to find the best model parameters.
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        
        # Optionally, calibrate the model so the predicted probabilities are more reliable.
        self.model = CalibratedClassifierCV(best_xgb, cv=5)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model on the test set.
        preds = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"XGBoostStrategy model trained with XGBoost. Test Accuracy: {accuracy:.2f}")

    def simulate(self):
        # For simulation, generate synthetic data that mimics realistic match scenarios.
        np.random.seed(101)
        simulation_size = 50
        sim_data = pd.DataFrame({
            'odds_diff': np.random.randn(simulation_size) * 0.5 + 0.1,
            'player_form': np.random.randn(simulation_size) * 0.5,
            'head_to_head': np.random.randn(simulation_size) * 0.3
        })

        # Get predicted probabilities for the "profitable" class.
        prob_predictions = self.model.predict_proba(sim_data)[:, 1]
        # Define a threshold probability above which a bet is placed.
        threshold = 0.6
        bets = prob_predictions > threshold

        profitable_trades = bets.sum()
        unprofitable_trades = simulation_size - profitable_trades

        # A simplistic profit calculation: adjust your stake size using a Kelly-like approach.
        profit_per_trade = self.risk_target * self.capital
        total_profit = (profitable_trades * profit_per_trade -
                        unprofitable_trades * (profit_per_trade * 0.5))

        result = {
            "total_trades": simulation_size,
            "profitable_trades": int(profitable_trades),
            "unprofitable_trades": int(unprofitable_trades),
            "total_profit": total_profit,
            "average_bet_probability": float(prob_predictions.mean())
        }
        return result

# Initialize the strategy with a risk target (e.g., 5% of capital per trade) and total capital.
strategy = XGBoostStrategy(risk_target=0.05, capital=1000)
simulation_result = strategy.simulate()
print("Simulation Result:")
print(simulation_result)
