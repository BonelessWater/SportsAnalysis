from database import Database
from strategies import MLStrategy
from trading_system import TradingSystem

def main():
    # Initialize the database (creates tables if needed)
    db = Database()
    db.initialize()

    # Define risk and capital
    risk_target = 0.30
    capital = 100

    # Initialize trading system with our ML strategy.
    # The tuple (1.0, strategy_instance) allows for scaling strategies later.
    trading_system = TradingSystem(
        strategies=[
            (1.0, MLStrategy(risk_target=risk_target, capital=capital)),
        ]
    )

    # Run backtest simulation
    print("Running backtest...")
    results = trading_system.backtest()

    # Display results
    for proportion, result in results:
        print(f"\nStrategy Proportion: {proportion}")
        for key, value in result.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
