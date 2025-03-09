class TradingSystem:
    def __init__(self, strategies):
        """
        :param strategies: List of tuples (proportion, strategy_instance)
        """
        self.strategies = strategies

    def backtest(self):
        results = []
        for proportion, strategy in self.strategies:
            print(f"Backtesting strategy with proportion: {proportion}")
            result = strategy.simulate()  # each strategy must implement simulate()
            results.append((proportion, result))
        return results
