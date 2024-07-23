import numpy as np
from typing import Tuple, Dict

class Backtest:
    """
    A class to backtest a trading strategy.

    Attributes:
    signal (np.ndarray): A numpy array of trading signals (1 for long, -1 for short).
    start_val (float): The starting value of the portfolio.
    risk_per_trade (float): The maximum percentage of the portfolio to risk on a trade.
    target_returns (np.ndarray): A numpy array of the returns of the portfolio.
    portfolio_value (float): The current value of the portfolio.
    strategy_returns (np.ndarray): The returns for the strategy.
    cumulative_profits (np.ndarray): The cumulative profits.
    total_portfolio_value (np.ndarray): The total portfolio value at each step.
    """

    def __init__(self, signal: np.ndarray, start_val: float, risk_per_trade: float, target_returns: np.ndarray):
        if len(signal) != len(target_returns):
            raise ValueError("Signal and target_returns must have the same length")

        self.signal = signal
        self.portfolio_value = start_val
        self.risk_per_trade = risk_per_trade
        self.target_returns = target_returns
        self.strategy_returns = None
        self.cumulative_profits = None
        self.total_portfolio_value = None

    def calculate_strategy_returns(self) -> None:
        """Calculate the returns for the strategy."""
        self.strategy_returns = self.signal * self.target_returns * self.risk_per_trade

    def calculate_profits(self) -> None:
        """Calculate the profit per trade and update the portfolio value."""
        if self.strategy_returns is None:
            self.calculate_strategy_returns()

        self.cumulative_profits = np.zeros_like(self.strategy_returns)
        current_portfolio_value = self.portfolio_value

        for i, strategy_return in enumerate(self.strategy_returns):
            profit = current_portfolio_value * strategy_return
            self.cumulative_profits[i] = profit
            current_portfolio_value += profit

        # Initialize total_portfolio_value with the starting value
        self.total_portfolio_value = np.zeros(len(self.cumulative_profits) + 1)
        self.total_portfolio_value[0] = self.portfolio_value
        self.total_portfolio_value[1:] = self.portfolio_value + np.cumsum(self.cumulative_profits)

    def calculate_win_ratio(self) -> float:
        """Calculate and return the win ratio."""
        wins = (self.signal == np.sign(self.target_returns))
        return np.mean(wins)
    
    def calculate_buy_and_hold_returns(self) -> float:
        """Calculate and return the buy and hold returns as a percentage."""
        return (np.prod(1 + self.target_returns) - 1)

    def calculate_strategy_total_return(self) -> float:
        """Calculate and return the total return for the strategy."""
        return self.total_portfolio_value[-1] / self.portfolio_value - 1

    def get_performance_statistics(self) -> dict:
        """Return a dictionary of performance statistics."""
        return {
            "Win Ratio": self.calculate_win_ratio(),
            "Buy and Hold Returns": self.calculate_buy_and_hold_returns(),
            "Strategy Total Return": self.calculate_strategy_total_return()
        }

    def run_backtest(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the complete backtest.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The cumulative profits and total portfolio value.
        """
        self.calculate_strategy_returns()
        self.calculate_profits()
        return self.cumulative_profits, self.total_portfolio_value
