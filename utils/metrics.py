import numpy as np

def calculate_sharpe(returns, risk_free_rate=0.0):
    """
    Calculate the annualized Sharpe ratio of daily returns.
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio * np.sqrt(252)

def calculate_volatility(returns):
    """
    Annualized volatility.
    """
    return np.std(returns) * np.sqrt(252)

def calculate_cumulative_return(portfolio_values):
    """
    Compute cumulative return from portfolio value array.
    """
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
