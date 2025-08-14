import numpy as np

def equal_weight_strategy(price_array):
    weights = np.ones(price_array.shape[1]) / price_array.shape[1]
    price_array = np.where(price_array < 1e-6, 1e-6, price_array)
    price_array = np.nan_to_num(price_array, nan=1.0, posinf=1.0, neginf=1.0)

    portfolio_values = (price_array / price_array[0]) @ weights
    return portfolio_values

def buy_and_hold_strategy(price_array):
    initial_price = price_array[0]
    price_array = np.where(price_array < 1e-6, 1e-6, price_array)
    price_array = np.nan_to_num(price_array, nan=1.0, posinf=1.0, neginf=1.0)

    portfolio_values = price_array / initial_price
    portfolio_values = np.mean(portfolio_values, axis=1)
    return portfolio_values

def random_strategy(price_array, seed=42):
    np.random.seed(seed)
    n_assets = price_array.shape[1]
    portfolio_value = [1.0]

    for t in range(1, len(price_array)):
        weights = np.random.dirichlet(np.ones(n_assets))
        daily_return = (price_array[t] / price_array[t-1]) @ weights
        portfolio_value.append(portfolio_value[-1] * daily_return)

    return np.array(portfolio_value)
