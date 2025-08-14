import numpy as np
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, price_array, window_size=60, transaction_cost=0.001,
                 max_drawdown_penalty=0.1, volatility_penalty=0.05,
                 strategy_mode="long_only"):
        super().__init__()
        self.price_array = price_array
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.max_drawdown_penalty = max_drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.strategy_mode = strategy_mode

        self.n_assets = price_array.shape[1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size-1, self.n_assets), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.portfolio_value = 1.0
        self.prev_portfolio_value = 1.0
        self.max_portfolio_value = 1.0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns = []
        self.transactions = []
        self.weights_over_time = []

        return self._get_observation()

    def _get_observation(self):
        window = self.price_array[self.current_step - self.window_size:self.current_step]
        # Clean window to avoid divide-by-zero/NaN
        window = np.where(window <= 0, 1e-3, window)
        log_returns = np.log(window[1:] / (window[:-1] + 1e-8))
        return log_returns

    def step(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        if self.strategy_mode == "long_only":
            action = np.clip(action, 0, None)
        elif self.strategy_mode == "long_short":
            pass  # allow negative weights
        elif self.strategy_mode == "hedged":
            action = np.clip(action, -1, 1)
        else:
            raise ValueError("Invalid strategy_mode")

        action = action / (np.sum(np.abs(action)) + 1e-8)
        prev_weights = self.weights.copy()
        self.weights = action
        self.weights_over_time.append(self.weights.copy())

        prev_prices = self.price_array[self.current_step - 1]
        current_prices = self.price_array[self.current_step]

        prev_prices = np.where(prev_prices <= 0, 1e-3, prev_prices)
        current_prices = np.where(current_prices <= 0, 1e-3, current_prices)

        returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
        portfolio_return = np.dot(self.weights, returns)

        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)

        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        self.portfolio_returns.append(portfolio_return)

        turnover = np.sum(np.abs(prev_weights - self.weights))
        transaction_cost = turnover * self.transaction_cost
        self.portfolio_value *= (1 - transaction_cost)
        self.transactions.append(transaction_cost)

        volatility = np.std(self.portfolio_returns[-10:] if len(self.portfolio_returns) > 10 else self.portfolio_returns)
        drawdown = (self.max_portfolio_value - self.portfolio_value) / (self.max_portfolio_value + 1e-8)

        # reward = portfolio_return \
        #          - self.max_drawdown_penalty * max(0, drawdown) \
        #          - self.volatility_penalty * volatility \
        #          - 0.001 * np.sum(np.square(self.weights))

        reward = portfolio_return \
                    - 0.5 * self.max_drawdown_penalty * drawdown \
                    - 0.5 * self.volatility_penalty * volatility \
                    - 0.0005 * np.sum(np.square(self.weights))  # reduce L2 regularization


        self.current_step += 1
        if self.current_step >= len(self.price_array):
            self.done = True


        return self._get_observation(), reward, self.done, {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
            "transaction_cost": transaction_cost
        }

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}")

    def save_logs(self, path):
        import pandas as pd
        df = pd.DataFrame({
            "portfolio_value": [1.0] + [self.prev_portfolio_value] * (len(self.transactions) - 1) + [self.portfolio_value],
            "transaction_cost": self.transactions + [0.0],
        })
        df.to_csv(path, index=False)
