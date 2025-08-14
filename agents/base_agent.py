from env.portfolio_env import PortfolioEnv
from utils.save_utils import save_metadata, get_timestamp
import os
import json
from stable_baselines3.common.vec_env import DummyVecEnv

class BaseAgent:
    def __init__(self, model_class, price_array, window_size=60,
                 log_dir=None, max_drawdown_penalty=0.1, volatility_penalty=0.05,
                 strategy_mode="long_only", use_optuna=True, optuna_params_path="optuna_best_params.json"):

        self.env = DummyVecEnv([lambda: PortfolioEnv(
            price_array=price_array,
            window_size=window_size,
            max_drawdown_penalty=max_drawdown_penalty,
            volatility_penalty=volatility_penalty,
            strategy_mode=strategy_mode
        )])

        self.model_class = model_class
        self.price_array = price_array
        self.window_size = window_size
        self.log_dir = log_dir

        # ✅ Common model config
        model_kwargs = {
            "verbose": 1,
            "tensorboard_log": log_dir
        }

        # ✅ PPO-specific config
        if model_class.__name__ == "PPO":
            model_kwargs["ent_coef"] = 0.01
            model_kwargs["normalize_advantage"] = True

            if use_optuna:
                try:
                    with open(optuna_params_path, "r") as f:
                        best_params = json.load(f)
                    model_kwargs.update(best_params)
                    print(f"✅ Loaded Optuna best hyperparameters from {optuna_params_path}")
                except FileNotFoundError:
                    print(f"⚠️ {optuna_params_path} not found. Using default PPO parameters.")

        # Instantiate model
        self.model = model_class("MlpPolicy", self.env, **model_kwargs)

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        metadata = {
            "window_size": self.window_size,
            "price_array_shape": list(self.price_array.shape),
            "log_dir": self.log_dir,
            "timestamp": get_timestamp()
        }
        save_metadata(path + "_meta.json", metadata)

    def load(self, path):
        self.model = self.model_class.load(path)
        return self.model
