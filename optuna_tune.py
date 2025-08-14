import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.portfolio_env import PortfolioEnv
from utils.data_loader import load_data
from utils.metrics import calculate_sharpe

# ---- CONSTANTS ----
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
WINDOW_SIZE = 30
TIMESTEPS = 20_000  # Low for tuning speed

# ---- LOAD & PREPROCESS DATA ----
data = load_data(TICKERS)
raw_price_array = np.stack([data[t]['Price'].values for t in TICKERS], axis=1)

# Normalize as returns (stationary input)
price_array = np.diff(raw_price_array, axis=0) / (raw_price_array[:-1] + 1e-8)


# ---- OBJECTIVE FUNCTION ----
def optimize(trial):
    # Hyperparameters to search
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)

    # Create env
    env = DummyVecEnv([lambda: PortfolioEnv(
        price_array=price_array,
        window_size=WINDOW_SIZE,
        max_drawdown_penalty=0.1,
        volatility_penalty=0.05
    )])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0
    )

    # Train
    model.learn(total_timesteps=TIMESTEPS)

    # Evaluate
    obs = env.reset()
    done = False
    portfolio_values = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info[0]['portfolio_value'])

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = calculate_sharpe(returns)

    trial.set_user_attr("sharpe", sharpe)
    return sharpe


# ---- RUN OPTIMIZATION ----
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize, n_trials=20)

    print("\n🎯 Best Trial:")
    print(study.best_trial)

    print("\n🧪 All Trials:")
    for trial in study.trials:
        print(f"Trial#{trial.number}: Sharpe={trial.user_attrs.get('sharpe', 0):.3f}, Params={trial.params}")

import json
with open("optuna_best_params.json", "w") as f:
    json.dump(study.best_trial.params, f, indent=4)




# model_kwargs = {
#     "verbose": 1,
#     "tensorboard_log": log_dir,
#     "normalize_advantage": True,
#     "ent_coef": 0.001,        # Lower entropy helps stable convergence
#     "learning_rate": 3e-4,
#     "gamma": 0.99,
#     "gae_lambda": 0.95,
#     "n_steps": 512,
#     "batch_size": 64,
# }