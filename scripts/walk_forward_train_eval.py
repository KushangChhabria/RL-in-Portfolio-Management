import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from utils.data_loader import load_data
from agents.ppo_agent import PPOAgent
from env.portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
from utils.metrics import calculate_sharpe, calculate_volatility, calculate_cumulative_return

# Parameters
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
WINDOW_SIZE = 30
TIMESTEPS = 50_000
START_YEAR = 2015
END_YEAR = 2024
TRAIN_YEARS = 3
TEST_YEARS = 1

# Load full data
data = load_data(TICKERS)
price_df = pd.concat([data[t]['Price'].rename(t) for t in TICKERS], axis=1)
price_df = price_df.dropna()
price_df.index = pd.to_datetime(price_df.index)

print("📅 Date range:", price_df.index.min().date(), "→", price_df.index.max().date())
print("📆 Years available:", sorted(price_df.index.year.unique()))

# Extract year-wise boundaries
def walk_forward_splits(price_df, train_years, test_years):
    years = sorted(price_df.index.year.unique())
    splits = []
    for i in range(len(years) - train_years - test_years + 1):
        train_start = years[i]
        train_end = years[i + train_years - 1]
        test_start = years[i + train_years]
        test_end = years[i + train_years + test_years - 1]

        train_mask = (price_df.index.year >= train_start) & (price_df.index.year <= train_end)
        test_mask = (price_df.index.year >= test_start) & (price_df.index.year <= test_end)

        train_data = price_df[train_mask].values
        test_data = price_df[test_mask].values

        # Debug shape check
        print(f"🧪 Split {train_start}-{train_end} → {test_start}-{test_end} | Train: {train_data.shape} | Test: {test_data.shape}")

        # ✅ Only include if both are large enough
        if len(train_data) > WINDOW_SIZE and len(test_data) > WINDOW_SIZE:
            splits.append({
                'train_range': f'{train_start}-{train_end}',
                'test_range': f'{test_start}-{test_end}',
                'train_data': train_data,
                'test_data': test_data
            })
        else:
            print(f"⚠️ Skipping split due to insufficient data (need > {WINDOW_SIZE} rows)")

    return splits

splits = walk_forward_splits(price_df, TRAIN_YEARS, TEST_YEARS)

if len(splits) == 0:
    print("❌ No valid splits available. Please check data range or parameters.")
    exit()

# Walk-forward training and evaluation
results = []
for idx, split in enumerate(splits):
    print(f"\n🔁 Split {idx + 1}: Train {split['train_range']} → Test {split['test_range']}")

    try:
        # Train PPO agent
        agent = PPOAgent(price_array=split['train_data'], window_size=WINDOW_SIZE)
        agent.train(timesteps=TIMESTEPS)

        # Evaluate on test data
        env = PortfolioEnv(price_array=split['test_data'], window_size=WINDOW_SIZE)
        obs = env.reset()
        done = False
        portfolio_values = []

        model = agent.model

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            portfolio_values.append(info['portfolio_value'])

        if len(portfolio_values) < 2:
            print("⚠️ Not enough values to compute returns. Skipping.")
            continue

        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        sharpe = calculate_sharpe(returns)
        vol = calculate_volatility(returns)
        ret = calculate_cumulative_return(values)

        print(f"📊 Sharpe: {sharpe:.2f} | Volatility: {vol:.2f} | Return: {ret*100:.2f}%")

        results.append({
            'Split': f"{split['train_range']} → {split['test_range']}",
            'Sharpe': sharpe,
            'Volatility': vol,
            'Return': ret
        })

    except Exception as e:
        print(f"❌ Error during Split {idx+1}: {e}")

# Save summary CSV
if results:
    df_results = pd.DataFrame(results)
    os.makedirs("logs", exist_ok=True)
    df_results.to_csv("logs/walk_forward_results.csv", index=False)
    print("\n✅ Walk-forward results saved to logs/walk_forward_results.csv")
else:
    print("❌ No results to save. Check previous warnings or errors.")
