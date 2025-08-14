# train/train_ppo.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import json

from utils.data_loader import load_data
from agents.ppo_agent import PPOAgent
from utils.save_utils import save_metadata, get_timestamp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100_000, help="Number of training timesteps")
    parser.add_argument('--drawdown_penalty', type=float, default=0.1, help="Penalty for drawdowns")
    parser.add_argument('--volatility_penalty', type=float, default=0.05, help="Penalty for volatility")
    parser.add_argument('--strategy_mode', type=str, default="long_only", help="Strategy mode: long_only or long_short")
    args = parser.parse_args()

    tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
    data = load_data(tickers)

    raw_price_array = np.stack([data[t]['Price'].values for t in tickers], axis=1)
    price_array = np.log(raw_price_array[1:] / (raw_price_array[:-1] + 1e-8))  # ✅ Use log returns

    if (price_array.std(axis=0) < 1e-6).any():
        raise ValueError("⚠️ Input data has near-zero variance. Check source prices.")

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Initialize PPOAgent
    agent = PPOAgent(
        price_array=price_array,
        window_size=60,
        log_dir=log_dir,
        max_drawdown_penalty=args.drawdown_penalty,
        volatility_penalty=args.volatility_penalty,
        strategy_mode=args.strategy_mode
    )

    try:
        agent.train(timesteps=args.timesteps)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    model_path = "ppo_portfolio"
    agent.save(model_path)

    save_metadata(f"{model_path}_meta.json", {
        "window_size": agent.window_size,
        "price_array_shape": price_array.shape,
        "log_dir": log_dir,
        "timestamp": get_timestamp()
    })

    print(f"✅ PPO model saved to '{model_path}.zip'")


if __name__ == "__main__":
    main()
