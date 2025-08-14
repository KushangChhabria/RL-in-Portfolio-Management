# train/train_td3.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import json
from utils.data_loader import load_data, get_price_array, train_test_split
from agents.td3_agent import TD3Agent
from utils.save_utils import save_metadata, get_timestamp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--strategy_mode', type=str, default="long_only", choices=["long_only", "long_short", "hedged"])
    parser.add_argument('--drawdown_penalty', type=float, default=0.1)
    parser.add_argument('--volatility_penalty', type=float, default=0.05)
    args = parser.parse_args()

    tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
    data = load_data(tickers)
    price_array = get_price_array(data)

    # ✅ Ensure consistent train/test split
    train_array, _ = train_test_split(price_array, split_date='2023-07-01', data=data, tickers=tickers)

    if (train_array.std(axis=0) < 1e-6).any():
        raise ValueError("Training price data has near-zero variance for some assets.")

    # ✅ Create logs directory
    log_dir = "./logs_td3"
    os.makedirs(log_dir, exist_ok=True)

    agent = TD3Agent(
        price_array=train_array,
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

    model_path = "td3_portfolio"
    agent.save(model_path)

    save_metadata(f"{model_path}_meta.json", {
        "window_size": agent.window_size,
        "price_array_shape": train_array.shape,
        "log_dir": log_dir,
        "strategy_mode": args.strategy_mode,
        "drawdown_penalty": args.drawdown_penalty,
        "volatility_penalty": args.volatility_penalty,
        "timestamp": get_timestamp()
    })

    print(f"✅ TD3 model saved to '{model_path}.zip'")


if __name__ == "__main__":
    main()
