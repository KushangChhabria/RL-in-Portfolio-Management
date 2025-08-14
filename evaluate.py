import numpy as np
import argparse
import matplotlib.pyplot as plt

from env.portfolio_env import PortfolioEnv
from utils.data_loader import load_data
from utils.baseline_strategies import equal_weight_strategy, buy_and_hold_strategy, random_strategy
from utils.metrics import calculate_sharpe, calculate_volatility, calculate_cumulative_return
from utils.plot_utils import plot_weights_heatmap, plot_transaction_costs
from utils.broker_kite_mock import place_order, get_ltp

from stable_baselines3 import PPO, TD3


def evaluate_strategy(values):
    values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0)
    values = np.where(values < 1e-8, 1e-8, values)
    returns = np.diff(values) / values[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    volatility = calculate_volatility(returns)
    sharpe = calculate_sharpe(returns)
    total_return = calculate_cumulative_return(values)
    return sharpe, volatility, total_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ppo_portfolio", help="Path to trained PPO model")
    parser.add_argument('--use_live', action='store_true', help="Use simulated Kite live data")
    parser.add_argument('--strategy_mode', type=str, default="long_only", help="Strategy mode: long_only or long_short")
    parser.add_argument('--drawdown_penalty', type=float, default=0.1, help="Penalty for drawdowns")
    parser.add_argument('--volatility_penalty', type=float, default=0.05, help="Penalty for high volatility")
    args = parser.parse_args()

    tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']

    if args.use_live:
        print("🟢 Using mock live prices from broker_kite_mock...")
        prices = []
        for _ in range(200):
            live_prices = get_ltp(tickers)
            prices.append([live_prices[t] for t in tickers])
        price_array = np.array(prices)
    else:
        print("📁 Using historical data from local cache...")
        data = load_data(tickers)
        price_array = np.stack([data[t]['Price'].values for t in tickers], axis=1)
        price_array = np.nan_to_num(price_array, nan=1.0, posinf=1.0, neginf=1.0)
        price_array = np.where(price_array < 1e-6, 1.0, price_array)

    env = PortfolioEnv(
        price_array=price_array,
        window_size=60,
        max_drawdown_penalty=args.drawdown_penalty,
        volatility_penalty=args.volatility_penalty,
        strategy_mode=args.strategy_mode
    )

    if "td3" in args.model_path.lower():
        model = TD3.load(args.model_path)
    else:
        model = PPO.load(args.model_path)

    obs = env.reset()
    done = False
    portfolio_values = []
    weights_over_time = []
    transaction_costs = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        weights_over_time.append(info['weights'])
        transaction_costs.append(info['transaction_cost'])

        if args.use_live:
            side = 'buy' if np.sum(action) > 0 else 'sell'
            quantity = round(np.sum(np.abs(action)) * 10)
            place_order('MOCK_PORTFOLIO', side, quantity, 'market')

    portfolio_values = np.array(portfolio_values)
    weights_over_time = np.array(weights_over_time)
    transaction_costs = np.array(transaction_costs)

    env.save_logs("logs/eval_transactions.csv")
    print("✅ Transaction log saved to logs/eval_transactions.csv")

    sharpe, vol, ret = evaluate_strategy(portfolio_values)
    print(f"\n📊 PPO Agent Performance:")
    print(f"Sharpe Ratio : {sharpe:.2f}")
    print(f"Volatility   : {vol:.2f}")
    print(f"Return       : {ret * 100:.2f}%")

    if not args.use_live:
        ew_values = equal_weight_strategy(price_array)
        bh_values = buy_and_hold_strategy(price_array)
        rand_values = random_strategy(price_array)

        print("\n📊 Baseline Strategies:")
        for name, val in zip(["Equal Weight", "Buy & Hold", "Random"], [ew_values, bh_values, rand_values]):
            s, v, r = evaluate_strategy(val)
            print(f"{name:<15} | Sharpe: {s:.2f} | Vol: {v:.2f} | Return: {r * 100:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label='PPO Agent')
        plt.plot(ew_values, label='Equal Weight')
        plt.plot(bh_values, label='Buy & Hold')
        plt.plot(rand_values, label='Random')
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("logs/performance_comparison.png")
        plt.show()

    plot_weights_heatmap(weights_over_time, tickers)
    plot_transaction_costs(transaction_costs)

    print("\n📌 Final Portfolio Weights Allocation:")
    for ticker, weight in zip(tickers, weights_over_time[-1]):
        print(f"{ticker:<12}: {weight:.4f}")


if __name__ == "__main__":
    main()
