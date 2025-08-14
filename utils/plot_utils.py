import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_portfolio(portfolio_values, title="Portfolio Value Over Time"):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weights_heatmap(weights_over_time, tickers):
    plt.figure(figsize=(12, 6))
    sns.heatmap(weights_over_time.T, cmap="viridis", cbar=True,
                xticklabels=False, yticklabels=tickers)
    plt.title("Asset Weights Over Time (PPO)")
    plt.xlabel("Time Step")
    plt.ylabel("Assets")
    plt.tight_layout()
    plt.show()

def plot_transaction_costs(costs):
    plt.figure(figsize=(10, 4))
    plt.plot(costs, label="Transaction Cost")
    plt.title("Transaction Costs per Step")
    plt.xlabel("Step")
    plt.ylabel("Cost (₹)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

