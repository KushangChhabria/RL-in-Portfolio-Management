import pandas as pd
import matplotlib.pyplot as plt
import os

# Load walk-forward results
RESULTS_PATH = "logs/walk_forward_results.csv"
assert os.path.exists(RESULTS_PATH), f"Result file not found at {RESULTS_PATH}"

df = pd.read_csv(RESULTS_PATH)

# Plot Sharpe Ratio over splits
plt.figure(figsize=(10, 4))
plt.plot(df['Split'], df['Sharpe'], marker='o', label='Sharpe Ratio')
plt.xticks(rotation=45)
plt.title('Walk-Forward Sharpe Ratio per Split')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/sharpe_over_time.png")
plt.show()

# Plot Volatility over splits
plt.figure(figsize=(10, 4))
plt.plot(df['Split'], df['Volatility'], marker='s', color='orange', label='Volatility')
plt.xticks(rotation=45)
plt.title('Walk-Forward Volatility per Split')
plt.ylabel('Volatility')
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/volatility_over_time.png")
plt.show()

# Plot Return over splits
plt.figure(figsize=(10, 4))
plt.plot(df['Split'], df['Return'] * 100, marker='^', color='green', label='Return (%)')
plt.xticks(rotation=45)
plt.title('Walk-Forward Return per Split')
plt.ylabel('Return (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/return_over_time.png")
plt.show()

print("\n✅ Saved plots to 'logs/' folder: sharpe_over_time.png, volatility_over_time.png, return_over_time.png")
