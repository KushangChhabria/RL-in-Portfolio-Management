import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to walk-forward result CSV
csv_path = "logs/walk_forward_results.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Run walk_forward_train_eval.py first.")

# Load data
df = pd.read_csv(csv_path)

# Extract fields
splits = df['Split']
sharpe = df['Sharpe']
volatility = df['Volatility']
returns = df['Return'] * 100  # convert to percentage

# Plot 1: Sharpe Ratio
plt.figure(figsize=(10, 4))
plt.bar(splits, sharpe, color='skyblue')
plt.title("🔹 Sharpe Ratio per Walk-Forward Split")
plt.ylabel("Sharpe Ratio")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Volatility
plt.figure(figsize=(10, 4))
plt.bar(splits, volatility, color='orange')
plt.title("🔸 Volatility per Walk-Forward Split")
plt.ylabel("Volatility")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Cumulative Return
plt.figure(figsize=(10, 4))
plt.bar(splits, returns, color='green')
plt.title("🟢 Cumulative Return per Walk-Forward Split")
plt.ylabel("Return (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Line Plot Summary
plt.figure(figsize=(10, 5))
plt.plot(splits, sharpe, marker='o', label='Sharpe')
plt.plot(splits, volatility, marker='x', label='Volatility')
plt.plot(splits, returns, marker='s', label='Return (%)')
plt.title("📈 Performance Metrics Over Walk-Forward Splits")
plt.ylabel("Metric Value")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
