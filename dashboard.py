import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from agents.td3_agent import TD3Agent
from utils.data_loader import load_data, get_price_array
from utils.metrics import calculate_sharpe, calculate_volatility, calculate_cumulative_return
from utils.baseline_strategies import equal_weight_strategy, buy_and_hold_strategy, random_strategy
from utils.plot_utils import plot_weights_heatmap

st.set_page_config(page_title="RL Portfolio Planner", layout="centered")
st.title("📈 RL-based Portfolio Strategy Planner")

# ---------------------- Sidebar Inputs ----------------------
st.sidebar.header("User Inputs")
agent_choice = st.sidebar.selectbox("Choose RL Agent", ["PPO", "TD3"])
use_demo = st.sidebar.checkbox("Use Demo Account (₹1,00,000)", value=True)

if not use_demo:
    investment = st.sidebar.number_input("Enter investment amount (₹)", min_value=10000, step=10000, value=100000)
else:
    investment = 100000

duration = st.sidebar.selectbox("Select Duration (months)", [3, 6, 12, 18, 24])
profit_goal = st.sidebar.slider("Expected Profit Goal (%)", min_value=5, max_value=200, step=5, value=50)
risk_level = st.sidebar.radio("Risk Level", ["Low", "Medium", "High"], index=1)

risk_penalty_map = {"Low": 0.1, "Medium": 0.05, "High": 0.01}
drawdown_penalty = risk_penalty_map[risk_level]
volatility_penalty = risk_penalty_map[risk_level]

# ---------------------- Load Data ----------------------
tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
data = load_data(tickers)
price_array = get_price_array(data)

# ---------------------- Load Agent ----------------------
st.subheader(f"🎯 Strategy Planning with {agent_choice} Agent")

AgentClass = PPOAgent if agent_choice == "PPO" else TD3Agent
model_path = "ppo_portfolio" if agent_choice == "PPO" else "td3_portfolio"

agent = AgentClass(
    price_array=price_array,
    window_size=60,
    log_dir=None,
    max_drawdown_penalty=drawdown_penalty,
    volatility_penalty=volatility_penalty
)
model = agent.load(model_path)

# ---------------------- Simulate Agent ----------------------
from env.portfolio_env import PortfolioEnv

env = PortfolioEnv(
    price_array=price_array,
    window_size=60,
    max_drawdown_penalty=drawdown_penalty,
    volatility_penalty=volatility_penalty,
    strategy_mode="long_only"
)
obs = env.reset()
portfolio_values, weights_over_time = [], []
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(info["portfolio_value"])
    weights_over_time.append(info["weights"])

final_value = portfolio_values[-1]
returns = np.diff(portfolio_values) / portfolio_values[:-1]
volatility = calculate_volatility(returns)
sharpe = calculate_sharpe(returns)
return_pct = calculate_cumulative_return(portfolio_values) * 100
achievable_profit = (final_value - 1.0) * 100

# ---------------------- Display Strategy Outcome ----------------------
st.success("✅ Strategy Computed")
st.markdown(f"**Achievable Profit:** {achievable_profit:.2f}%")
st.markdown(f"**Target Profit:** {profit_goal:.2f}%")
st.markdown(f"**Sharpe Ratio:** {sharpe:.2f}")
st.markdown(f"**Volatility:** {volatility:.2f}")
st.markdown(f"**Final Portfolio Value:** ₹{investment * final_value:,.2f}")

# ---------------------- Portfolio Weights ----------------------
st.subheader("📊 Final Portfolio Weights")
weights_df = pd.DataFrame([weights_over_time[-1]], columns=tickers)
st.dataframe(weights_df.T.rename(columns={0: "Weight"}).style.format("{:.2%}"))

# ---------------------- Value Plot ----------------------
st.subheader("📈 Portfolio Value Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(portfolio_values, label="RL Agent")
ax.set_xlabel("Steps")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

# ---------------------- Baseline Comparison ----------------------
st.subheader("📉 Baseline Strategy Comparison")
baselines = {
    "Equal Weight": equal_weight_strategy(price_array),
    "Buy & Hold": buy_and_hold_strategy(price_array),
    "Random": random_strategy(price_array),
}
for name, val in baselines.items():
    r = np.diff(val) / val[:-1]
    s = calculate_sharpe(r)
    v = calculate_volatility(r)
    ret = calculate_cumulative_return(val) * 100
    st.markdown(f"**{name}** | Sharpe: {s:.2f}, Volatility: {v:.2f}, Return: {ret:.2f}%")

# ---------------------- Portfolio Heatmap ----------------------
st.subheader("🧯 Weight Allocation Heatmap")
plot_weights_heatmap(np.array(weights_over_time), tickers)
st.pyplot(plt.gcf())






























# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# from stable_baselines3 import PPO, TD3, SAC
# from env.portfolio_env import PortfolioEnv
# from utils.data_loader import load_data
# from utils.metrics import calculate_sharpe, calculate_volatility, calculate_cumulative_return
# from utils.baseline_strategies import equal_weight_strategy, buy_and_hold_strategy, random_strategy
# from utils.plot_utils import plot_weights_heatmap, plot_transaction_costs
# from utils.broker_kite_mock import get_ltp, get_holdings, get_positions, place_order

# # --- Streamlit Setup ---
# st.set_page_config(layout="wide")
# st.title("💹 RL Portfolio Management Dashboard")

# # --- Sidebar Planner ---
# st.sidebar.header("📌 Strategy Planner")
# desired_profit = st.sidebar.slider("Target Profit (%)", 1, 100, 20)
# duration = st.sidebar.slider("Investment Duration (days)", 10, 200, 60)
# risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])
# demo_mode = st.sidebar.checkbox("🧪 Enable Demo Account Mode")

# # --- Agent Selection ---
# st.sidebar.header("🧠 Choose RL Agent")
# agent_choice = st.sidebar.selectbox("Select Agent", ["PPO", "TD3", "SAC"], index=0)

# # --- Load Data ---
# tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
# data = load_data(tickers)
# price_matrix = np.stack([data[t]['Price'].values for t in tickers], axis=1)
# price_matrix = np.nan_to_num(price_matrix, nan=1.0, posinf=1.0, neginf=1.0)
# price_matrix = np.where(price_matrix < 1e-6, 1.0, price_matrix)
# log_return_matrix = np.diff(np.log(price_matrix + 1e-8), axis=0)
# log_return_matrix = np.vstack([np.zeros((1, len(tickers))), log_return_matrix])

# # --- Risk Penalty Settings ---
# penalty_map = {
#     "Low": (0.2, 0.1),
#     "Medium": (0.1, 0.05),
#     "High": (0.05, 0.01),
# }
# drawdown_penalty, vol_penalty = penalty_map[risk_level]

# # --- Load Selected Agent ---
# agent_map = {
#     "PPO": (PPO, "ppo_portfolio"),
#     "TD3": (TD3, "td3_portfolio"),
#     "SAC": (SAC, "sac_portfolio"),
# }

# model_class, model_path = agent_map[agent_choice]
# model = model_class.load(model_path)

# # --- Evaluation Environment ---
# env = PortfolioEnv(price_array=log_return_matrix, window_size=30,
#                    max_drawdown_penalty=drawdown_penalty,
#                    volatility_penalty=vol_penalty)
# obs = env.reset()
# portfolio_values = []
# weights_over_time = []
# transaction_costs = []

# while True:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     portfolio_values.append(info['portfolio_value'])
#     weights_over_time.append(info['weights'])
#     transaction_costs.append(info['transaction_cost'])
#     if done:
#         break

# portfolio_values = np.array(portfolio_values)
# weights_over_time = np.array(weights_over_time)
# transaction_costs = np.array(transaction_costs)

# # --- Evaluation Function ---
# def evaluate_strategy(values):
#     values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0)
#     values = np.where(values < 1e-8, 1e-8, values)
#     returns = np.diff(values) / values[:-1]
#     returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
#     return (
#         calculate_sharpe(returns),
#         calculate_volatility(returns),
#         calculate_cumulative_return(values)
#     )

# # --- Show Agent Performance ---
# sharpe, vol, ret = evaluate_strategy(portfolio_values)

# st.subheader(f"🤖 {agent_choice} Agent Performance")
# st.metric("Sharpe Ratio", f"{sharpe:.2f}")
# st.metric("Volatility", f"{vol:.2f}")
# st.metric("Return", f"{ret * 100:.2f}%")

# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(portfolio_values, label=f"{agent_choice} Portfolio Value", color='blue')
# ax.set_title("RL Agent Portfolio Value Over Time")
# ax.grid(True)
# ax.legend()
# st.pyplot(fig)

# # --- Baseline Comparison ---
# st.subheader("📊 Baseline Strategy Comparison")
# baselines = {
#     "Equal Weight": equal_weight_strategy(log_return_matrix),
#     "Buy & Hold": buy_and_hold_strategy(log_return_matrix),
#     "Random": random_strategy(log_return_matrix)
# }

# for name, values in baselines.items():
#     s, v, r = evaluate_strategy(values)
#     st.write(f"**{name}** | Sharpe: `{s:.2f}` | Vol: `{v:.2f}` | Return: `{r * 100:.2f}%`")

# fig2, ax2 = plt.subplots(figsize=(10, 4))
# ax2.plot(portfolio_values, label=f'{agent_choice} Agent')
# for name, values in baselines.items():
#     ax2.plot(values, label=name)
# ax2.set_title("Performance Comparison")
# ax2.legend()
# ax2.grid(True)
# st.pyplot(fig2)

# # --- Portfolio Heatmap ---
# st.subheader("🌡️ Portfolio Weights Heatmap")
# fig3, ax3 = plt.subplots(figsize=(10, 5))
# sns.heatmap(pd.DataFrame(weights_over_time, columns=tickers).T, cmap="viridis", ax=ax3)
# ax3.set_ylabel("Tickers")
# ax3.set_xlabel("Timestep")
# ax3.set_title("Portfolio Allocation Over Time")
# st.pyplot(fig3)

# # --- Transaction Costs ---
# st.subheader("📉 Transaction Costs Over Time")
# fig4, ax4 = plt.subplots(figsize=(10, 3))
# ax4.plot(transaction_costs, label="Transaction Cost", color='red')
# ax4.set_title("Transaction Costs")
# ax4.grid(True)
# st.pyplot(fig4)

# # --- Final Allocation Table ---
# st.subheader("📜 Final Portfolio Weights")
# st.table(pd.DataFrame({"Ticker": tickers, "Weight": weights_over_time[-1]}))

# # --- Demo Account ---
# if demo_mode:
#     st.subheader("🧪 Demo Account Mode")
#     live_prices = get_ltp(tickers)
#     st.write("**Mock LTP:**", live_prices)
#     place_order("MOCK_ACCOUNT", "buy", 5, "market")
#     st.write("**Mock Holdings:**", get_holdings())
#     st.write("**Mock Positions:**", get_positions())

# # --- Transaction Log Viewer ---
# if os.path.exists("logs/eval_transactions.csv"):
#     st.subheader("📑 Transaction Log Viewer")
#     df = pd.read_csv("logs/eval_transactions.csv")
#     st.dataframe(df.tail(30))
