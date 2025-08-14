import streamlit as st
import numpy as np
import json
import os

from agents.ppo_agent import PPOAgent
from agents.td3_agent import TD3Agent
from utils.data_loader import load_data, get_price_array
from utils.metrics import calculate_sharpe, calculate_volatility, calculate_cumulative_return

TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
WINDOW_SIZE = 60
DEMO_AMOUNT = 100000
MODEL_PATHS = {
    "PPO": "ppo_portfolio.zip",
    "TD3": "td3_portfolio.zip"
}

@st.cache_data
def load_prices():
    data = load_data(TICKERS)
    price_array = get_price_array(data)
    price_array = np.nan_to_num(price_array, nan=1.0, posinf=1.0, neginf=1.0)
    price_array = np.where(price_array < 1e-6, 1.0, price_array)
    return price_array

def get_agent(name, price_array, strategy_mode, risk_level):
    kwargs = {
        "price_array": price_array,
        "window_size": WINDOW_SIZE,
        "log_dir": None,
        "strategy_mode": strategy_mode,
        "max_drawdown_penalty": {"Low": 0.01, "Medium": 0.05, "High": 0.1}[risk_level],
        "volatility_penalty": {"Low": 0.01, "Medium": 0.05, "High": 0.1}[risk_level],
        "use_optuna": False
    }
    if name == "PPO":
        return PPOAgent(**kwargs)
    elif name == "TD3":
        return TD3Agent(**kwargs)

def simulate(agent, model_path, amount):
    model = agent.load(model_path)
    obs = agent.env.reset()
    done = False
    portfolio_values, weights_time = [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = agent.env.step(action)
        portfolio_values.append(info[0]['portfolio_value'])
        weights_time.append(info[0]['weights'])

    portfolio_values = np.array(portfolio_values)
    weights = weights_time[-1] if weights_time else np.ones(len(TICKERS)) / len(TICKERS)

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = calculate_sharpe(returns)
    volatility = calculate_volatility(returns)
    total_return = calculate_cumulative_return(portfolio_values)

    final_value = amount * portfolio_values[-1]
    return final_value, sharpe, volatility, total_return, weights

# ------------------------ Streamlit UI ------------------------ #
st.title("📈 RL Portfolio Strategy Planner")

st.sidebar.header("User Preferences")

algo = st.sidebar.selectbox("Select Agent", ["PPO", "TD3"])
demo = st.sidebar.checkbox("Use Demo Account", value=True)

amount = DEMO_AMOUNT if demo else st.sidebar.number_input("Investment Amount (₹, multiple of 10,000)", step=10000, min_value=10000)
duration = st.sidebar.selectbox("Investment Duration (months)", [3, 6, 12, 18, 24])
expected_profit = st.sidebar.number_input("Expected Profit (%)", step=1, min_value=0)
risk = st.sidebar.radio("Risk Appetite", ["Low", "Medium", "High"], index=1)

if not demo and amount % 10000 != 0:
    st.sidebar.warning("Amount must be a multiple of ₹10,000")

if st.sidebar.button("📊 Plan Strategy"):
    price_array = load_prices()

    # Assume simulation on the latest N days for user's desired duration
    test_days = duration * 21
    if test_days > len(price_array) - WINDOW_SIZE:
        st.error("⛔ Not enough data for selected duration.")
    else:
        sliced_prices = price_array[-(test_days + WINDOW_SIZE):]  # simulate using the last part of data
        agent = get_agent(algo, sliced_prices, strategy_mode="long_only", risk_level=risk)

        model_path = MODEL_PATHS[algo]
        final_value, sharpe, vol, ret, weights = simulate(agent, model_path, amount)
        actual_profit = ((final_value - amount) / amount) * 100

        st.subheader("📈 Strategy Report")
        st.markdown(f"**Initial Amount:** ₹{amount:,.0f}")
        st.markdown(f"**Expected Profit:** {expected_profit:.2f}%")
        st.markdown(f"**Actual Achievable Profit:** {actual_profit:.2f}%")
        st.markdown(f"**Expected Value:** ₹{final_value:,.0f}")
        st.markdown(f"**Volatility:** {vol:.2f}")
        st.markdown(f"**Sharpe Ratio:** {sharpe:.2f}")

        st.subheader("📊 Final Portfolio Weights")
        for ticker, weight in zip(TICKERS, weights):
            st.write(f"{ticker}: {weight:.4f}")

        if actual_profit >= expected_profit:
            st.success("🎯 Goal Achieved!")
        else:
            st.warning("⚠️ Goal not fully met, but closest possible strategy has been provided.")
