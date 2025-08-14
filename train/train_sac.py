import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import load_data, get_price_array, train_test_split
from agents.sac_agent import SACAgent

tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']

data = load_data(tickers)
price_array = get_price_array(data)
train_array, _ = train_test_split(price_array, split_date='2023-07-01', data=data, tickers=tickers)

if (train_array.std(axis=0) < 1e-6).any():
    raise ValueError("Training price data has near-zero variance for some assets.")

agent = SACAgent(
    price_array=train_array,
    window_size=30,
    log_dir="./logs_sac",
    max_drawdown_penalty=0.1,
    volatility_penalty=0.05
)

agent.train(timesteps=50_000)
agent.save("sac_portfolio")
