import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.live_data import fetch_live_data

TICKERS = ["RELIANCE.NS", "HDFCBANK.NS"]

df = fetch_live_data(TICKERS, period="2d", interval="1h")
print(df.tail())
