# utils/broker_kite_mock.py
import random
import time
from datetime import datetime

# Simulated live price source (can use yfinance or random walk)
mock_prices = {
    "RELIANCE.NS": 2800.0,
    "HDFCBANK.NS": 1550.0,
    "INFY.NS": 1450.0,
    "TCS.NS": 3600.0,
    "ITC.NS": 440.0
}

order_history = []
holdings = {}


def get_ltp(tickers):
    ltp_data = {}
    for ticker in tickers:
        # Simulate small price fluctuation
        fluctuation = random.uniform(-0.01, 0.01) * mock_prices.get(ticker, 100)
        mock_prices[ticker] += fluctuation
        ltp_data[ticker] = round(mock_prices[ticker], 2)
    return ltp_data


def place_order(symbol, side, quantity, order_type='market'):
    print(f"📦 MOCK ORDER => {side.upper()} {quantity} units of {symbol} as a {order_type} order.")
    return True


    # Log order
    order_history.append({
        "order_id": order_id,
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "status": status,
        "timestamp": timestamp
    })

    # Simulate holdings update
    if side == "buy":
        holdings[symbol] = holdings.get(symbol, 0) + qty
    elif side == "sell":
        holdings[symbol] = holdings.get(symbol, 0) - qty

    return {
        "order_id": order_id,
        "status": status,
        "timestamp": timestamp
    }


def get_positions():
    return holdings.copy()


def get_holdings():
    return holdings.copy()


def get_order_history():
    return order_history.copy()
