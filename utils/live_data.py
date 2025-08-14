import yfinance as yf
import pandas as pd
import datetime

def fetch_live_data(tickers, period="7d", interval="1h"):
    """
    Fetch recent live price data for given tickers.
    
    Args:
        tickers (list of str): e.g. ["RELIANCE.NS", "HDFCBANK.NS"]
        period (str): e.g. "1d", "5d", "7d", "1mo", etc.
        interval (str): e.g. "1m", "5m", "15m", "1h", "1d"

    Returns:
        pd.DataFrame: Multi-column DataFrame with prices indexed by datetime.
    """
    data = yf.download(tickers=tickers, period=period, interval=interval, group_by='ticker', threads=True, auto_adjust=True)

    if len(tickers) == 1:
        # Single ticker case
        return pd.DataFrame({tickers[0]: data['Close']}).dropna()
    else:
        close_prices = {}
        for t in tickers:
            if t in data:
                close_prices[t] = data[t]['Close']
        return pd.DataFrame(close_prices).dropna()

# Example usage
if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "HDFCBANK.NS"]
    df = fetch_live_data(tickers)
    print(df.tail())
