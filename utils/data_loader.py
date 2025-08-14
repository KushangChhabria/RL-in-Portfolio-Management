# utils/data_loader.py
import os
import time
import pandas as pd
import yfinance as yf
import numpy as np

def load_data(tickers):
    data = {}
    for ticker in tickers:
        df = pd.read_csv(f"data/{ticker}.csv")
        df['Date'] = pd.to_datetime(df['Date'])  # ✅ Ensure datetime
        df.set_index('Date', inplace=True)
        df['log_return'] = df['log_return'].replace([np.inf, -np.inf], 0).fillna(0)
        data[ticker] = df
    return data



def get_price_array(data: dict) -> np.ndarray:
    price_matrix = pd.concat([df['Price'] for df in data.values()], axis=1)
    price_matrix.columns = list(data.keys())

    # Drop rows where any price is non-finite or zero
    price_matrix = price_matrix.replace([np.inf, -np.inf], np.nan).dropna()
    price_matrix = price_matrix[(price_matrix > 1e-3).all(axis=1)]  # remove 0 or near-zero rows

    return price_matrix.values


def train_test_split(price_array, split_date, data, tickers):
    date_index = data[tickers[0]].index
    split_ts = pd.to_datetime(split_date)

    if split_ts not in date_index:
        split_ts = date_index[date_index.get_indexer([split_ts], method='bfill')[0]]
        print(f"⚠️ Split date '{split_date}' not found. Using next trading day: {split_ts.date()}")

    split_index = date_index.get_loc(split_ts)
    train = price_array[:split_index]
    test = price_array[split_index:]
    return train, test
