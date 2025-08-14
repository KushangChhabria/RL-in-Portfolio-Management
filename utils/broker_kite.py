import os
import datetime

# Kite Connect SDK (install with `pip install kiteconnect`)
from kiteconnect import KiteConnect, KiteTicker

KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_API_SECRET = os.getenv("KITE_API_SECRET")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")  # Store after login

# Initialize KiteConnect object
kite = KiteConnect(api_key=KITE_API_KEY)

if KITE_ACCESS_TOKEN:
    kite.set_access_token(KITE_ACCESS_TOKEN)


def login_url():
    """Returns the login URL to authenticate user manually (once)."""
    return kite.login_url()


def generate_access_token(request_token):
    """Once user logs in manually and gets `request_token`, generate session."""
    data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = data["access_token"]
    return access_token


def get_account_balance():
    """Returns available cash balance."""
    try:
        profile = kite.margins()["equity"]
        return profile["available"]["cash"]
    except Exception as e:
        print("❌ Error fetching balance:", e)
        return 0.0


def place_order(symbol, quantity, transaction_type="BUY", order_type="MARKET"):
    """Places a market order for given symbol and quantity."""
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=kite.PRODUCT_MIS
        )
        print(f"✅ Order placed: {symbol} | Qty: {quantity} | Type: {transaction_type}")
        return order_id
    except Exception as e:
        print("❌ Order failed:", e)
        return None


def get_positions():
    """Returns current portfolio positions."""
    try:
        positions = kite.positions()["net"]
        return positions
    except Exception as e:
        print("❌ Failed to fetch positions:", e)
        return []


def get_holdings():
    """Returns long-term holdings (invested stocks)."""
    try:
        return kite.holdings()
    except Exception as e:
        print("❌ Failed to fetch holdings:", e)
        return []


# 🧪 TEST SCRIPT
if __name__ == "__main__":
    if not KITE_ACCESS_TOKEN:
        print(" Please go to the following URL, login, and paste request_token below:")
        print(login_url())
        request_token = input("Paste request_token: ").strip()
        access_token = generate_access_token(request_token)
        print("✅ ACCESS TOKEN (store as env):", access_token)
    else:
        print(" Already authenticated.")
        print("Balance:", get_account_balance())
        print("Holdings:", get_holdings())
        print("Positions:", get_positions())