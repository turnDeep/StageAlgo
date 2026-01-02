import yfinance as yf
from curl_cffi import requests as curl_requests
import time
import random

class AlphaSynthesisDataLoader:
    def __init__(self):
        # Session not needed for default yfinance (it handles it)
        pass

    def fetch_data(self, ticker):
        """
        Fetches history and financials for a ticker.
        Returns (history_df, financials_df) or (None, None) if failed.
        """
        try:
            # Use default yfinance session mechanism which is currently more reliable
            obj = yf.Ticker(ticker)

            # Fetch history
            # Need enough for 200MA and 52-week high anchor (252 days)
            # Fetching 2y to be safe
            hist = obj.history(period="2y")

            # Random sleep to avoid rate limits
            time.sleep(random.uniform(1.5, 3.0))

            if hist.empty:
                return None, None

            # Fetch financials
            # Note: financial fetch might be separate request.
            # Accessing .financials triggers a request.
            financials = None
            try:
                financials = obj.financials
                time.sleep(random.uniform(1.5, 3.0))
            except Exception:
                # Financials often fail or are empty for ETFs/small caps.
                # Don't fail the whole fetch.
                pass

            return hist, financials

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            time.sleep(10) # Backoff
            return None, None

    def close(self):
        pass
