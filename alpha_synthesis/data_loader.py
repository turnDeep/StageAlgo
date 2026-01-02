import yfinance as yf
from curl_cffi import requests as curl_requests
import time
import random

class AlphaSynthesisDataLoader:
    def __init__(self):
        # Create session impersonating Chrome
        self.session = curl_requests.Session(impersonate="chrome")

    def fetch_data(self, ticker):
        """
        Fetches history and financials for a ticker.
        Returns (history_df, financials_df) or (None, None) if failed.
        """
        try:
            obj = yf.Ticker(ticker, session=self.session)

            # Fetch history
            # Need enough for 200MA and 52-week high anchor (252 days)
            # Fetching 2y to be safe
            hist = obj.history(period="2y")

            # Random sleep to avoid rate limits
            time.sleep(random.uniform(0.5, 1.0))

            if hist.empty:
                return None, None

            # Fetch financials
            # Note: financial fetch might be separate request.
            # Accessing .financials triggers a request.
            financials = None
            try:
                financials = obj.financials
                time.sleep(random.uniform(0.5, 1.0))
            except Exception:
                # Financials often fail or are empty for ETFs/small caps.
                # Don't fail the whole fetch.
                pass

            return hist, financials

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error fetching {ticker}: {e}")
            time.sleep(10) # Backoff
            return None, None

    def close(self):
        self.session.close()
