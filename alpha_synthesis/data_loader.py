import yfinance as yf
from curl_cffi import requests as curl_requests
import time
import random
import traceback

class AlphaSynthesisDataLoader:
    def __init__(self):
        # Create session impersonating Chrome
        self.session = curl_requests.Session(impersonate="chrome")

    def fetch_data(self, ticker):
        """
        Fetches history and financials for a ticker.
        Returns (history_df, financials_df) or (None, None) if failed.
        """
        retries = 3
        for attempt in range(retries):
            try:
                obj = yf.Ticker(ticker, session=self.session)

                # Fetch history
                # Need enough for 200MA and 52-week high anchor (252 days)
                # Fetching 2y to be safe
                hist = obj.history(period="2y")

                # Random sleep to avoid rate limits
                time.sleep(random.uniform(0.5, 1.0))

                if hist is None or hist.empty:
                    # Retry if empty result but no error raised (sometimes happens)
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
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

            except TypeError as e:
                # Catch yfinance "NoneType is not subscriptable" error
                # This usually means blocked request or empty JSON
                # print(f"  [Attempt {attempt+1}/{retries}] blocked/empty for {ticker}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 + attempt * 2) # Exponential backoff
                    continue
                else:
                    return None, None

            except Exception as e:
                # traceback.print_exc()
                # print(f"Error fetching {ticker}: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    return None, None

        return None, None

    def close(self):
        self.session.close()
