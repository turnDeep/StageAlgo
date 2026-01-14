import trendln
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch real data with dates
df = yf.download("QQQ", period="1mo", interval="1d")
if hasattr(df.columns, 'droplevel'):
    df.columns = df.columns.droplevel(1)

try:
    fig = trendln.plot_support_resistance(df['Close'], accuracy=2)
    plt.savefig('test_trendln_plot.png')
    print("Saved test plot")
except Exception as e:
    print("Error:", e)
