"""Make labeled candlestick images for many stocks.

Each image shows 20 trading days. Label = UP if the close 5 trading days after
day 20 is higher than the close on day 20, else DOWN.
Saved as candlesticks/UP|DOWN/TICKER_STARTDATE_ENDDATE.png (team contract).

Run from the Team-10 folder:  python generate_images.py
Safe to stop and re-run: it skips images that already exist.
"""
import os

import matplotlib
matplotlib.use("Agg")  # draw straight to files, never open a window (much faster)
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import yfinance as yf

# ponytail: today's big stocks = survivorship bias (we skip companies that died). Say so in the slides.
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "JNJ", "V", "PG", "XOM",
    "UNH", "HD", "MA", "BAC", "KO", "PEP", "CVX", "MRK", "ABT", "CSCO",
    "INTC", "ORCL", "WMT", "DIS", "MCD", "NKE", "T", "VZ", "PFE", "CMCSA",
    "ADBE", "CRM", "QCOM", "TXN", "IBM", "AMGN", "HON", "CAT", "GS", "MS",
    "BA", "MMM", "GE", "LOW", "SBUX", "COST", "UPS", "LMT", "BMY", "GILD",
]
START, END = "2010-01-01", "2025-01-01"  # 15 years
WINDOW = 20   # days in one picture
FORWARD = 5   # how many trading days ahead we predict
STEP = 20     # new picture every 20 days: no two pictures share a day, ~9.4k images (was 5 -> 37.5k)
OUT_DIR = "candlesticks"
CACHE = "prices.pkl"  # downloaded prices, saved so re-runs don't hit Yahoo again


def download():
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True,
                         group_by="ticker", progress=False)
    prices.to_pickle(CACHE)
    return prices


def label_windows(df, window=WINDOW, forward=FORWARD, step=STEP):
    """Yield (20-day slice, "UP"/"DOWN") for one stock's price table."""
    for i in range(0, len(df) - window - forward + 1, step):  # stop early so day i+24 always exists
        w = df.iloc[i : i + window]
        future_close = df["Close"].iloc[i + window + forward - 1]
        yield w, "UP" if future_close > w["Close"].iloc[-1] else "DOWN"


def image_name(ticker, w):
    return f"{ticker}_{w.index[0]:%Y%m%d}_{w.index[-1]:%Y%m%d}.png"


def draw(w, path):  # same look as sample_image.py: 300x300, no axes
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    mpf.plot(w, type="candle", ax=ax, style="charles")
    ax.set_axis_off()
    fig.savefig(path, dpi=100, facecolor="white", pad_inches=0)
    plt.close(fig)  # free the memory, or thousands of open charts crash Python


def main():
    prices = download()
    for label in ("UP", "DOWN"):
        os.makedirs(f"{OUT_DIR}/{label}", exist_ok=True)

    counts = {"UP": 0, "DOWN": 0}
    for ticker in TICKERS:
        df = prices[ticker].dropna()  # drop days before the stock existed
        if len(df) < WINDOW + FORWARD:
            print(f"{ticker}: no data, skipped")
            continue
        for w, label in label_windows(df):
            path = f"{OUT_DIR}/{label}/{image_name(ticker, w)}"
            if not os.path.exists(path):
                draw(w, path)
            counts[label] += 1
        print(f"{ticker} done. Total so far: {counts}")

    total = sum(counts.values())
    print(f"\nDone: {total} images, {counts['UP'] / total:.1%} UP")


if __name__ == "__main__":
    main()
