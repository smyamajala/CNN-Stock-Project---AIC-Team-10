
import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

# 1. Download historical stock data
ticker = "AAPL"

df = yf.download(
    ticker,
    start="2020-01-01",
    end="2025-12-31",
    auto_adjust=True,
    progress=False
)

# Handle multi-level column names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Check that data was downloaded
if df.empty:
    raise ValueError("No stock data was downloaded.")

print(f"Downloaded {len(df)} trading days.")

# 2. Create output folders
output_dir = "sample_candlesticks"

os.makedirs(f"{output_dir}/UP", exist_ok=True)
os.makedirs(f"{output_dir}/DOWN", exist_ok=True)

# 3. Set parameters
window_size = 20
forward_days = 5
number_of_images = 20

# 4. Generate sample images
for i in range(number_of_images):

    # Get 20 days of stock data
    window_df = df.iloc[i : i + window_size]

    # Get closing price on day 20
    current_close = window_df["Close"].iloc[-1]

    # Get closing price 5 trading days later
    future_close = df["Close"].iloc[
        i + window_size + forward_days - 1
    ]

    # Assign UP or DOWN label
    label = "UP" if future_close > current_close else "DOWN"

    # Create filename
    start_date = window_df.index[0].strftime("%Y%m%d")
    end_date = window_df.index[-1].strftime("%Y%m%d")

    filename = (
        f"{output_dir}/{label}/"
        f"{ticker}_{start_date}_{end_date}.png"
    )

    # 5. Generate candlestick image
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

    mpf.plot(
        window_df,
        type="candle",
        ax=ax,
        style="charles"
    )

    # Remove axes and labels
    ax.set_axis_off()

    # Save image at a consistent figure size
    plt.savefig(
        filename,
        dpi=100,
        facecolor="white",
        pad_inches=0
    )

    plt.close(fig)

    print(
        f"Image {i + 1}/{number_of_images}: "
        f"{label} - {filename}"
    )

print("\nSample image generation complete!")
