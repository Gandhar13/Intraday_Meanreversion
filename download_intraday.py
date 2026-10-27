# I have used this for the testing pourpose we can use the kaggle for the whole history fo data

import pandas as pd
import yfinance as yf

# Use the same stock list as your daily data
prices = pd.read_pickle("data/nifty500_prices.pkl")
stocks = list(prices.columns)

# Yahoo needs the .NS suffix for NSE stocks
tickers = [s if s.endswith(".NS") else f"{s}.NS" for s in stocks]

frames = []
BATCH = 50

for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i + BATCH]
    print(f"Downloading {i + 1}-{i + len(batch)} of {len(tickers)}")

    df = yf.download(
        batch,
        period="60d",
        interval="5m",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    for t in batch:
        try:
            close = df[t]["Close"].dropna()
        except KeyError:
            print(f"  No data: {t}")
            continue
        if close.empty:
            continue
        out = close.reset_index()
        out.columns = ["Datetime", "Close"]
        out["Stock"] = t.replace(".NS", "")
        frames.append(out)

data = pd.concat(frames, ignore_index=True)
data["Datetime"] = pd.to_datetime(data["Datetime"]).dt.tz_localize(None)
data = data[["Datetime", "Stock", "Close"]]

data.to_csv("data/nifty500_intraday.csv", index=False)
print(f"Saved {len(data)} rows for {data['Stock'].nunique()} stocks")