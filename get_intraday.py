#This is the kaggle data downloading code

# use this code and generate the api from the kaggle website 
#pip install kagglehub pandas
#python -c "import kagglehub; kagglehub.login()"

import os
import pandas as pd
import kagglehub

# ============================================================
# SETTINGS
# ============================================================

DATASET = "debashis74017/algo-trading-data-nifty-100-data-with-indicators"

OUTPUT_FILE = "data/candidates_intraday_365.csv"   # separate file, keeps your current one safe

BAR_MINUTES = 5
LAST_DAYS = 365

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"


# ============================================================
# STOCK LIST (from your daily data)
# ============================================================

# THis is for the whole data
# prices = pd.read_pickle(PRICE_FILE)
# stocks = [str(s).replace(".NS", "") for s in prices.columns]

# print(f"Stocks to download: {len(stocks)}")

#This is for the ADF passsed stocks
res = pd.read_csv("results/intraday_stationarity_results.csv")
stocks = res[res["ADF Stationary"] == True]["Stock"].tolist()

print(f"Stocks to download: {len(stocks)}")


# ============================================================
# DOWNLOAD + CONVERT ONE STOCK AT A TIME
# ============================================================

frames = []
missing = []

for i, stock in enumerate(stocks, start=1):

    filename = f"{stock}_minute.csv"

    try:
        path = kagglehub.dataset_download(DATASET, path=filename)
    except Exception:
        print(f"[{i}/{len(stocks)}] Not found: {stock}")
        missing.append(stock)
        continue

    try:
        df = pd.read_csv(
            path,
            usecols=lambda c: c.lower().strip() in ("date", "close")
        )
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.rename(columns={"date": "Datetime", "close": "Close"})

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna()

        if df["Datetime"].dt.tz is not None:
            df["Datetime"] = df["Datetime"].dt.tz_localize(None)

        df = df.sort_values("Datetime").set_index("Datetime")

        if LAST_DAYS is not None:
            cutoff = df.index.max() - pd.Timedelta(days=LAST_DAYS)
            df = df[df.index >= cutoff]

        df = df.between_time(MARKET_OPEN, MARKET_CLOSE)

        bars = (
            df["Close"]
            .resample(f"{BAR_MINUTES}min")
            .last()
            .dropna()
        )

        out = bars.reset_index()
        out["Stock"] = stock
        frames.append(out[["Datetime", "Stock", "Close"]])

        print(f"[{i}/{len(stocks)}] {stock}: {len(out)} bars")

    except Exception as e:
        print(f"[{i}/{len(stocks)}] Error for {stock}: {e}")

    finally:
        # delete the raw file to save disk space
        try:
            os.remove(path)
        except Exception:
            pass


# ============================================================
# SAVE
# ============================================================

data = pd.concat(frames, ignore_index=True)
data = data.sort_values(["Stock", "Datetime"])
data.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Rows:", len(data))
print("Stocks:", data["Stock"].nunique())
print("Missing:", len(missing))
print(missing)