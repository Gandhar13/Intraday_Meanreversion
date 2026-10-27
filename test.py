import pandas as pd

df = pd.read_csv("results/intraday_stationarity_results.csv")
s = df[df["ADF Stationary"] == True].copy()
s["Half-Life Days"] = s["Half-Life Bars"] / 75   # ~75 five-minute bars per day
print(s[["Stock", "ADF p-value", "Half-Life Bars", "Half-Life Days"]]
      .sort_values("Half-Life Bars").to_string(index=False))