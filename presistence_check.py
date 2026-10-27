import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("data/candidates_intraday_365.csv", parse_dates=["Datetime"])
data = data.sort_values(["Stock", "Datetime"])


def analyse(series):
    if len(series) < 200:
        return np.nan, np.nan
    p = adfuller(series, maxlag=20, regression="c", autolag="AIC")[1]
    lag = series.shift(1)
    d = pd.DataFrame({"delta": series - lag, "x": lag}).dropna()
    beta = sm.OLS(d["delta"], sm.add_constant(d["x"])).fit().params["x"]
    hl = -np.log(2) / beta if beta < 0 else np.inf
    return p, hl


rows = []
for stock, g in data.groupby("Stock"):
    s = g.set_index("Datetime")["Close"]
    mid = len(s) // 2
    p1, h1 = analyse(s.iloc[:mid])
    p2, h2 = analyse(s.iloc[mid:])
    rows.append({
        "Stock": stock,
        "p_first": p1, "HL_first": h1,
        "p_second": p2, "HL_second": h2,
        "Persistent": (p1 < 0.05) and (p2 < 0.05),
    })

out = pd.DataFrame(rows).sort_values("HL_second")
out.to_csv("results/persistence_results.csv", index=False)

print(out.to_string(index=False))
print("\nStationary in both halves:", int(out["Persistent"].sum()),
      "of", len(out))