import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests

MIN_HL, MAX_HL = 10, 60

prices = pd.read_pickle("data/nifty500_prices.pkl")
cands = pd.read_csv("results/half_life_filtered.csv")["Stock"].tolist()


def analyse(s):
    s = s.dropna()
    if len(s) < 100:
        return np.nan, np.nan
    p = adfuller(s, maxlag=20, regression="c", autolag="AIC")[1]
    lag = s.shift(1)
    d = pd.DataFrame({"delta": s - lag, "x": lag}).dropna()
    beta = sm.OLS(d["delta"], sm.add_constant(d["x"])).fit().params["x"]
    hl = -np.log(2) / beta if beta < 0 else np.inf
    return p, hl


rows = []
for stock in cands:
    s = prices[stock].dropna()
    mid = len(s) // 2
    p1, h1 = analyse(s.iloc[:mid])
    p2, h2 = analyse(s.iloc[mid:])
    rows.append({"Stock": stock, "Obs": len(s),
                 "p_first": p1, "HL_first": h1,
                 "p_second": p2, "HL_second": h2})

out = pd.DataFrame(rows)

# Multiple-testing correction on each half
for col in ("p_first", "p_second"):
    ok = out[col].notna()
    out.loc[ok, col + "_BH"] = multipletests(out.loc[ok, col],
                                             method="fdr_bh")[1]

out["Persistent"] = (
    (out["p_first"] < 0.05) & (out["p_second"] < 0.05)
    & out["HL_first"].between(MIN_HL, MAX_HL)
    & out["HL_second"].between(MIN_HL, MAX_HL)
)

out = out.sort_values("HL_second")
out.to_csv("results/daily_persistence.csv", index=False)
print(out.to_string(index=False))
print("\nPersistent in both halves:", int(out["Persistent"].sum()),
      "of", len(out))