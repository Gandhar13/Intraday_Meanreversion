import pandas as pd
import numpy as np
import statsmodels.api as sm


# ============================================================
# SETTINGS
# ============================================================

PRICE_FILE = "data/nifty500_prices.pkl"
ADF_FILE = "results/adf_price_results_AIC.csv"

OUTPUT_FILE = "results/half_life_results.csv"
FILTERED_FILE = "results/half_life_filtered.csv"

# ------------------------------------------------------------
# Initial research filter
#
# These values are NOT claimed to be optimal.
# They are used to remove stocks with extremely slow
# or extremely fast estimated mean reversion.
# ------------------------------------------------------------

MIN_HALF_LIFE_DAYS = 10
MAX_HALF_LIFE_DAYS = 60


# ============================================================
# LOAD DATA
# ============================================================

prices = pd.read_pickle(PRICE_FILE)

adf = pd.read_csv(ADF_FILE)


# ============================================================
# KEEP ONLY ADF-STATIONARY STOCKS
# ============================================================

candidates = adf[
    adf["p-value"] < 0.05
]["Stock"].tolist()


print("=" * 70)
print("HALF-LIFE ANALYSIS")
print("=" * 70)

print(
    f"ADF-stationary stocks: {len(candidates)}"
)

print(
    f"Half-life filter: "
    f"{MIN_HALF_LIFE_DAYS}–{MAX_HALF_LIFE_DAYS} days"
)

print()


# ============================================================
# HALF-LIFE CALCULATION
# ============================================================

results = []


for stock in candidates:

    series = prices[stock].dropna()

    if len(series) < 100:
        continue



    lagged = series.shift(1)

  

    delta = series - lagged

    temp = pd.DataFrame({
        "x_lag": lagged,
        "delta": delta
    }).dropna()



    X = sm.add_constant(
        temp["x_lag"]
    )

    y = temp["delta"]

    model = sm.OLS(
        y,
        X
    ).fit()

    beta = model.params["x_lag"]


    # --------------------------------------------------------
    # Calculate half-life
    # --------------------------------------------------------

    if beta < 0:

        half_life = (
            -np.log(2) / beta
        )

    else:

        half_life = np.inf


    # --------------------------------------------------------
    # Half-life filter
    # --------------------------------------------------------

    passes_filter = (
        MIN_HALF_LIFE_DAYS
        <= half_life
        <= MAX_HALF_LIFE_DAYS
    )




    results.append({

        "Stock": stock,

        "Observations": len(series),

        "ADF p-value": adf.loc[
            adf["Stock"] == stock,
            "p-value"
        ].iloc[0],

        "Beta": beta,

        "Half-Life (days)": half_life,

        "Half-Life Filter": passes_filter

    })




results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Half-Life (days)"
)


# ============================================================
# FINAL FILTERED UNIVERSE
# ============================================================

filtered = results_df[
    results_df["Half-Life Filter"] == True
].copy()



results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

filtered.to_csv(
    FILTERED_FILE,
    index=False
)


# ============================================================
# OUTPUT 1 — ALL ADF-STATIONARY STOCKS
# ============================================================

print("=" * 70)
print("ALL ADF-STATIONARY STOCKS")
print("=" * 70)

print(
    results_df[
        [
            "Stock",
            "Observations",
            "ADF p-value",
            "Beta",
            "Half-Life (days)"
        ]
    ].to_string(index=False)
)


# ============================================================
# OUTPUT 2 — AFTER HALF-LIFE FILTER
# ============================================================

print("\n")
print("=" * 70)
print("STOCKS REMAINING AFTER HALF-LIFE FILTER")
print("=" * 70)

print(
    f"Filter: {MIN_HALF_LIFE_DAYS}–"
    f"{MAX_HALF_LIFE_DAYS} days"
)

print(
    f"\nStocks before filter: {len(results_df)}"
)

print(
    f"Stocks after filter:  {len(filtered)}"
)

print()

if len(filtered) > 0:

    print(
        filtered[
            [
                "Stock",
                "Observations",
                "ADF p-value",
                "Beta",
                "Half-Life (days)"
            ]
        ].to_string(index=False)
    )

else:

    print("No stocks passed the half-life filter.")




print("\n")
print("=" * 70)
print("SAVED FILES")
print("=" * 70)

print(
    OUTPUT_FILE
)

print(
    FILTERED_FILE
)