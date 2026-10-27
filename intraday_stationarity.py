import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/nifty500_intraday.csv"

# 5-minute candles
BAR_MINUTES = 5

# ADF significance level
ALPHA = 0.05

# Maximum ADF lag
MAX_LAG = 20

# Intraday half-life filter
#
# Minimum: 1 bars = 5 minutes
# Maximum: 30 bars = 2.5 hours
#
# These are initial research filters.
MIN_HALF_LIFE_BARS = 1
MAX_HALF_LIFE_BARS = 30


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_FILE)

data["Datetime"] = pd.to_datetime(data["Datetime"])

data = data.sort_values(
    ["Stock", "Datetime"]
)

print("Total rows:", len(data))
print("Number of stocks:", data["Stock"].nunique())


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH STOCK
# ============================================================

stocks = data["Stock"].unique()

for i, stock in enumerate(stocks, start=1):

    print(f"[{i}/{len(stocks)}] {stock}")

    stock_data = data[
        data["Stock"] == stock
    ].copy()

    # --------------------------------------------------------
    # Remove missing prices
    # --------------------------------------------------------

    series = (
        stock_data["Close"]
        .dropna()
    )

    # Need enough observations
    if len(series) < 200:

        continue


    # ========================================================
    # ADF TEST
    # ========================================================

    try:

        adf_result = adfuller(
            series,
            maxlag=MAX_LAG,
            regression="c",
            autolag="AIC"
        )

        adf_stat = adf_result[0]
        p_value = adf_result[1]
        used_lag = adf_result[2]


    except Exception as e:

        print(
            f"ADF error for {stock}: {e}"
        )

        continue


    # ========================================================
    # ONLY CONTINUE IF ADF SAYS STATIONARY
    # ========================================================

    if p_value >= ALPHA:

        results.append({

            "Stock": stock,

            "Observations": len(series),

            "ADF Statistic": adf_stat,

            "ADF p-value": p_value,

            "ADF Lag": used_lag,

            "ADF Stationary": False,

            "Beta": np.nan,

            "Half-Life Bars": np.nan,

            "Half-Life Minutes": np.nan,

            "Half-Life Hours": np.nan,

            "Half-Life Filter": False

        })

        continue


    # ========================================================
    # HALF-LIFE CALCULATION
    # ========================================================

    lagged = series.shift(1)

    delta = series - lagged

    regression_data = pd.DataFrame({

        "delta": delta,

        "lagged_price": lagged

    }).dropna()


    X = sm.add_constant(
        regression_data["lagged_price"]
    )

    y = regression_data["delta"]


    model = sm.OLS(
        y,
        X
    ).fit()


    beta = model.params["lagged_price"]


    # --------------------------------------------------------
    # Calculate half-life
    # --------------------------------------------------------

    if beta < 0:

        half_life_bars = (
            -np.log(2) / beta
        )

        half_life_minutes = (
            half_life_bars * BAR_MINUTES
        )

        half_life_hours = (
            half_life_minutes / 60
        )

    else:

        half_life_bars = np.inf
        half_life_minutes = np.inf
        half_life_hours = np.inf


    # ========================================================
    # HALF-LIFE FILTER
    # ========================================================

    passes_half_life = (

        MIN_HALF_LIFE_BARS
        <= half_life_bars
        <= MAX_HALF_LIFE_BARS

    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "Stock": stock,

        "Observations": len(series),

        "ADF Statistic": adf_stat,

        "ADF p-value": p_value,

        "ADF Lag": used_lag,

        "ADF Stationary": True,

        "Beta": beta,

        "Half-Life Bars": half_life_bars,

        "Half-Life Minutes": half_life_minutes,

        "Half-Life Hours": half_life_hours,

        "Half-Life Filter": passes_half_life

    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# FINAL CANDIDATES
# ============================================================

final_candidates = results_df[
    (results_df["ADF Stationary"] == True)
    &
    (results_df["Half-Life Filter"] == True)
].copy()


# Sort by half-life

final_candidates = final_candidates.sort_values(
    "Half-Life Minutes"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results/intraday_stationarity_results.csv",
    index=False
)

final_candidates.to_csv(
    "results/intraday_final_candidates.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("INTRADAY STATIONARITY ANALYSIS")
print("=" * 70)

print(
    "Stocks tested:",
    len(stocks)
)

print(
    "ADF stationary:",
    len(
        results_df[
            results_df["ADF Stationary"] == True
        ]
    )
)

print(
    "Passed half-life filter:",
    len(final_candidates)
)


print("\n")
print("=" * 70)
print("FINAL INTRADAY CANDIDATES")
print("=" * 70)

if len(final_candidates) > 0:

    print(
        final_candidates[
            [
                "Stock",
                "Observations",
                "ADF Statistic",
                "ADF p-value",
                "Half-Life Bars",
                "Half-Life Minutes",
                "Half-Life Hours"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "No stocks passed both filters."
    )


print("\nSaved:")
print(
    "results/intraday_stationarity_results.csv"
)

print(
    "results/intraday_final_candidates.csv"
)