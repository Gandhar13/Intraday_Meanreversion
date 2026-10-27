import pandas as pd
from statsmodels.tsa.stattools import adfuller

prices = pd.read_pickle("data/nifty500_prices.pkl")

results = []

print(f"Running ADF test on {len(prices.columns)} stocks...\n")

for i, stock in enumerate(prices.columns, start=1):

    series = prices[stock].dropna()

    print(f"[{i}/{len(prices.columns)}] {stock}")

    if len(series) < 100:
        continue

    try:
        result = adfuller(
            series,
            maxlag=20,
            regression="c",
            autolag="AIC"
        )

        results.append({
            "Stock": stock,
            "Observations": len(series),
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Used Lag": result[2],
            "Stationary": "Yes" if result[1] < 0.05 else "No"
        })

    except Exception as e:
        print(f"Error for {stock}: {e}")


results_df = pd.DataFrame(results)

results_df = results_df.sort_values("p-value")

results_df.to_csv(
    "results/adf_price_results_AIC.csv",
    index=False
)

stationary = results_df[results_df["p-value"] < 0.05]

print("\n" + "=" * 60)
print("ADF TEST — AIC LAG SELECTION")
print("=" * 60)

print(f"Stocks tested:       {len(results_df)}")
print(f"Stationary (5%):     {len(stationary)}")
print(f"Non-stationary:      {len(results_df) - len(stationary)}")

print("\n" + "=" * 60)
print("STATIONARY STOCKS")
print("=" * 60)

print(
    stationary[
        ["Stock", "Observations", "ADF Statistic",
         "p-value", "Used Lag"]
    ].to_string(index=False)
)

print("\nSaved to:")
print("results/adf_price_results_AIC.csv")