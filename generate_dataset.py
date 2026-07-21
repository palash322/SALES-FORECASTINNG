import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(
    start="2020-01-01",
    end="2025-12-31",
    freq="D"
)

sales = (
    500
    + np.linspace(0, 300, len(dates))
    + 80 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    + np.random.normal(0, 25, len(dates))
)

sales = np.maximum(sales, 50).round(2)

df = pd.DataFrame({
    "Date": dates,
    "Sales": sales
})

df.to_csv("data/train.csv", index=False)

print("Dataset created successfully!")
print(df.head())
