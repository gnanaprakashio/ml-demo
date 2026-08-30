"""
generate_dataset.py
--------------------
This script creates a synthetic (but realistic) house price dataset
for the college ML assignment. It is run ONCE to produce
'house_prices.csv'. Students can later replace this file with a
real-world dataset (e.g., from Kaggle) if required.

The relationship between features and price is built with simple
linear logic + random noise, so a Linear Regression model can learn
it well -- which is exactly what we want for a beginner assignment.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # for reproducible results

N = 500  # number of houses (rows) in the dataset

# ---- Generate feature columns ----
area = np.random.randint(500, 4000, N)          # Area in sq.ft
bedrooms = np.random.randint(1, 6, N)            # Number of bedrooms
bathrooms = np.random.randint(1, 4, N)           # Number of bathrooms
floors = np.random.randint(1, 4, N)              # Number of floors
age = np.random.randint(0, 30, N)                # Age of house in years
parking = np.random.randint(0, 3, N)             # Parking spaces

# ---- Build price using a linear formula + noise ----
# Base price + contribution from each feature (values in INR)
price = (
    300000                       # base price
    + area * 1800                # price increases with area
    + bedrooms * 150000           # more bedrooms -> higher price
    + bathrooms * 100000          # more bathrooms -> higher price
    + floors * 80000              # more floors -> higher price
    - age * 5000                  # older house -> lower price
    + parking * 60000             # parking adds value
    + np.random.normal(0, 150000, N)  # random noise (market variation)
)

price = np.clip(price, 300000, None).round(-3)  # no negative prices, round to nearest 1000

df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "Age": age,
    "Parking": parking,
    "Price": price.astype(int)
})

# ---- Introduce a few missing values on purpose ----
# This lets the assignment demonstrate "handling missing values"
missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx, "Bathrooms"] = np.nan

df.to_csv("house_prices.csv", index=False)
print("Dataset created: house_prices.csv")
print(df.head())
