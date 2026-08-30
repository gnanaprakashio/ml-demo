"""
train_model.py
----------------
This script trains a Linear Regression model to predict house prices.

STEPS PERFORMED:
1. Load the dataset
2. Display basic info about the dataset
3. Handle missing values
4. Perform basic preprocessing
5. Perform Exploratory Data Analysis (EDA) and save graphs
6. Select features
7. Split into train/test sets
8. Train a Linear Regression model
9. Evaluate the model (MAE, MSE, RMSE, R2)
10. Save the trained model using Joblib

Run this file BEFORE running app.py:
    python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so plots can be saved without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders for outputs if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# -----------------------------------------------------------------
# STEP 1: Load the dataset
# -----------------------------------------------------------------
print("STEP 1: Loading dataset...")
df = pd.read_csv("dataset/house_prices.csv")

# -----------------------------------------------------------------
# STEP 2: Display basic information about the dataset
# -----------------------------------------------------------------
print("\nSTEP 2: Basic dataset information")
print("Shape of dataset:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())

# -----------------------------------------------------------------
# STEP 3: Handle missing values
# -----------------------------------------------------------------
print("\nSTEP 3: Handling missing values...")
# We fill missing numeric values with the median of that column.
# Median is preferred over mean because it is less affected by outliers.
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Filled missing values in '{col}' with median = {median_value}")

# -----------------------------------------------------------------
# STEP 4: Basic data preprocessing
# -----------------------------------------------------------------
print("\nSTEP 4: Basic preprocessing...")
# Remove duplicate rows, if any
before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before - df.shape[0]} duplicate rows")

# -----------------------------------------------------------------
# STEP 5: Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------
print("\nSTEP 5: Performing EDA and saving graphs to static/graphs/ ...")

# Graph 1: House price distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Price"], kde=True, color="teal")
plt.title("House Price Distribution")
plt.xlabel("Price (INR)")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig("static/graphs/price_distribution.png")
plt.close()

# Graph 2: Area vs Price
plt.figure(figsize=(7, 5))
sns.scatterplot(x="Area", y="Price", data=df, color="darkorange")
plt.title("Area vs Price")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig("static/graphs/area_vs_price.png")
plt.close()

# Graph 3: Bedrooms vs Price
plt.figure(figsize=(7, 5))
sns.boxplot(x="Bedrooms", y="Price", data=df, hue="Bedrooms", palette="Set2", legend=False)
plt.title("Bedrooms vs Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig("static/graphs/bedrooms_vs_price.png")
plt.close()

# Graph 4: Correlation heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/graphs/correlation_heatmap.png")
plt.close()

print("EDA graphs saved successfully.")

# -----------------------------------------------------------------
# STEP 6: Select features and target variable
# -----------------------------------------------------------------
print("\nSTEP 6: Selecting features...")
FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "Age", "Parking"]
TARGET = "Price"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------------------------------------------
# STEP 7: Train-test split
# -----------------------------------------------------------------
print("\nSTEP 7: Splitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------------------------------------------
# STEP 8: Train the Linear Regression model
# -----------------------------------------------------------------
print("\nSTEP 8: Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")
print("Model coefficients:", dict(zip(FEATURES, model.coef_)))
print("Model intercept:", model.intercept_)

# -----------------------------------------------------------------
# STEP 9: Evaluate the model
# -----------------------------------------------------------------
print("\nSTEP 9: Evaluating model performance on test data...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  (Mean Absolute Error)      : {mae:,.2f}")
print(f"MSE  (Mean Squared Error)       : {mse:,.2f}")
print(f"RMSE (Root Mean Squared Error)  : {rmse:,.2f}")
print(f"R2 Score                        : {r2:.4f}")

# Save evaluation metrics to a text file (useful for README/report)
with open("model/evaluation_metrics.txt", "w") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("=========================\n")
    f.write(f"MAE  : {mae:,.2f}\n")
    f.write(f"MSE  : {mse:,.2f}\n")
    f.write(f"RMSE : {rmse:,.2f}\n")
    f.write(f"R2   : {r2:.4f}\n")

# -----------------------------------------------------------------
# STEP 10: Save the trained model using Joblib
# -----------------------------------------------------------------
print("\nSTEP 10: Saving trained model to model/house_price_model.pkl ...")
joblib.dump(model, "model/house_price_model.pkl")
print("Model saved successfully!")

print("\nAll steps completed. You can now run 'python app.py' to start the web app.")
