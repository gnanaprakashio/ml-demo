import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# House Price Analysis - Exploratory Data Analysis (EDA)

This notebook explores the house price dataset before building the
Linear Regression model. It covers:
- Loading and inspecting the data
- Handling missing values
- Visualizing relationships between features and price
- Checking correlations between variables
"""))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("../dataset/house_prices.csv")
df.head()"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Basic dataset information"))
cells.append(nbf.v4.new_code_cell(
"""print("Shape:", df.shape)
df.info()"""))
cells.append(nbf.v4.new_code_cell("df.describe()"))
cells.append(nbf.v4.new_code_cell("df.isnull().sum()"))

cells.append(nbf.v4.new_markdown_cell("## 2. Handle missing values\n\nWe fill missing numeric values with the column median."))
cells.append(nbf.v4.new_code_cell(
"""for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
df.isnull().sum()"""))

cells.append(nbf.v4.new_markdown_cell("## 3. House Price Distribution\nShows how house prices are spread out. Most houses fall in the middle range, forming a roughly bell-shaped distribution."))
cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,5))
sns.histplot(df["Price"], kde=True, color="teal")
plt.title("House Price Distribution")
plt.xlabel("Price (INR)")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Area vs Price\nShows a clear positive relationship — as area increases, price tends to increase too."))
cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,5))
sns.scatterplot(x="Area", y="Price", data=df, color="darkorange")
plt.title("Area vs Price")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Bedrooms vs Price\nShows how the number of bedrooms affects price on average."))
cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,5))
sns.boxplot(x="Bedrooms", y="Price", data=df, hue="Bedrooms", palette="Set2", legend=False)
plt.title("Bedrooms vs Price")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Correlation Heatmap\nShows how strongly each feature is correlated with price and with each other. Area typically shows the strongest correlation with price."))
cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Conclusion

- `Area` has the strongest positive correlation with `Price`.
- `Bedrooms`, `Bathrooms`, `Floors`, and `Parking` also positively influence price.
- `Age` has a mild negative effect (older houses tend to be cheaper).
- These relationships are roughly linear, which is why Linear Regression
  is a suitable model for this problem — see `train_model.py`.
"""))

nb['cells'] = cells

with open("house_price_analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created: house_price_analysis.ipynb")
