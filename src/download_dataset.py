from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

# Fetch California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # Pandas DataFrame

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

# Save as CSV
df.to_csv("data/raw_data.csv", index=False)

print("Dataset saved to data/raw_data.csv — rows:", df.shape[0], "cols:", df.shape[1])
