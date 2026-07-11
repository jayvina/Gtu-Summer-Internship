"""
Python Data Analysis using Pandas

Dataset:
Global Data on Sustainable Energy (2000-2020)

Task:
1. Import dataset
2. Analyze dataset
3. Handle null values
4. Study relations between columns
5. Summarize work
"""

import pandas as pd

# ============================================================
# STEP 1 : Import Dataset
# ============================================================

df = pd.read_csv("global-data-on-sustainable-energy.csv")

print("="*60)
print("FIRST FIVE RECORDS")
print("="*60)
print(df.head())

# ============================================================
# STEP 2 : Dataset Information
# ============================================================

print("\nShape of Dataset:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

# ============================================================
# STEP 3 : Checking Null Values
# ============================================================

print("\nNull Values Before Cleaning")
print(df.isnull().sum())

# Numerical columns -> Median
# Categorical columns -> Mode

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nNull Values After Cleaning")
print(df.isnull().sum())

# ============================================================
# STEP 4 : General Analysis
# ============================================================

print("\nNumber of Rows:", len(df))
print("Number of Columns:", len(df.columns))

if "Country" in df.columns:
    print("Unique Countries:", df["Country"].nunique())

if "Year" in df.columns:
    print("Years Covered:", df["Year"].min(), "-", df["Year"].max())

# ============================================================
# STEP 5 : Relation Between Columns
# ===========================================================

print("\nCorrelation Matrix")
numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr())

# ============================================================
# STEP 6 : Pandas Operations
# ============================================================

print("\nFirst 10 Records")
print(df.head(10))

print("\nLast 5 Records")
print(df.tail())

print("\nSorted by Year")
if "Year" in df.columns:
    print(df.sort_values("Year").head())

print("\nAverage of Numerical Columns")
print(df.mean(numeric_only=True))

print("\nMaximum Values")
print(df.max(numeric_only=True))

print("\nMinimum Values")
print(df.min(numeric_only=True))

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

