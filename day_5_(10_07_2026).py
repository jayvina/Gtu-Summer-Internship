import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Dataset:

df = pd.read_csv("global-data-on-sustainable-energy.csv")

# ---------------- Basic Information ----------------
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# ---------------- Null Values ----------------
print(df.isnull().sum())

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print(df.isnull().sum())

# ---------------- Feature Engineering ----------------
# Create a new feature if columns exist
if "Renewable energy share in the total final energy consumption (%)" in df.columns and \
   "Electricity from fossil fuels (TWh)" in df.columns:
    df["Green_Energy_Index"] = (
        df["Renewable energy share in the total final energy consumption (%)"] /
        (df["Electricity from fossil fuels (TWh)"] + 1)
    )

# ---------------- Analysis ----------------
# GDP generally increases with electricity access.
# Renewable energy share is expected to reduce fossil fuel dependency.
# Higher CO2 emissions often indicate greater fossil fuel usage.
# Population growth generally increases energy demand.

print(df.corr(numeric_only=True))

# ---------------- Encoding ----------------
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ---------------- Scaling ----------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------- Visualization ----------------

sns.histplot(df[num_cols[0]], kde=True)
plt.title("Distribution")
plt.show()

sns.boxplot(x=df[num_cols[1]])
plt.title("Box Plot")
plt.show()

sns.heatmap(df[num_cols].corr(), cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
plt.title("Scatter Plot")
plt.show()

# ---------------- Insights ----------------
print("""
Insights
--------
1. Filled numerical null values with median because it is robust to outliers.
2. Filled categorical null values with mode because it preserves the most common category.
3. LabelEncoder converted text columns into numeric values for ML compatibility.
4. StandardScaler standardized numerical columns so features are on the same scale.
5. Feature engineering created Green_Energy_Index to compare renewable energy with fossil fuel usage.
6. Heatmap helped identify relationships between numerical features.
7. Histogram showed data distribution.
8. Box plot detected outliers.
9. Scatter plot visualized relationships between two numerical variables.
10. Dataset is now cleaned and ready for machine learning or further analysis.
""")
