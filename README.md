import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = "/content/Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv"
df = pd.read_csv(file_path)

# -------------------------------
# 1. Basic Info
# -------------------------------
print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n", df.dtypes)

# -------------------------------
# 2. Summary Statistics
# -------------------------------
print("\nSummary statistics:\n", df.describe())

# -------------------------------
# 3. Missing Values
# -------------------------------
print("\nMissing values:\n", df.isnull().sum())

# -------------------------------
# 4. Duplicate Rows
# -------------------------------
print("\nDuplicate rows:", df.duplicated().sum())

# -------------------------------
# 5. Unique Values
# -------------------------------
for col in df.columns:
    print(f"\nUnique values in {col}: {df[col].nunique()}")

# -------------------------------
# 6. Value Counts (Categorical)
# -------------------------------
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\nValue counts for {col}:\n")
    print(df[col].value_counts())

# -------------------------------
# 7. Correlation Matrix
# -------------------------------
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

print("\nCorrelation Matrix:\n", correlation)

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
plt.imshow(correlation)
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 8. Distribution Plots
# -------------------------------
numeric_cols = numeric_df.columns

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# 9. Boxplots (Outliers)
# -------------------------------
for col in numeric_cols:
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.show()

# -------------------------------
# 10. Pairwise Relationships
# -------------------------------
pd.plotting.scatter_matrix(numeric_df, figsize=(12, 10))
plt.show()
