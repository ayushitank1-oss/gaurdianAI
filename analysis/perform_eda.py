import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('../data/insurance_fraud_data.csv')
print(f"Dataset Shape: {df.shape}")

# 2. Missing values check
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("No missing values detected.")
else:
    print("Missing values found:")
    print(null_counts[null_counts > 0])

# 3. Outlier Detection and Handling
# Focus on financial columns
fin_cols = ['annual_income', 'total_claim', 'injury_claim', 'annual premium', 'vehicle_price']
print("\nDescriptive statistics for financial columns:")
print(df[fin_cols].describe())

def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Clipping outliers
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

df = handle_outliers(df, fin_cols)
print("\nOutliers handled by clipping/capping.")

# 4. EDA Visuals (Saving to separate files for proof of task)
sns.set_style('whitegrid')

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='fraud reported', data=df, palette='viridis')
plt.title('Distribution of Fraud Reported')
plt.savefig('eda_fraud_dist.png')
plt.close()

# Correlation Matrix (Numerical only)
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('eda_correlation.png')
plt.close()

# Key Features vs Target
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='fraud reported', y='form defects', data=df)
plt.title('Form Defects vs Fraud')

plt.subplot(1, 2, 2)
sns.boxplot(x='fraud reported', y='safety_rating', data=df)
plt.title('Safety Rating vs Fraud')
plt.savefig('eda_key_features.png')
plt.close()

print("\nVisual EDA artifacts saved: eda_fraud_dist.png, eda_correlation.png, eda_key_features.png")

# 5. Saving Cleaned Data
df.to_csv('../data/insurance_fraud_data_cleaned.csv', index=False)
print("Cleaned dataset saved as ../data/insurance_fraud_data_cleaned.csv")
