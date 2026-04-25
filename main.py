# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 2. LOAD DATASET
# ==============================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ==============================
# 3. DATA CLEANING
# ==============================
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df = df.dropna()

# Convert Churn to numeric
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ==============================
# 4. FEATURE ENGINEERING
# ==============================
# Tenure already exists (months)

# Monthly Charges already exists

# Customer Lifetime Value (CLV)
df['CLV'] = df['MonthlyCharges'] * df['tenure']

# ==============================
# 5. BASIC OVERVIEW
# ==============================
print("Total Customers:", len(df))
print("Churn Rate:", df['Churn'].mean())

# ==============================
# 6. CHURN ANALYSIS
# ==============================
# Churn by Contract
contract_churn = df.groupby('Contract')['Churn'].mean()
print("\nChurn by Contract:\n", contract_churn)

# Churn by Internet Service
internet_churn = df.groupby('InternetService')['Churn'].mean()
print("\nChurn by Internet Service:\n", internet_churn)

# ==============================
# 7. RETENTION ANALYSIS
# ==============================
# Retention by tenure groups
df['TenureGroup'] = pd.cut(df['tenure'],
                          bins=[0, 12, 24, 48, 60, 72],
                          labels=['0-1yr','1-2yr','2-4yr','4-5yr','5-6yr'])

retention = df.groupby('TenureGroup')['Churn'].mean()
print("\nChurn by Tenure Group:\n", retention)

# ==============================
# 8. COHORT ANALYSIS (SIMPLIFIED)
# ==============================
# Since signup date not available, simulate cohort using tenure

cohort = df.groupby('tenure')['Churn'].mean()

# ==============================
# 9. CUSTOMER LIFETIME ANALYSIS
# ==============================
print("\nAverage CLV:", df['CLV'].mean())

# ==============================
# 10. VISUALIZATIONS
# ==============================

# --- Churn Distribution ---
plt.figure()
df['Churn'].value_counts().plot(kind='bar')
plt.title("Churn Distribution")
plt.xlabel("Churn (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# --- Churn by Contract ---
plt.figure()
contract_churn.plot(kind='bar')
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.show()

# --- Churn by Tenure Group ---
plt.figure()
retention.plot(kind='bar')
plt.title("Churn by Tenure Group")
plt.ylabel("Churn Rate")
plt.show()

# --- CLV Distribution ---
plt.figure()
df['CLV'].plot(kind='hist', bins=30)
plt.title("Customer Lifetime Value Distribution")
plt.xlabel("CLV")
plt.show()

# ==============================
# 11. KEY INSIGHTS (PRINT)
# ==============================
print("\n--- KEY INSIGHTS ---")
print("1. Customers with month-to-month contracts churn the most.")
print("2. New customers (low tenure) have higher churn.")
print("3. High CLV customers tend to stay longer.")
print("4. Retention improves after first year.")
