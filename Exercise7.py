import numpy as np
import pandas as pd

# Fix for Matplotlib GUI issues
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# Generate Synthetic Customer Data
# =====================================

np.random.seed(42)
n_customers = 500

ages = np.random.randint(18, 70, n_customers)
income = np.random.normal(50000, 20000, n_customers).clip(15000, 150000)
purchase_freq = np.random.poisson(5, n_customers)
avg_purchase_value = np.random.normal(100, 30, n_customers).clip(10, 500)

customers = pd.DataFrame(
    {
        "age": ages,
        "income": income,
        "purchase_frequency": purchase_freq,
        "avg_purchase_value": avg_purchase_value,
    }
)

# =====================================
# Calculate Customer Lifetime Value
# =====================================

max_frequency = customers["purchase_frequency"].max()

customers["churn_risk"] = 1 - customers["purchase_frequency"] / max_frequency

customers["CLV"] = (
    customers["purchase_frequency"]
    * customers["avg_purchase_value"]
    * (1 + customers["churn_risk"])
)

# =====================================
# Create Age Groups
# =====================================

customers["age_group"] = pd.cut(
    customers["age"],
    bins=[18, 25, 35, 50, 70],
    labels=["18-25", "26-35", "36-50", "51-70"],
    include_lowest=True,
)

# =====================================
# Age Group Analysis
# =====================================

age_summary = customers.groupby("age_group", observed=False).agg(
    num_customers=("age", "count"),
    avg_income=("income", "mean"),
    avg_CLV=("CLV", "mean"),
    total_CLV=("CLV", "sum"),
)

print("\nAGE GROUP SUMMARY")
print(age_summary)

# =====================================
# Top 10% Customers by CLV
# =====================================

clv_threshold = customers["CLV"].quantile(0.90)

top_customers = customers[customers["CLV"] >= clv_threshold].sort_values(
    by="CLV", ascending=False
)

print("\nTOP 10% CUSTOMERS BY CLV")
print(top_customers.head(10))

print(f"\nNumber of Top Customers: {len(top_customers)}")

# =====================================
# Visualization 1:
# Income vs CLV
# =====================================

plt.figure(figsize=(10, 6))

sns.scatterplot(data=customers, x="income", y="CLV", hue="age_group", alpha=0.7)

plt.title("Income vs Customer Lifetime Value")
plt.xlabel("Income ($)")
plt.ylabel("CLV")
plt.tight_layout()

plt.savefig("income_vs_clv.png")
plt.close()

# =====================================
# Visualization 2:
# Average CLV by Age Group
# =====================================

plt.figure(figsize=(8, 5))

avg_clv = customers.groupby("age_group", observed=False)["CLV"].mean().reset_index()

sns.barplot(data=avg_clv, x="age_group", y="CLV")

plt.title("Average CLV by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average CLV")
plt.tight_layout()

plt.savefig("avg_clv_by_age_group.png")
plt.close()

# =====================================
# Visualization 3:
# Correlation Heatmap
# =====================================

plt.figure(figsize=(8, 6))

numeric_cols = customers.select_dtypes(include=np.number)

sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig("correlation_heatmap.png")
plt.close()

# =====================================
# Analysis Summary
# =====================================

print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print("""
Key Findings:

1. Customer Lifetime Value (CLV) is primarily driven by
   purchase frequency and average purchase value.

2. Different age groups contribute different levels of
   revenue potential.

3. The top 10% of customers represent the highest-value
   segment and should be targeted for retention efforts.

4. Customers with higher purchase frequencies tend to
   have lower churn risk and higher CLV.

Recommendations:

• Create loyalty programs for high-CLV customers.
• Develop targeted marketing campaigns for the strongest
  age-group segments.
• Encourage repeat purchases through discounts and rewards.
• Monitor high-churn-risk customers and re-engage them
  with personalized offers.
""")

print("\nPlots saved successfully:")
print("- income_vs_clv.png")
print("- avg_clv_by_age_group.png")
print("- correlation_heatmap.png")
