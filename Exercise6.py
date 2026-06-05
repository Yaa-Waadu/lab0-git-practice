import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# Recreate Dataset (Exercise 5)
# =====================================

np.random.seed(42)
n_students = 200

data = {
    "student_id": range(1000, 1000 + n_students),
    "major": np.random.choice(["CS", "Math", "Physics", "Biology"], n_students),
    "year": np.random.choice([1, 2, 3, 4], n_students),
    "exam_score": np.random.normal(75, 10, n_students).clip(0, 100),
    "assignments_completed": np.random.randint(0, 11, n_students),
    "hours_studied": np.random.normal(15, 5, n_students).clip(1, 40),
}

df = pd.DataFrame(data)

# Add missing values
df.loc[np.random.choice(n_students, 10), "exam_score"] = np.nan
df.loc[np.random.choice(n_students, 5), "hours_studied"] = np.nan

# Fill missing values
df["exam_score"] = df.groupby("major")["exam_score"].transform(
    lambda x: x.fillna(x.mean())
)

df["hours_studied"] = df.groupby("year")["hours_studied"].transform(
    lambda x: x.fillna(x.median())
)

# Create performance categories
df["performance"] = pd.cut(
    df["exam_score"],
    bins=[-np.inf, 70, 80, 90, np.inf],
    labels=["Needs Improvement", "Average", "Good", "Excellent"],
)

# =====================================
# Task 1: Distribution Visualization
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram with KDE
sns.histplot(df["exam_score"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Distribution of Exam Scores")
axes[0].set_xlabel("Exam Score")
axes[0].set_ylabel("Frequency")

# Box Plot
sns.boxplot(data=df, x="major", y="exam_score", ax=axes[1])
axes[1].set_title("Exam Scores by Major")
axes[1].set_xlabel("Major")
axes[1].set_ylabel("Exam Score")

plt.tight_layout()
plt.show()

# =====================================
# Task 2: Relationship Visualization
# =====================================

plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="hours_studied", y="exam_score", hue="major")

sns.regplot(data=df, x="hours_studied", y="exam_score", scatter=False, ci=None)

plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend(title="Major")
plt.tight_layout()
plt.show()

# =====================================
# Task 3: Advanced Dashboard
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Average Exam Score by Major
avg_scores = df.groupby("major")["exam_score"].mean().reset_index()

sns.barplot(data=avg_scores, x="major", y="exam_score", ax=axes[0, 0])

axes[0, 0].set_title("Average Exam Score by Major")
axes[0, 0].set_xlabel("Major")
axes[0, 0].set_ylabel("Average Score")

# 2. Number of Students by Year
sns.countplot(data=df, x="year", ax=axes[0, 1])

axes[0, 1].set_title("Number of Students by Year")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Count")

# 3. Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 0])

axes[1, 0].set_title("Correlation Matrix")

# 4. Violin Plot by Performance Category
sns.violinplot(
    data=df,
    x="performance",
    y="exam_score",
    order=["Needs Improvement", "Average", "Good", "Excellent"],
    ax=axes[1, 1],
)

axes[1, 1].set_title("Exam Score Distribution by Performance")
axes[1, 1].set_xlabel("Performance Category")
axes[1, 1].set_ylabel("Exam Score")

plt.tight_layout()
plt.show()
