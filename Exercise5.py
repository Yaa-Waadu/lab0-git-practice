import pandas as pd
import numpy as np

# Create sample dataset
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

# Introduce some NaN values
df.loc[np.random.choice(n_students, 10), "exam_score"] = np.nan
df.loc[np.random.choice(n_students, 5), "hours_studied"] = np.nan

# =========================
# Task 1: Data Cleaning and Exploration
# =========================

# Display basic information
print("Dataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe())

# Count missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing exam_score with mean score for the student's major
df["exam_score"] = df.groupby("major")["exam_score"].transform(
    lambda x: x.fillna(x.mean())
)

# Fill missing hours_studied with median for the student's year
df["hours_studied"] = df.groupby("year")["hours_studied"].transform(
    lambda x: x.fillna(x.median())
)

print("\nMissing Values After Filling:")
print(df.isnull().sum())

# =========================
# Task 2: Analysis
# =========================

# Average exam score by major
avg_score_by_major = df.groupby("major")["exam_score"].mean()
print("\nAverage Exam Score by Major:")
print(avg_score_by_major)

# Major with highest average exam score
best_major = avg_score_by_major.idxmax()
print("\nMajor with Highest Average Exam Score:")
print(best_major)

# Correlation between hours studied and exam score
correlation = df["hours_studied"].corr(df["exam_score"])
print("\nCorrelation between Hours Studied and Exam Score:")
print(correlation)

# Create performance categories
df["performance"] = pd.cut(
    df["exam_score"],
    bins=[-np.inf, 70, 80, 90, np.inf],
    labels=["Needs Improvement", "Average", "Good", "Excellent"],
)

print("\nPerformance Distribution:")
print(df["performance"].value_counts())

# =========================
# Task 3: Advanced Analysis
# =========================

# Statistics by major and year
summary = df.groupby(["major", "year"]).agg(
    num_students=("student_id", "count"),
    avg_exam_score=("exam_score", "mean"),
    avg_hours_studied=("hours_studied", "mean"),
)

print("\nMajor-Year Summary:")
print(summary)

# Top 5 students (including ties)
ranked_df = df.copy()
ranked_df["rank"] = ranked_df["exam_score"].rank(method="min", ascending=False)

top_students = ranked_df[ranked_df["rank"] <= 5].sort_values(
    by="exam_score", ascending=False
)

print("\nTop Students (Including Ties):")
print(top_students[["student_id", "major", "exam_score", "rank"]])

# Pivot table: average exam score by major and year
pivot_table = pd.pivot_table(
    df, values="exam_score", index="major", columns="year", aggfunc="mean"
)

print("\nPivot Table: Average Exam Score by Major and Year")
print(pivot_table)
