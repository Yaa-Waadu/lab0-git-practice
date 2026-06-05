"""
Exercise 4: NumPy Array Operations
Complete the following tasks using NumPy.
"""

import numpy as np

"""
Exercise 4: NumPy Array Operations
"""

# Task 1: Create a 5x5 matrix where border elements are 1 and interior is 0
# (5 points)

matrix = np.ones((5, 5), dtype=int)
matrix[1:-1, 1:-1] = 0

print("Task 1: Border Matrix")
print(matrix)


# Task 2: Normalize a random array
# (5 points)

np.random.seed(42)
random_data = np.random.randn(100, 3)

# Normalize each column
mean = np.mean(random_data, axis=0)
std = np.std(random_data, axis=0)

normalized_data = (random_data - mean) / std

print("\nTask 2: Normalized Data")
print("Column Means:", np.mean(normalized_data, axis=0))
print("Column Std Devs:", np.std(normalized_data, axis=0))


# Task 3: Implement linear regression solution using normal equation
# (10 points)

# Given X (features) and y (target), compute theta
# theta = (X^T X)^(-1) X^T y

X = np.random.randn(50, 3)
true_theta = np.array([2.5, -1.2, 3.7])
y = X @ true_theta + np.random.randn(50) * 0.1

# Calculate theta_hat using the normal equation
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

print("\nTask 3: Linear Regression using Normal Equation")
print("Estimated coefficients (theta_hat):")
print(theta_hat)

print("\nTrue coefficients (true_theta):")
print(true_theta)

print("\nDifference:")
print(theta_hat - true_theta)
