import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''# Load the dataset
data = pd.read_csv('data_original.csv')

# 1. Histogram for the distribution of 'fare'
plt.figure(figsize=(8, 6))
sns.histplot(data['fare'], bins=30, kde=True, color='blue')
plt.title('Distribution of Fare', fontsize=16)
plt.xlabel('Fare', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Scatterplot of 'nsmiles' vs 'fare'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='nsmiles', y='fare', data=data, alpha=0.7, color='green')
plt.title('Relationship Between Fare and NSmiles', fontsize=16)
plt.xlabel('NSmiles', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 3. Scatterplot of 'fare' vs 'fare_low'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='fare', y='fare_low', data=data, alpha=0.7, color='green')
plt.title('Relationship Between Fare and Lowest Fare', fontsize=16)
plt.xlabel('Fare', fontsize=12)
plt.ylabel('Fare_Low', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()'''


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

'''# Load the dataset and limit to the first 50,000 rows
data = pd.read_csv('data_original.csv').head(50000)

# Create the 'route' variable by concatenating 'airport_1' and 'airport_2'
data['route'] = data['airport_1'] + '-' + data['airport_2']

# Define response variable and feature variables
response_variable = 'fare'
feature_variables = ['nsmiles', 'passengers', 'route']

# Extract features and target
X = data[feature_variables]
y = data[response_variable]

# One-hot encode the 'route' variable
X = pd.get_dummies(X, columns=['route'], drop_first=True)  # Convert categorical variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute R² score
r2 = r2_score(y_test, y_pred)

# Determine axis limits for equal scaling
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

# Plot predicted vs observed values with equal axis scaling
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)  # 45-degree reference line
plt.title(f'Predicted vs Observed Fare\nR² = {r2:.2f}', fontsize=16)
plt.xlabel('Observed Fare', fontsize=12)
plt.ylabel('Predicted Fare', fontsize=12)
plt.xlim(min_val, max_val)  # Set equal x-axis limits
plt.ylim(min_val, max_val)  # Set equal y-axis limits
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()'''

import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

file_path = "data_original.csv"  
data = pd.read_csv(file_path)
data = data.dropna()

# Create the 'route' variable by concatenating 'airport_1' and 'airport_2'
data['route'] = data['airport_1'] + '-' + data['airport_2']

# Define numerical and categorical features
numerical_features = ['Year', 'nsmiles', 'passengers', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
categorical_features = [
    'quarter', 'citymarketid_1', 'citymarketid_2', 'city1', 'city2',
    'airportid_1', 'airportid_2', 'airport_1', 'airport_2',
    'carrier_lg', 'carrier_low',
    'Geocoded_City1', 'Geocoded_City2', 'tbl1apk', 'route'
]

# Extract features and target variable
X = data[numerical_features + categorical_features]
y = data['fare']  # Response variable

# Apply Ordinal Encoding to categorical features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_features] = encoder.fit_transform(X[categorical_features])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bias-variance tradeoff: Accuracy vs Model Complexity
accuracy_scores = []

for i in range(1, len(X_train.columns) + 1):
    selected_features = X_train.columns[:i]  # Select first 'i' features
    X_train_iter = X_train[selected_features]
    X_test_iter = X_test[selected_features]

    model = LinearRegression()
    model.fit(X_train_iter, y_train)
    y_pred_iter = model.predict(X_test_iter)

    accuracy_scores.append(r2_score(y_test, y_pred_iter))  # Store R² score

# Plot accuracy vs model complexity
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(X_train.columns) + 1), accuracy_scores, marker='o', linestyle='-')
plt.title('Model Accuracy vs Number of Features', fontsize=16)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()