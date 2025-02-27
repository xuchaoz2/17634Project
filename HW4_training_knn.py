import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load training, validation, and test datasets
train_file = "data_train.csv"
val_file = "data_validation.csv"
test_file = "data_test.csv"

train_data = pd.read_csv(train_file)
val_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)

# Define features and target variable
features = ['Year', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4', 'nsmiles', 'passengers_log', 'route']
target = 'fare'

# Extract features and target
X_train, y_train = train_data[features], train_data[target]
X_val, y_val = val_data[features], val_data[target]
X_test, y_test = test_data[features], test_data[target]

# Standardize features for KNN (important for distance-based models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train a KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')  # Using distance-based weighting
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_val = knn_model.predict(X_val)
y_pred_test = knn_model.predict(X_test)

# Evaluate model performance
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

# Print results
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R² Score: {r2_test:.4f}")

# Make predictions using the trained model
y_pred_test = knn_model.predict(X_test)  # Replace with your model (Decision Tree, Random Forest, etc.)

# Compute residuals
residuals = y_test - y_pred_test

# Compute performance metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1️⃣ Scatter Plot: Predicted vs. Actual
axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='black')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 45-degree line
axes[0].set_xlabel('Actual Fare')
axes[0].set_ylabel('Predicted Fare')
axes[0].set_title(f'KNN: Predicted vs Actual (R² = {r2:.2f})')

# 2️⃣ Histogram: Residual Distribution
sns.histplot(residuals, bins=30, kde=True, color='blue', ax=axes[1])
axes[1].set_xlabel('Residuals (Errors)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'KNN: Residual Distribution (RMSE = {rmse:.2f})')

# 3️⃣ Scatter Plot: Residuals vs. Predicted
axes[2].scatter(y_pred_test, residuals, alpha=0.6, edgecolors='black')
axes[2].axhline(0, color='red', linestyle='dashed', lw=2)  # Horizontal line at 0
axes[2].set_xlabel('Predicted Fare')
axes[2].set_ylabel('Residuals (Errors)')
axes[2].set_title('KNN: Residuals vs Predicted')

# Display RMSE & R² at the bottom
fig.text(0.5, 0.01, f"KNN - RMSE_validation: {val_rmse:.2f}, RMSE_test: {test_rmse:.2f}, R²: {r2_test:.4f}", ha='center', fontsize=7)

# Adjust layout and show plots
plt.tight_layout()
plt.show()