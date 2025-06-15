import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
df = pd.read_csv("data/train.csv")

# Select features and target
features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'GarageCars']
X = df[features]
y = df['SalePrice']

# Handle missing values
X = X.fillna(X.mean())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Model 1: Linear Regression
# ------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("=== Linear Regression ===")
print(f"RMSE: {lr_rmse:.2f}")
print(f"R^2: {lr_r2:.4f}")

# ------------------------
# Model 2: Random Forest
# ------------------------
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n=== Random Forest ===")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R^2: {rf_r2:.4f}")

# ------------------------
# Model 3: XGBoost
# ------------------------
xgb_model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\n=== XGBoost ===")
print(f"RMSE: {xgb_rmse:.2f}")
print(f"R^2: {xgb_r2:.4f}")

# ------------------------
# Save output images
# ------------------------
os.makedirs("images", exist_ok=True)

# Plot: Actual vs Predicted (Random Forest)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted SalePrice")
plt.tight_layout()
plt.savefig("images/rf_pred_vs_actual.png")

# Plot: Actual vs Predicted (Linear Regression)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, lr_pred, alpha=0.5, color='orange')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted SalePrice")
plt.tight_layout()
plt.savefig("images/lr_pred_vs_actual.png")

# Plot: Actual vs Predicted (XGBoost)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, xgb_pred, alpha=0.5, color='green')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("XGBoost: Actual vs Predicted SalePrice")
plt.tight_layout()
plt.savefig("images/xgb_pred_vs_actual.png")

# Plot: Feature Importance (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=features)
print(importances) 
importances.sort_values().plot(kind='barh')
plt.xlabel("Feature Importance")
plt.title("Random Forest: Feature Importance")
plt.xlim(0, max(importances) * 1.1)  
plt.tight_layout()
plt.savefig("images/rf_feature_importance.png")

