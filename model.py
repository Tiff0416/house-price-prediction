import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.4f}")

# Save prediction plot
os.makedirs("images", exist_ok=True)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted SalePrice")
plt.savefig("images/pred_vs_actual.png")

# Feature importance
importances = pd.Series(model.feature_importances_, index=features)
importances = importances.sort_values()

plt.clf() 
importances.plot(kind='barh')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()

