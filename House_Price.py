# ==========================================================
# House Price Prediction using Machine Learning
# Author : Laiba Wajid
# Internship Task 6
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("House Price Prediction Dataset.csv")

print("="*60)
print("FIRST FIVE ROWS")
print("="*60)
print(df.head())

print("\n")

print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(df.info())

print("\n")

print("="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

# ==========================================================
# Data Preprocessing
# ==========================================================

df.dropna(inplace=True)

encoder = LabelEncoder()

df["Location"] = encoder.fit_transform(df["Location"])
df["Condition"] = encoder.fit_transform(df["Condition"])
df["Garage"] = encoder.fit_transform(df["Garage"])

# ==========================================================
# Features and Target
# ==========================================================

X = df.drop(["Id", "Price"], axis=1)
y = df["Price"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Train Model
# ==========================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================================
# Prediction
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# Model Evaluation
# ==========================================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ==========================================================
# Compare Actual and Predicted
# ==========================================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions.astype(int)
})

print("\n")
print("="*60)
print("ACTUAL VS PREDICTED")
print("="*60)
print(results.head(10))

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--'
)

plt.grid(True)s

plt.savefig("House_Price_Prediction_Result.png")

plt.show()

print("\nGraph saved as: House_Price_Prediction_Result.png")