import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

# ========== 1. Load Data ==========
df = pd.read_csv('data/mpg.csv')

# ========== 2. Data Exploration ==========
print("=== First 5 Rows ===")
print(df.head())

print("\n=== Data Structure ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe())

# ========== 3. Check Missing Values ==========
print("\n=== Missing Values ===")
print(df.isnull().sum())
df = df.dropna()

# ========== 4. Check Duplicates ==========
print(f"\n=== Duplicate Count: {df.duplicated().sum()} ===")
df = df.drop_duplicates()
print(f"Samples after dedup: {len(df)}")

# ========== 5. Split Features and Target ==========
X = df.drop('mpg', axis=1)
y = df['mpg']

# ========== 6. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ========== 7. Preprocessing (Standardization) ==========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== 8. Train Model (Linear Regression) ==========
# Target (mpg) is continuous, this is a regression problem
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nCoefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")

# ========== 9. Model Evaluation ==========
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== RMSE: {rmse:.4f} ===")
print(f"=== R² Score: {r2:.4f} ===")

# ========== 10. Predict New Car ==========
new_car = pd.DataFrame({
    'cylinders': [6],
    'displacement': [250.0],
    'weight': [3200.0],
    'acceleration': [15.0]
})
new_car_scaled = scaler.transform(new_car)
prediction = model.predict(new_car_scaled)
print(f"\n=== New Car Predicted MPG: {prediction[0]:.2f} ===")
