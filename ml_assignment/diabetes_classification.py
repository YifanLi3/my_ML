import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. Load Data ==========
df = pd.read_csv('data/diabetes.csv')

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

# ========== 5. Check Label Distribution ==========
print(f"\n=== Label Unique Values: {df['diabetes'].unique()} ===")
print(df['diabetes'].value_counts())

# ========== 6. Outlier Analysis ==========
# Physiological metrics like glucose, bp, bmi should not be 0
zero_cols = ['glucose', 'bp', 'skin', 'insulin', 'bmi']
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"{col} has {zero_count} zero values, replacing with median")
        df[col] = df[col].replace(0, df[col].median())

# ========== 7. Split Features and Labels ==========
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# ========== 8. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ========== 9. Preprocessing (Standardization) ==========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== 10. Train Model (Decision Tree Classifier) ==========
# Binary classification (0/1), using Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ========== 11. Model Evaluation ==========
y_pred = clf.predict(X_test)

print(f"\n=== Accuracy: {accuracy_score(y_test, y_pred):.4f} ===")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Diabetic(1)']))

# ========== 12. Predict New Patient ==========
new_patient = pd.DataFrame({
    'pregnancies': [5],
    'glucose': [140],
    'bp': [80],
    'skin': [30],
    'insulin': [200],
    'bmi': [32.0],
    'age': [45]
})
new_patient_scaled = scaler.transform(new_patient)
prediction = clf.predict(new_patient_scaled)
print(f"\n=== New Patient Prediction: {'Diabetic(1)' if prediction[0] == 1 else 'Normal(0)'} ===")
