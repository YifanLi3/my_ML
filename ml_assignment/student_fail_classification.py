import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========== 1. Load Data ==========
df = pd.read_csv('data/student_fail.csv')

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
print(f"\n=== Label Unique Values: {df['fail'].unique()} ===")
print(df['fail'].value_counts())

# ========== 6. Split Features and Labels ==========
X = df.drop('fail', axis=1)
y = df['fail']

# ========== 7. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ========== 8. Preprocessing (Standardization) ==========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== 9. Train Model (Decision Tree Classifier) ==========
# Binary classification (0/1), using Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ========== 10. Model Evaluation ==========
y_pred = clf.predict(X_test)

print(f"\n=== Accuracy: {accuracy_score(y_test, y_pred):.4f} ===")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  TP={cm[1][1]} FP={cm[0][1]} FN={cm[1][0]} TN={cm[0][0]}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Pass(0)', 'Fail(1)']))

# ========== 11. Predict New Student ==========
new_student = pd.DataFrame({
    'homework': [55],
    'attend': [10],
    'study_hours': [3],
    'midterm': [45]
})
new_student_scaled = scaler.transform(new_student)
prediction = clf.predict(new_student_scaled)
print(f"\n=== New Student Prediction: {'Fail(1)' if prediction[0] == 1 else 'Pass(0)'} ===")
