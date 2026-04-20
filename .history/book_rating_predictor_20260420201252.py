# ============================================================
#   BOOK RATING PREDICTOR - FINAL IMPROVED ML PROJECT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

print("=" * 55)
print("   BOOK RATING PREDICTOR - ML Project")
print("=" * 55)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/6] Loading dataset...")

df = pd.read_csv("Books.csv", on_bad_lines="skip", encoding="latin-1")

print(f"Loaded {len(df)} rows")
print("Columns:", df.columns.tolist())


# ─────────────────────────────────────────────
# STEP 2: CLEAN DATA (FIX MAIN ISSUE)
# ─────────────────────────────────────────────
print("\n[2/6] Cleaning data...")

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert rating column properly (VERY IMPORTANT FIX)
df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["average_rating"])

# Convert numeric columns
df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
df["pages"] = pd.to_numeric(df["pages"], errors="coerce")

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Create label
df["label"] = df["average_rating"].apply(lambda x: 1 if x >= 4 else 0)

print("Good books:", df["label"].sum())
print("Not good books:", len(df) - df["label"].sum())


# ─────────────────────────────────────────────
# STEP 3: FEATURES (IMPROVED)
# ─────────────────────────────────────────────
print("\n[3/6] Preparing features...")

features = ["average_rating", "ratings_count", "pages"]

X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ─────────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[4/6] Training models...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# Select best model
if rf_acc >= lr_acc:
    best_model = rf
    best_pred = rf_pred
    best_name = "Random Forest"
    best_acc = rf_acc
else:
    best_model = lr
    best_pred = lr_pred
    best_name = "Logistic Regression"
    best_acc = lr_acc

print(f"\nBest Model: {best_name} ({best_acc*100:.2f}%)")

print("\nClassification Report:\n")
print(classification_report(y_test, best_pred))


# ─────────────────────────────────────────────
# STEP 5: VISUALIZATION
# ─────────────────────────────────────────────
print("\n[5/6] Creating charts...")

plt.figure()

# Rating distribution
plt.hist(df["average_rating"], bins=30)
plt.axvline(x=4, linestyle="--")
plt.title("Rating Distribution")
plt.savefig("rating_distribution.png")

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("confusion_matrix.png")

print("Charts saved!")


# ─────────────────────────────────────────────
# STEP 6: PREDICT NEW BOOK
# ─────────────────────────────────────────────
print("\n[6/6] Prediction...")

sample = pd.DataFrame([[4.3, 50000, 300]], columns=features)

if best_name == "Logistic Regression":
    sample_scaled = scaler.transform(sample)
    pred = best_model.predict(sample_scaled)[0]
    prob = best_model.predict_proba(sample_scaled)[0]
else:
    pred = best_model.predict(sample)[0]
    prob = best_model.predict_proba(sample)[0]

print("\nPrediction:", "GOOD BOOK ✅" if pred == 1 else "NOT GOOD ❌")
print("Confidence:", round(max(prob)*100, 2), "%")

print("\nPROJECT COMPLETE 🚀")