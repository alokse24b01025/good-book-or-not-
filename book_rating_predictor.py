# ============================================================
#   BOOK RATING PREDICTOR - FINAL ML PROJECT (IMPROVED)
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 60)
print("   BOOK RATING PREDICTOR - FINAL ML PROJECT")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
print("\n[1/6] Loading dataset...")

df = pd.read_csv("Books.csv", encoding="latin-1", on_bad_lines="skip")

# ------------------------------------------------------------
# STEP 2: CLEAN DATA
# ------------------------------------------------------------
print("\n[2/6] Cleaning data...")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert rating properly
df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
df = df.dropna(subset=["average_rating"])

# Convert numeric
df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
df["pages"] = pd.to_numeric(df["pages"], errors="coerce")

# Fill missing
df.fillna({
    "ratings_count": 0,
    "pages": df["pages"].median(),
    "description": ""
}, inplace=True)

# Target
df["label"] = (df["average_rating"] >= 4).astype(int)

print("Good books:", df["label"].sum())
print("Not good books:", len(df) - df["label"].sum())


# ------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING (FIXED)
# ------------------------------------------------------------
print("\n[3/6] Feature Engineering...")

# ✅ IMPORTANT: include rating now
X_num = df[["average_rating", "ratings_count", "pages"]]

# Text feature
tfidf = TfidfVectorizer(max_features=300, stop_words="english")
X_text = tfidf.fit_transform(df["description"])

# Combine
from scipy.sparse import hstack
X = hstack([X_text, X_num])

y = df["label"]


# ------------------------------------------------------------
# STEP 4: TRAIN MODEL (BALANCED)
# ------------------------------------------------------------
print("\n[4/6] Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Balanced model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# ------------------------------------------------------------
# STEP 5: PREDICT NEW BOOK (SMART)
# ------------------------------------------------------------
print("\n[5/6] Predict new book...")

rating = float(input("Enter rating (0-5): "))
reviews = int(input("Enter reviews: "))
pages = int(input("Enter pages: "))
description = input("Enter book description: ")

# Convert input
text_vec = tfidf.transform([description])
num_vec = np.array([[rating, reviews, pages]])

from scipy.sparse import hstack
final_input = hstack([text_vec, num_vec])

prediction = model.predict(final_input)[0]
probability = model.predict_proba(final_input)[0]

# ✅ Smart override (important)
if rating < 3:
    prediction = 0

print("\nPrediction:",
      "GOOD BOOK ✅" if prediction == 1 else "NOT GOOD ❌")

print("Confidence:", round(max(probability)*100, 2), "%")

print("\nPROJECT COMPLETE 🚀")